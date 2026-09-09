# C0 — Dependency map: `mgl_air_backend.cpp` & `mgl_render.cpp`

> Track **C0** was docs-only; **C1** started monolith knives (IntegerReadback out).
> Snapshot: `main` @ C1 format-class PSO strip (~13.8k air / ~19.6k render LOC). Re-measure with `wc -l` after splits.
> Purpose: make include / caller / domain boundaries visible before any TU knife.

---

## 0. Why these two

| TU | ~LOC | Role | Risk if sink blindly |
|----|-----:|------|----------------------|
| `MGL/src/mgl_air_backend.cpp` | ~13762 | GLSL AST → LLVM AIR → `.metallib` | Mixes expr emit, stage ABI, legacy rewrite, reflect helpers; **type (C1b) + resource (C1c) + math (C1d) + VarSym (C1e) + matrix builtins (C1f) + stmt emit (C1g) extracted** |
| `MGL/src/mgl_render.cpp` | ~19484 | Metal-cpp runtime + ~800 `mglRender*` C ABI helpers | Catch-all for plans that belong in domain files (`mgl_buffer_plan`, tess, readback, binding, …) |

Policy (OBJC TODO / ARCH): **do not grow these**; new sinks land in domain TUs.

---

## 1. `mgl_air_backend.cpp`

### 1.1 Includes (project)

```mermaid
flowchart LR
  subgraph airTU["mgl_air_backend.cpp"]
    CG[Codegen / emit*]
    ASM[Module assembly + metallib]
    LEG[Legacy GLSL wiring]
  end

  subgraph frontend["GLSL / IR frontend"]
    AST["mgl_glsl_ast.h"]
    PAR["mgl_glsl_parser.h"]
    SEM["mgl_glsl_sema.h"]
    IR["mgl_ir.h"]
    FS["mgl_frontend_session.h"]
    LEGC["mgl_legacy_compat.h"]
  end

  subgraph abi["Stage / slot ABI"]
    GS["mgl_air_gs_abi.h"]
    TESS["mgl_air_tess_abi.h"]
    SLOTS["mgl_buffer_slots.h"]
    SABI["mgl_shader_abi.h"]
    LIM["glm_limits.h"]
  end

  subgraph out["Emit / reflect"]
    ML["mgl_metallib_writer.h"]
    REF["mgl_air_reflect.h"]
    ENV["mgl_env_flag.h"]
  end

  subgraph llvm["LLVM"]
    LLVMIR["IR / BitcodeWriter / Passes"]
  end

  frontend --> airTU
  abi --> airTU
  out --> airTU
  llvm --> airTU
```

Also pulls standard C++ STL (`map` / `vector` / `string` / …).

### 1.2 Internal domains (section banners)

Anonymous `namespace { … }` holds almost all helpers; C API is `extern "C"` after close.

| Approx lines | Banner / domain |
|-------------:|-----------------|
| ~87–185 | GS AST→ABI map; C1b/C1c/C1d/C1e/C1f/C1g facades; `storeStageOut` |
| ~~bootstrap + type helpers~~ | **C1b extracted** → `mgl_air_type.{h,cpp}` + `mgl_air_codegen.h` (`MType`/`Codegen`/carriers/LLVM/mangle/`typeFromIR`) |
| ~~resource collection~~ | **C1c extracted** → `mgl_air_resource.{h,cpp}` (`uniformBlock*` / `collectUniforms` / opaque leaves / sampler path) |
| ~162–… | **expression codegen** (`findSymbol` / swizzle / … → emitExpr) |
| ~~matrix builtins~~ | **C1f extracted** → `mgl_air_matrix.{h,cpp}` (`emitMatrixBuiltin` / `emitMatrixBinOp` + det helpers; thin `AirMatrixDeps` facade); emitExpr still deferred |
| … | **uniform-block member chains** / related stores |
| ~~math builtins~~ | **C1d extracted** → `mgl_air_math.{h,cpp}` (`emitMathBuiltin` + float-intrinsic helpers; thin `AirMathDeps` facade) |
| ~~statements~~ | **C1g extracted** → `mgl_air_stmt.{h,cpp}` (`emitStmt` / `emitCompound` + break/continue scan; thin `AirStmtDeps` facade); emitExpr still deferred |
| … | **AIR metadata** (`addModuleFlags`) |
| ~~module-assembly VarSym classify/location~~ | **C1e extracted** → `mgl_air_varsym.{h,cpp}` (`collectStageVarSyms` / `assignStageVarSymLocations` / `varyingLocationSpan` / attrib / stride); residual module assembly + compileGLSLImpl remain |
| … | **legacy GLSL wiring** + `compileGLSLImpl` |
| …–EOF | **exported C ABI** (compile / reflect / interface check) |

### 1.3 Exported C API (`mgl_shader_abi.h`)

| Symbol | Role |
|--------|------|
| `mglShaderCompileGLSL` | Stage metallib compile |
| `mglShaderCompileGLSLCapture` | VS XFB capture variant |
| `mglShaderCompileGLSLTessCapture` | VS tess-capture variant |
| `mglShaderCompileGLSLCullDistanceCapture` | VS cull-distance capture |
| `mglAirReflectGLSLStageInfo` | Stage metadata fill |
| `mglAirCompileGLSLWithReflectInfo(Ex)` | Compile + reflect lists + stage info |
| `mglAirCompileGLSLWithReflect` | Compile + reflect |
| `mglShaderFree` | Free metallib bytes |
| `mglShaderInterfaceCheck` | VS/FS link interface |
| `mglShaderTessInterfaceCheck` | TCS/TES link interface |

Pipeline (conceptual):

```text
GLSL source
  → mglFrontendRewriteLegacy (optional)
  → FrontendSession (parse + sema → TU + IR)
  → compileGLSLImpl (emitExpr/Stmt → LLVM Module + air.* meta)
  → metallib writer → bytes
  → (optional) mglAirReflect* / stage info
```

### 1.4 Callers / siblings (do not edit in C0)

```mermaid
flowchart TB
  CA["mgl_compile_artifact.c"] -->|mglAirCompileGLSL*| AIR["mgl_air_backend.cpp"]
  PROG["program.c"] -->|compile / interface| AIR
  RP["MGLRenderer+RenderPass.m"] -->|sparse ABI refs| AIR
  REF["mgl_air_reflect.c"] -.->|contract comments / shared layout| AIR
  TEX["MGLRenderer+Texture.m"] -.->|slot contract comments| AIR
  ABI["mgl_shader_abi.h"] --- AIR
  LOADER["mgl_air_loader.cpp"] -->|consumes metallib at runtime| MTL[Metal library/PSO]
  AIR -->|bytes| CA
  CA --> PROG
```

`mgl_air_loader.*` is the **runtime** loader (not this TU); keep split when planning knives.

---

## 2. `mgl_render.cpp`

### 2.1 Includes (project)

```mermaid
flowchart LR
  subgraph renderTU["mgl_render.cpp"]
    R[Metal-cpp Renderer / owners]
    H[mglRender* helpers]
  end

  METAL["mgl_metal.h"] --> renderTU
  RH["mgl_render.h"] --> renderTU
  BE["mgl_renderer_backend.h"] --> renderTU
  LOADER["mgl_air_loader.h"] --> renderTU
  TESSABI["mgl_air_tess_abi.h"] --> renderTU
  AUX["mgl_aux_assets.h"] --> renderTU
  CPC["mgl_compute_pipeline_cache.h"] --> renderTU
  ENV["mgl_env_flag.h"] --> renderTU
  PREF["mgl_program_reflection.h"] --> renderTU
  TB["mgl_types_buffer/texture/program/state/sync.h"] --> renderTU
  CTX["glm_context.h / glm_limits.h"] --> renderTU
  CAP["mgl_capability.h"] --> renderTU
  SYNC["mgl_sync.h"] --> renderTU
  SABI["mgl_shader_abi.h"] --> renderTU
  SLOTS["mgl_buffer_slots.h"] --> renderTU
  BPLAN["mgl_buffer_plan.h"] --> renderTU
  TDOM["mgl_tess_domain.h"] --> renderTU
```

ObjC runtime / Mach / Block headers are also included for Metal object class probes.

### 2.2 Domain clusters (by `mglRender*` line bands)

~805 exported `mglRender*` symbols live in this one TU. Coarse map for future split (not a prescription of PR order):

| Approx lines | Domain cluster | Examples |
|-------------:|----------------|----------|
| ~100–1470 | Metal-cpp internals / pass & PSO state types | `namespace mgl`, anonymous helpers |
| ~1448–1600 | Init / capability / AIR load entry | `mglRenderInit`, `LoadAIRMainFunction` |
| ~1600–2900 | Buffer storage / CoW / dirty / map / flush | `BindBufferStorage`, `SnapshotShared*` |
| ~2896–3280 | Program bind / sync / flush / query | `BindAIRProgram`, timer/sample query |
| ~3280–3800 | Buffer/texture create & views | `CreateTexture*`, `CreateBuffer*` |
| ~3808–6700 | Texture upload / format / readback copy | `TextureSubUploadPlan`, `Copy*ToGL` |
| ~6638–7200 | Stage binding / tess factor helpers | `EncodeStageBindingCopyBacks`, tess factor |
| ~~7238–7549~~ | ~~Integer readback classify~~ | **C1 extracted** → `mgl_readback_policy.{h,c}` (`Convert` + `Source`/`Packed`/`Classify`) |
| ~~CopyRows / depth / GetTexImagePlan / MSAA stride~~ | ~~Y-flip / depth pack / plan~~ | **C1 extracted** → same TU; Metal `EncodeMultisampleResolve*` residual in monolith |
| ~~7550–~8038 slot/sampler/stage/plain-uniform~~ | ~~Binding policy~~ | **C1 extracted** → `mgl_binding_policy.{h,c}` (O3.3) |
| ~~7221–~7290 + WritableStorage~~ | ~~stage-bind helpers~~ | **C1 extracted** → `mgl_binding_stage.{h,c}` (O3.3 residual) |
| ~~NeedsExplicitTopology–DrawModeFullyCulled + packed-DS/default-depth~~ | ~~format-class PSO~~ | **C1 extracted** → `mgl_pso_format_class.{h,c}` (O3.2); residual generatePipeline apply / BindingState |
| ~7740+ (post-O3.2) | Texture/integer/format tables residual | integer map, GLSL type names, … |
| ~9400–11200 | Format tables / clear mask / UBO pack | pixel-format class, plain-uniform pack |
| ~11214–12000 | Index expand / gather / blit plan | fan/strip/quad expand, `BlitFramebufferPlan` |
| ~12001–13600 | Tess / XFB size / texture swizzle bake | `SeedTessDomain`, swizzle upload bake |
| ~13597–15055 | Sampler / depth-stencil create + aux PSO | `CreateSampler*`, aux library cache |
| ~15056–17045 | Binding-state apply / masks | texture/buffer slot record |
| ~17046–20265 | Command recovery / encoder owners | recovery snapshot / reset |
| ~20266–EOF | Batch replay draw cmds | `mglRenderReplayBatchDraws` |

Anonymous `namespace` reopen points (~112, ~3004, ~15056, ~15537, ~17046, ~20266) roughly track these clusters.

### 2.3 Callers (include of `mgl_render.h`)

~50 TUs include `mgl_render.h`. Representative edges:

```mermaid
flowchart TB
  RH["mgl_render.h / mgl_render.cpp"]

  subgraph draw["Draw / tess / GS"]
    DI["mgl_draw_issue.cpp"]
    DG["mgl_draw_gs.cpp"]
    DT["mgl_draw_tess.cpp"]
    DC["mgl_draw_cull.cpp"]
    DE["mgl_draw_encode.m"]
  end

  subgraph batch["Batch encode ports"]
    BF["mgl_batch_flush_restore_encode.m"]
    BD["mgl_batch_dyn_bind_encode.m"]
    BB["MGLRenderer+Batch.m"]
  end

  subgraph objc["Thick ObjC categories"]
    TEX["+Texture.m"]
    RP["+RenderPass.m"]
    BL["+Blit.m"]
    BS["+BindingState.m"]
    BUF["+Buffer.m"]
  end

  subgraph core["GL core / backend"]
    BUFSC["buffers.c"]
    BE["mgl_renderer_backend.cpp"]
    RB["mgl_readback.m"]
  end

  draw --> RH
  batch --> RH
  objc --> RH
  core --> RH
```

Full include list is discoverable with:

```bash
rg -l '#include "mgl_render.h"' MGL/
```

### 2.4 Already-extracted neighbors (do not re-merge)

Prefer extending these instead of growing `mgl_render.cpp`:

- `mgl_buffer_plan.*`, `mgl_render_pass_plan.*`, `mgl_tess_domain.*`
- `mgl_readback_policy.*` (**C1** — IntegerReadback + Y-flip/depth/GetTexImagePlan/MSAA stride)
- `mgl_binding_policy.*` (**C1 / O3.3** — slot/sampler/stage/plain-uniform)
- `mgl_binding_stage.*` (**C1 / O3.3 residual** — stage UBO/SSBO + attrib + fallback-slot/present-mask/inline + map-count/post-mtl usable plans + helpers)
- `mgl_binding_texture.*` (**C1 / O3.3 residual** — sampled/storage/depth-recover/Y-flip RT/sampler-materialize/diag/apply-mask/FINAL ports + image-view helpers; log ports in `mgl_binding_texture_log.m` freeze/shrink)
- `mgl_pso_format_class.*` (**C1 / O3.2** — topology / format-class / blend·stencil·cull / viewport)
- `mgl_air_type.*` + `mgl_air_codegen.h` (**C1b** — MType / type helpers; not emitExpr)
- `mgl_air_resource.*` (**C1c** — uniform/opaque resource collection)
- `mgl_air_math.*` (**C1d** — math/pack/bitfield builtins; not emitExpr/matrix)
- `mgl_air_varsym.*` (**C1e** — VarSym stage classify / location assign / stride; not emitExpr/matrix)
- `mgl_air_matrix.*` (**C1f** — matrix builtins/binops; not whole emitExpr)
- `mgl_air_stmt.*` (**C1g** — emitStmt/compound; not whole emitExpr)
- `mgl_draw_{issue,gs,tess,cull,gs_metal}.*`
- `mgl_batch_{path,hazard,replay,restore,issue,rt_mark}.*`
- `mgl_compute_pipeline_cache.*`, `mgl_renderer_backend.*`

---

## 3. Cross-TU relationship (high level)

```mermaid
flowchart LR
  GLSL[GLSL source] --> AIR[mgl_air_backend.cpp]
  AIR -->|metallib bytes| ART[compile artifact / Program]
  ART --> LOAD[mgl_air_loader]
  LOAD --> RENDER[mgl_render.cpp PSO / bind]
  DRAW[mgl_draw_* / batch_*] --> RENDER
  OBJC[thin ObjC ports] --> DRAW
  OBJC --> RENDER
```

`mgl_air_backend.cpp` and `mgl_render.cpp` share **ABI headers** (`mgl_shader_abi`, tess/GS ABI, buffer slots) but should **not** call into each other directly; the seam is metallib bytes + stage info + loader.


---

## 4. C1 knife log — readback_policy (O4.1)

Chose **render readback policy → `mgl_readback_policy.*`** (DXMT / O4.1) over air type/expr (**C1b**): pure `extern "C"` helpers with no `Codegen` / Metal-cpp owner coupling; air type+expr is tangled through `emitExpr` / `MType` across multi-kLOC.

### 4.1 First strip — IntegerReadback

| Item | Detail |
|------|--------|
| Moved | `mglRenderConvertIntegerReadback`, `mglRenderIntegerReadbackSourceClassify`, `mglRenderIntegerReadbackPackedTypeClassify`, `mglRenderIntegerReadbackClassify` |
| New files | `MGL/include/mgl_readback_policy.h`, `MGL/src/mgl_readback_policy.c` |
| Monolith | bodies removed; `mgl_render.h` includes the domain header (Texture.m call sites unchanged) |
| Build | `Makefile` wildcard `*.c` picks up the TU; `test_metalcpp_smoke` explicit list updated |
| Pixel format | domain TU uses `MGLPixelFormat` numeric ABI (no metal-cpp) |
| LOC | `mgl_render.cpp` ~21043→~20599 (−444) |

### 4.2 Second strip — Y-flip / MSAA stride / depth pack / GetTexImagePlan

| Item | Detail |
|------|--------|
| Moved | `mglRenderCopyRows`, `mglRenderCopyDepthTextureBytesToFloat`, `mglRenderDepthReadbackPlan`, `mglRenderTextureRepackDepthPlanes`, `mglRenderMSAAArrayLayerStride`, `mglRenderGetTexImagePlan` (+ `MGLRenderGetTexImagePlan`) |
| Residual (Metal) | `mglRenderEncodeMultisampleResolve*` stays in `mgl_render.cpp` / ObjC ports — not a pure policy table |
| Not moved | format-convert loops that merely take `flip_y` (`Copy*TextureBytesToGL`, BGRA8 paths) — still entangled with decode tables in the monolith |
| LOC | `mgl_render.cpp` ~20599→~20470 (−129); `mgl_readback_policy.c` ~468→~606; header ~126→~196 |

**Next strip suggestion (render):** Optional later flip-aware format convert into `mgl_readback_policy` only if a clean boundary appears; do not sink back into `mgl_render.cpp`. **C1b (air type helpers) done** — see §4b.


---

## 4b. C1b knife log — air type helpers

Chose **air type helpers → `mgl_air_type.*` + `mgl_air_codegen.h`** (DXMT C1b). Narrow strip: former ~290–723 band (+ `typeFromIR`); **not** emitExpr / matrix builtins.

| Item | Detail |
|------|--------|
| Moved | `MType`; carrier predicates/encode/decode; `llvmScalar`/`llvmType`/`llvmTypeFromIR`; `coerceScalar`; array-mem helpers; AIR/MSL mangling; `varyingIfaceTag`; `typeFromIR` |
| Shared state | `Uniform` / `VarSym` / `LoopCtx` / `BreakCtx` / `Codegen` → `mgl_air_codegen.h` (backend-internal; required so type TU can see `Codegen&`) |
| Residual in monolith | `storeStageOut` (stage-out side effect); ~~resource collection~~ → **C1c**; emitExpr / matrix / stmt / legacy |
| New files | `MGL/include/mgl_air_type.h`, `MGL/src/mgl_air_type.cpp`, `MGL/include/mgl_air_codegen.h` |
| Monolith | bodies removed; anon-ns `using mgl::air::*` facade |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~16905→~16302 (−603); new `mgl_air_type.cpp` ~484 |
| Golden | `test_mgl_air_type` — non-Metal carrier/mangle/`typeFromIR` (LLVM+ir only; no Codegen emit, no Metal) |

**Next strip suggestion:** ~~math builtins~~ → **C1d done** (§4d). Further: module-assembly VarSym / later expr facade — keep emitExpr/matrix deferred.

---

## 4c. C1c knife log — air resource collection

Chose **air resource collection → `mgl_air_resource.*`** (DXMT C1c). Banner strip: former ~152–306 (`uniformBlock*` / `collectUniforms` / `appendOpaqueUniformLeaves` / `resolveSamplerAccessName`); **not** emitExpr / matrix / module-assembly VarSym loop.

| Item | Detail |
|------|--------|
| Moved | `uniformBlockType`, `uniformBlockElementCount`, `uniformBlockIsInstanceArray`, `collectUniforms`, `appendOpaqueUniformLeaves`, `resolveSamplerAccessName` |
| Shared state | Uses `Uniform` / `VarSym` / `typeFromIR` via `mgl_air_codegen.h` + `mgl_air_type.h` |
| Residual in monolith | Module-assembly stage VarSym classify + location assign; emitExpr / matrix / stmt / legacy |
| New files | `MGL/include/mgl_air_resource.h`, `MGL/src/mgl_air_resource.cpp` |
| Monolith | bodies removed; anon-ns `using mgl::air::*` facade |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~16302→~16157 (−145); new `mgl_air_resource.cpp` ~173 |

**Next strip suggestion:** ~~math builtins~~ → **C1d done** — see §4d.

---

## 4d. C1d knife log — air math builtins

Chose **air math builtins → `mgl_air_math.*`** (DXMT C1d). Coherent strip: `emitMathBuiltin` (+ `callFloatIntrinsic` / `fpConstOf` / `typeIsIntLike`); **not** emitExpr / matrix builtins.

| Item | Detail |
|------|--------|
| Moved | `emitMathBuiltin` (trig/exp/rounding/geometric/pack/bitfield); float-intrinsic helpers |
| Shared state | `AirMathDeps` hooks into monolith `emitExpr` / `callAirFn` / `dotProduct` / `broadcastTo` / `exprType` / SSBO lvalue path — backend keeps thin static facade |
| Residual in monolith | emitExpr / matrix / stmt / ~~module-assembly VarSym~~ → **C1e**; legacy |
| New files | `MGL/include/mgl_air_math.h`, `MGL/src/mgl_air_math.cpp` |
| Monolith | body removed; anon-ns static `emitMathBuiltin` → `mgl::air::emitMathBuiltin` + `AirMathDeps` |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~16157→~15216 (−941); new `mgl_air_math.cpp` ~1009 |

**Next strip suggestion:** ~~module-assembly VarSym~~ → **C1e done** (§4e). Later expr facade — still defer emitExpr/matrix bodies. Do **not** sink back into `mgl_air_backend.cpp`.

---


---

## 4e. C1e knife log — air VarSym classify / location

Chose **module-assembly VarSym classify/location → `mgl_air_varsym.*`** (DXMT C1e). Coherent strip: `collectStageVarSyms` + `assignStageVarSymLocations` + `varyingLocationSpan` / `airAttribLocation` / `stageRecordStride`; **not** emitExpr / matrix / emitStmt.

| Item | Detail |
|------|--------|
| Moved | Stage VarSym classify from IR symbols; input/output/patch location assign (+ attrib preference + iface peer remap); location span; record/patch stride helpers |
| Shared state | Uses `typeFromIR` / `appendOpaqueUniformLeaves` via `mgl_air_type` + `mgl_air_resource`; thin `AirIfaceLocationPeer` avoids Mach/GL in the new TU |
| Residual in monolith | emitExpr / ~~matrix~~ → **C1f**; stmt / remaining module assembly + legacy `compileGLSLImpl` |
| New files | `MGL/include/mgl_air_varsym.h`, `MGL/src/mgl_air_varsym.cpp` |
| Monolith | bodies removed; `using mgl::air::*` facade + call site in `compileGLSLImpl` |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~15216→~14998 (−223); new `mgl_air_varsym.cpp` ~312; air TU now &lt;15k |

**Next strip suggestion:** ~~matrix builtins~~ → **C1f done** (§4f). ~~stmt strip~~ → **C1g done** (§4g). Later expr facade (still defer whole emitExpr). Do **not** sink back into `mgl_air_backend.cpp`.



---

## 4f. C1f knife log — air matrix builtins

Chose **air matrix builtins → `mgl_air_matrix.*`** (DXMT C1f). Coherent strip: `emitMatrixBuiltin` / `emitMatrixBinOp` (+ `det2Sel` / `det3Sel` / `detMatrix`); **not** whole emitExpr.

| Item | Detail |
|------|--------|
| Moved | `emitMatrixBuiltin` (transpose/matrixCompMult/outerProduct/determinant/inverse); `emitMatrixBinOp` (M*vec/vec*M/M*M/M±scalar/M==); det helpers |
| Shared state | `AirMatrixDeps` hooks into monolith `emitExpr` / `dotProduct` / `scalarizeBoolCompare` — backend keeps thin static facade |
| Residual in monolith | emitExpr / ~~stmt~~ → **C1g**; remaining module assembly + legacy `compileGLSLImpl` |
| New files | `MGL/include/mgl_air_matrix.h`, `MGL/src/mgl_air_matrix.cpp` |
| Monolith | bodies removed; anon-ns static `emitMatrixBuiltin` / `emitMatrixBinOp` → `mgl::air::*` + `AirMatrixDeps` |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~14998→~14594 (−404); new `mgl_air_matrix.cpp` ~453 |

**Next strip suggestion:** ~~stmt strip~~ → **C1g done** (§4g). Later expr facade (still defer whole emitExpr body). Do **not** sink back into `mgl_air_backend.cpp`. Parallel: Batch honest cluster still ~1923; `mgl_render` ~20.5k — not claimed done.

---

## 4g. C1g knife log — air statement emit

Chose **air statement emit → `mgl_air_stmt.*`** (DXMT C1g). Coherent strip: `emitStmt` / `emitCompound` (+ `stmtContainsBreakOrContinue`); **not** whole emitExpr / assembleReturn.

| Item | Detail |
|------|--------|
| Moved | `emitStmt` (compound/expr/decl/return/discard/if/for/while/do-while/switch/break/continue); `emitCompound`; `stmtContainsBreakOrContinue` |
| Shared state | `AirStmtDeps` hooks into monolith `emitExpr` / `exprType` / `assembleReturn` / `cloneIRType` — backend keeps thin static facade |
| Residual in monolith | emitExpr / remaining module assembly + legacy `compileGLSLImpl` / stage-return assembly |
| New files | `MGL/include/mgl_air_stmt.h`, `MGL/src/mgl_air_stmt.cpp` |
| Monolith | bodies removed; anon-ns `emitStmt` → `mgl::air::emitStmt` + `AirStmtDeps` |
| Build | `Makefile` wildcard `*.cpp` picks up TU; explicit `test_mglair` / `test_mcrepro` / `test_mglair_gtest` lists updated |
| LOC | `mgl_air_backend.cpp` ~14594→~13762 (−832); new `mgl_air_stmt.cpp` ~890 |

**Next strip suggestion:** later expr facade (still defer whole emitExpr body), or remaining module-assembly residual. Do **not** sink back into `mgl_air_backend.cpp`. Parallel: Batch honest cluster still ~1923; `mgl_render` ~20.5k — not claimed done.


## 4h. C1 knife log — binding policy (O3.3)

Chose **render binding slot/sampler/stage/plain-uniform policy → `mgl_binding_policy.*`** (DXMT C1 / O3.3) over further readback format-convert: pure `extern "C"` tables with no Metal-cpp owner coupling; aligns BindingState callers without growing `+Binding.m`.

| Item | Detail |
|------|--------|
| New files | `MGL/include/mgl_binding_policy.h`, `MGL/src/mgl_binding_policy.c` |
| Moved | `ShaderResourceElementCount` / `ImageUnits*` / `ComputeTextureBind*` / `ShaderResourceType*` / `PlainUniform*` / `ClientBufferBinding*` / `StageBufferResourceElementCount` / `CombinedSamplerSlot*` / `SamplerNameLooks*` / `ResourceLooksSamplerLike` / `ResourceMetalSlot` / `SamplerUnitValid` / `ShaderStageValid` / `StageMapsVertexAttribs` / `VertexCaptureNeedsLoad` / `StageUsesComputeBufferMap` / `TextureBindingStageForShader` / `SamplerBindingStageForShader` / `SampledResourceUnit` / `DefaultSamplerUnit` / `MetalBindingPastUnits` / `ExpectedTypeUnset` |
| Residual | ~~PSO topology~~ → **O3.2 / §4i**; Metal binding-state apply (~15056+) stays in monolith; **do not** thicken `+Binding.m` |
| Build | `Makefile` wildcard `*.c` picks up TU; `test_metalcpp_smoke` explicit list updated |
| ABI | resource-type / shader-stage numeric (`mgl_types_program.h`); binding stage 0/1 |
| LOC | `mgl_render.cpp` ~20470→~20165 (−305); new `mgl_binding_policy.c` ~329; header ~98 |
| Smoke | Linux `cc -c -std=c11` `mgl_binding_policy.c` |

**Next strip suggestion (render):** (superseded by 4i) BindingState apply masks or generatePipeline apply residual.


## 4i. C1 knife log — format-class PSO (O3.2)

Chose **render format-class PSO builder → `mgl_pso_format_class.*`** (DXMT C1 / O3.2 / CTS Batch 4) over BindingState apply masks: pure `extern "C"` topology / format-class / blend·stencil maps with no Metal-cpp owner coupling; aligns RenderPass / PipelineCache without growing `+Binding.m` or `+RenderPass.m`.

| Item | Detail |
|------|--------|
| New files | `MGL/include/mgl_pso_format_class.h`, `MGL/src/mgl_pso_format_class.c` |
| Moved | `NeedsExplicitTopology` / `PrimitiveTopologyClass` / tess partition·winding·cpi / rasterization·pipeline-ready / VSWritesLayer / depth·stencil·color format-class + pass mismatch / attrib step helpers / blend factor·op / stencil op / cull·fill·depth-clip / scissor·viewport clamps / `PixelFormatIsPackedDepthStencil` / `DefaultDepthPixelFormat` |
| Residual | generatePipeline apply / Metal PSO create and BindingState apply (~15056+) stay in monolith; **do not** thicken `+Binding.m` / `+RenderPass.m` |
| Build | `Makefile` wildcard `*.c` picks up TU; `test_metalcpp_smoke` explicit list updated |
| ABI | GL enums via `glcorearb.h`; Metal value enums via `mgl_render_values.h` |
| LOC | `mgl_render.cpp` ~20165→~19558 (−607); new `mgl_pso_format_class.c` ~641; header ~127 |
| Smoke | Linux `cc -c -std=c11` `mgl_pso_format_class.c` |

**Next strip suggestion (render):** (superseded by 4j) BindingState V/F stage-buffer plan.


## 4j. C1 knife log — stage-buffer bind plan (O3.3 residual)

Chose **+BindingState V/F UBO·SSBO map orchestration → `mgl_binding_stage.*`** (DXMT O3.3 residual) over BindingState apply masks in monolith: pure `extern "C"` plan + helpers; ObjC keeps thin set*Buffer / set*Bytes ports. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`.

| Item | Detail |
|------|--------|
| New files | `MGL/include/mgl_binding_stage.h`, `MGL/src/mgl_binding_stage.c`, `test_legacy_compat/test_binding_stage.c` |
| Moved | `UseInlineFragmentBytes` / `CPUPointerLooksTagged` / `MetalDataPointerUsable` / `NeedsIsolatedStageBinding` / `AllowIsolateGPUWriteTarget` / `BindOffsetInBuffer` / `RequiredBindingBytesForMap` / `UseUniformConstantInline` / `IsolateUBO*` / `IsolateCopyLength` / `WritableStorageNeedsGPUAuthoritative` |
| Added | `mglBindingStagePlanMapEntry` (PRE/POST phases) / fallback resource-type table / `FallbackNeedsBind` / `ResolveSlot` |
| ObjC | `+BindingState` V/F map loops → plan@C + thin emit ports; fallback tables via `mglBindingStageFallbackResourceTypes` |
| Residual | attrib / texture / storage-image / Y-flip BindingState still thick; binding ports still ≫300 LOC; BindingState apply masks (~record/update) remain in monolith |
| Build | wildcard `*.c`; `test_metalcpp_smoke` list; `make test-binding-stage` |
| LOC | `mgl_render.cpp` ~19558→~19484 (−74 helpers); `+BindingState.m` ~4675→~4523 (−152); new `mgl_binding_stage.c` ~plan+helpers; `+Binding.m` unchanged |
| Smoke | Linux `cc -std=c11` `mgl_binding_stage.c` + `test_binding_stage` |

**Next strip suggestion (render/ObjC):** (superseded by 4k) BindingState attrib/texture/image plans.

## 4k. C1 knife log — attrib/texture/image BindingState plans (O3.3 residual)

Chose **+BindingState attrib + sampled/storage-image orchestration → `mgl_binding_stage` / `mgl_binding_texture`** (DXMT O3.3 residual) over BindingState apply masks: pure `extern "C"` plans; ObjC keeps thin set*Buffer / set*Texture / queue ports + shared RT copy helper. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`.

| Item | Detail |
|------|--------|
| New files | `MGL/include/mgl_binding_texture.h`, `MGL/src/mgl_binding_texture.c`, `test_legacy_compat/test_binding_texture.c` |
| Extended | `mgl_binding_stage.*` — attrib bind plan + IntegerAttrib*/SkipAlready*/VertexMetalBindOffset/BindingOffsetInMetal |
| Moved | ImageBindPixelFormat / ImageTargetIsMultisample / ImageLevelInRange / ImageNeeds* / ImageViewSliceCount → texture TU |
| Added | `mglBindingStagePlanAttribEntry`; `mglBindingTexturePlanStorageImage` / `PlanSampled` (GATE/COMPAT/RT/FINAL); sampler warmup/separate/array gates |
| ObjC | attrib loop → plan@C; storage V/F → shared stage ring; vertex sampled GATE/COMPAT/FINAL; shared `applySampledRenderTargetCopyPlan`; fragment GATE/COMPAT |
| Residual | InSampler depth recover → plan@C (this knife); Y-flip/sampler materialize + verbose TBIND logs remain; binding ports still ≫300 |
| Build | wildcard `*.c`; `test_metalcpp_smoke` list; `make test-binding-stage` runs stage+texture |
| LOC | `mgl_render.cpp` ~19484→~19398 (−86); `+BindingState.m` ~4523→~4302 (−221); `+Binding.m` unchanged (488) |
| Smoke | Linux `cc -std=c11` stage+texture units + `test_binding_stage` / `test_binding_texture` |

**Next strip suggestion (render/ObjC):** BindingState apply masks / Y-flip·sampler materialize; or O3.1 pass plan toward &lt;300 binding ports. Do **not** sink back into `mgl_render.cpp`; do **not** grow `+Binding.m`.


## 4l. C1 knife log — InSampler depth-recover plan (O3.3 residual)

Chose **+BindingState `recoverFragmentSampledDepthTexture` orchestration → `mgl_binding_texture` depth-recover plan** (DXMT O3.3 residual) over leaving the InSampler/history/RT decision tree in ObjC: pure `extern "C"` GATE/INSAMPLER/COPY/HISTORY/RT plan; rate-limited NSLog ports in `mgl_binding_texture_log.m`. ObjC stays plan@C + thin bindMTLTexture / copy-probe / apply. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`.

| Item | Detail |
|------|--------|
| Extended | `mgl_binding_texture.{h,c}` — `mglBindingTexturePlanDepthRecover` + InSampler name/log-hit helpers |
| New | `MGL/src/mgl_binding_texture_log.m` — rate-limited depth-recover NSLog ports |
| ObjC | `recoverFragmentSampledDepthTexture` → GATE/INSAMPLER/COPY/HISTORY/RT plan@C + thin apply |
| Residual | ~~Y-flip/sampler materialize~~ (see 4m); binding ports still ≫300 |
| Build | wildcard `*.m` picks up log TU; `make test-binding-stage` runs texture tests |
| LOC | `mgl_render.cpp` unchanged (~19398); `+BindingState.m` ~4302→~4240 (−62); `+Binding.m` unchanged (488) |
| Smoke | Linux `cc -std=c11` `test_binding_texture` (+ depth-recover cases) |

**Next strip suggestion (render/ObjC):** BindingState apply masks / Y-flip·sampler materialize; O3.1 pass plan toward &lt;300 ports. Do **not** sink back into `mgl_render.cpp`; do **not** grow `+Binding.m`.


## 4m. C1 knife log — Y-flip / sampler materialize (O3.3 residual)

Chose **+BindingState Y-flip RT apply ports + sampler materialize + sampled-diag gates → `mgl_binding_texture`** (DXMT O3.3 residual) over leaving V/F duplicate trees in ObjC: pure `extern "C"` sampler-materialize / RT base-level / diag plans; extend existing `mgl_binding_texture_log.m` (no new log shell). ObjC stays plan@C + thin createMTLSampler / view / queue. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`.

| Item | Detail |
|------|--------|
| Extended | `mgl_binding_texture.{h,c}` — `PlanSamplerMaterialize`, RT `apply_base_level_view`, `PlanSampledDiag`, mip-diag signature, rate-log helper |
| Extended | `mgl_binding_texture_log.m` — TBIND / sample-detail / RT Y-flip / sampler-resolve / fallback ports |
| ObjC | V/F shared materialize + COMPAT + TBIND emit; `resolveFragmentSampledYFlipAndSampler` / `applySampledRenderTargetCopyPlan` thin apply |
| Residual | binding ports still ≫300; verbose diag remain behind log ports |
| Forbidden | did **not** grow `+Binding.m`; did **not** sink into `mgl_render.cpp`; no new log TU |
| LOC | `mgl_render.cpp` unchanged; `+BindingState.m` ~4240→~4152 (−88); `+Binding.m` unchanged (488) |
| Smoke | Linux `cc -std=c11` `test_binding_texture` (+ sampler/RT/diag cases) |

**Next strip suggestion (render/ObjC):** (superseded by 4n) further BindingState residual / O3.1 pass plan.


## 4n. C1 knife log — port collapse / apply-masks (O3.3 residual)

Chose **+BindingState apply-masks + port collapse** (DXMT O3.3 residual; after 5bc4a20 texture_log growth review): sampler-warmup plan, V/F snapshot shared emit macros, collapsed TBIND/sample-detail/gui/readback diag ports, merged depth/RT/fallback log POD entry points (**shrink** `mgl_binding_texture_log.m`). Honest metric = BindingState + texture_log. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`; **freeze/shrink** log shell only.

| Item | Detail |
|------|--------|
| Extended | `mgl_binding_texture.{h,c}` — `PlanSamplerWarmup`, `SampledMarkKind`; stage `BuildPresentMask`/`CountPresent` |
| Shrunk | `mgl_binding_texture_log.m` — merged DepthRecover / RTSampleCopy / TexFallbackEx (345→~327) |
| ObjC | warmup single loop; shared snap emit; `emitSampledDiagPorts` V/F; deleted `resolveFragmentSampledYFlipAndSampler` wrapper |
| Forbidden | did **not** grow `+Binding.m` (488); did **not** sink into `mgl_render.cpp`; no new log TU; log shell not grown |
| LOC | `+BindingState.m` ~4152→~3855 (−297); texture_log 345→~327 (−18); **honest combined** ~4497→~4182 (−315) |
| Smoke | Linux `cc -std=c11` `test_binding_texture` (+ apply-mask cases) |

**Next strip suggestion (render/ObjC):** (superseded by 4o) BindingState residual toward &lt;300 ports.

## 4o. C1 knife log — stage fallback unify / residual snap (O3.3 residual)

Chose **+BindingState V/F stage-buffer fallback unify + residual snap macros + present-mask/inline apply** (DXMT O3.3 residual): shared `bindStageFallbackBuffersForStage` (plan@C `PlanFallbackSlot` + thin snap emit); collapse VATTR/VFB/VPS/FFB fat macros onto `MGL_BIND_SNAP_*`; wire `BuildPresentMask`/`CountPresent`/`InlineBytesSrc`. Honest metric = BindingState + texture_log. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`; **freeze** log shell.

| Item | Detail |
|------|--------|
| Extended | `mgl_binding_stage.{h,c}` — `PlanFallbackSlot`, `InlineBytesSrc`; present-mask apply |
| ObjC | unified stage fallback; residual snap wrappers; V/F mask/count/inline thin ports |
| Forbidden | did **not** grow `+Binding.m` (488); did **not** sink into `mgl_render.cpp`; texture_log held at 327 |
| LOC | `+BindingState.m` ~3855→~3648 (−207); texture_log 327→327 (0); **honest combined** ~4182→~3975 (−207) |
| Smoke | Linux `cc -std=c11` `test_binding_stage` (+ fallback-slot/inline/present) |

**Next strip suggestion (render/ObjC):** (superseded by 4p) BindingState residual toward &lt;300 ports.

## 4p. C1 knife log — stage map-entry unify / present-mask finalize (O3.3 residual)

Chose **+BindingState V/F stage-buffer map-entry apply unify + present-mask finalize** (DXMT O3.3 residual): shared `bindStageBufferMapEntriesForStage` (plan@C + `ClampMapCount`/`PostMtlUsable` + thin snap emit); `finalizeStageBufferPresentMask` for V/F mask/sparse/diag; resource-ordinal helper; compact fallback UPDATE/PERF macros. Honest metric = BindingState + texture_log. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`; **freeze** log shell.

| Item | Detail |
|------|--------|
| Extended | `mgl_binding_stage.{h,c}` — `ClampMapCount`, `PostMtlUsable` |
| ObjC | unified V/F map-entry apply; shared present-mask finalize; ordinal helper; fallback UPDATE/PERF macros |
| Forbidden | did **not** grow `+Binding.m` (488); did **not** sink into `mgl_render.cpp`; texture_log held at 327 |
| LOC | `+BindingState.m` ~3648→~3476 (−172); texture_log 327→327 (0); **honest combined** ~3975→~3803 (−172) |
| Smoke | Linux `cc -std=c11` `test_binding_stage` (+ clamp/post-mtl usable) |

**Next strip suggestion (render/ObjC):** (superseded by 4q) BindingState residual toward &lt;300 ports.





## 4q. C1 knife log — sampled V/F stage unify / FINAL ports (O3.3 residual)

Chose **+BindingState V/F sampled-texture apply unify + FINAL ports** (DXMT O3.3 residual): shared `bindSampledTexturesForStage` (GATE/resolve/FINAL plan@C + thin queue); `FillSampledFinalInput`/`ForceDefaultSampler`; SNAP/EMIT macros → headers; remove dead CreateBuffer/RequiredBytes/LevelView. Honest metric = BindingState + texture_log. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`; **freeze** log shell.

| Item | Detail |
|------|--------|
| Shared | `bindSampledTexturesForStage` — V/F sampled spine; fragment keeps depth-recover / mip-diag / trace / always-queue |
| New C | `mglBindingTextureFillSampledFinalInput` / `ForceDefaultSampler`; `test-binding-texture` FINAL cases |
| Headers | SNAP macros → `MGLRenderer+Draw_Private.h`; EMIT_DR/RT → `mgl_binding_texture.h` |
| Forbidden | did **not** grow `+Binding.m` (488); did **not** sink into `mgl_render.cpp`; texture_log held at 327 |
| LOC | `+BindingState.m` ~3476→~3313 (−163); texture_log 327→327 (0); **honest combined** ~3803→~3640 (−163) |

**Next strip suggestion (render/ObjC):** (superseded by 4r) BindingState residual toward &lt;300 ports.


## 4r. C1 knife log — stage-fill / emit ports / attrib SELECT (O3.3 residual)

Chose **+BindingState stage-fill + shared emit ports + attrib SELECT/POST** (DXMT O3.3 residual): `FillMapEntryInput`/`FillAttribSelect|PostMtl`/`AttribNeedsEmit` @C; GATE/COMPAT sampled fills @C; SMB/SFB emit → `Draw_Private` stage macros + Set*/Queue/Flush ports; attrib CURRENT/CONVERT/BIND → shared emit helper; inline V/F fallback wrappers. Honest metric = BindingState + texture_log. Do **not** thicken `+Binding.m`; do **not** sink into `mgl_render.cpp`; **freeze** log shell.

| Item | Detail |
|------|--------|
| New C | `mglBindingStageFillMapEntryInput` / `FillAttribSelectInput` / `FillAttribPostMtlInput` / `AttribNeedsEmit`; `mglBindingTextureFillSampledGateInput` / `FillSampledCompatInput` |
| Headers | Set*/Queue/Flush + `MGL_BIND_STAGE_*` emit macros → `MGLRenderer+Draw_Private.h` |
| ObjC | attrib emit helper; delete thin V/F fallback wrappers (inline shared stage fallback) |
| Forbidden | did **not** grow `+Binding.m` (488); did **not** sink into `mgl_render.cpp`; texture_log held at 327 |
| LOC | `+BindingState.m` ~3313→~3094 (−219); texture_log 327→327 (0); **honest combined** ~3640→~3421 (−219) |

**Next strip suggestion (render/ObjC):** BindingState residual toward &lt;300 ports; or O3.1 pass plan. Do **not** sink back into `mgl_render.cpp`; do **not** grow `+Binding.m` or `texture_log.m`.


## 5. C0 / C1 exit criteria

- [x] Includes / callers / domains documented for **only** these two TUs
- [x] C0 itself: no monolith edits (docs-only)
- [x] **C1** (first knife): IntegerReadback → `mgl_readback_policy.{h,c}`; `mgl_render.cpp` ~21043→~20599 (−444)
- [x] **C1** (O4.1 residual knife): Y-flip / depth pack / GetTexImagePlan / MSAA stride → same TU; `mgl_render.cpp` ~20599→~20470 (−129); Metal MSAA encode residual documented
- [x] **C1b** (air type helpers): `MType`/carriers/LLVM/mangle/`typeFromIR` → `mgl_air_type.*` + `mgl_air_codegen.h`; `mgl_air_backend.cpp` ~16905→~16302 (−603); emitExpr/matrix deferred; non-Metal golden `test_mgl_air_type`
- [x] **C1c** (air resource collection): `uniformBlock*`/`collectUniforms`/opaque leaves/sampler path → `mgl_air_resource.*`; `mgl_air_backend.cpp` ~16302→~16157 (−145); emitExpr/matrix deferred
- [x] **C1d** (air math builtins): `emitMathBuiltin` → `mgl_air_math.*`; `mgl_air_backend.cpp` ~16157→~15216 (−941); emitExpr/matrix deferred; Linux smoke `mgl_air_math.cpp` + llvm-19
- [x] **C1e** (air VarSym classify/location): → `mgl_air_varsym.*`; `mgl_air_backend.cpp` ~15216→~14998 (−223); air TU &lt;15k; emitExpr/matrix deferred; Linux smoke `mgl_air_varsym.cpp` + llvm-19
- [x] **C1f** (air matrix builtins): `emitMatrixBuiltin`/`emitMatrixBinOp` → `mgl_air_matrix.*`; `mgl_air_backend.cpp` ~14998→~14594 (−404); emitExpr deferred; Linux smoke `mgl_air_matrix.cpp` + llvm-19
- [x] **C1g** (air statement emit): `emitStmt`/`emitCompound` → `mgl_air_stmt.*`; `mgl_air_backend.cpp` ~14594→~13762 (−832); emitExpr deferred; Linux smoke `mgl_air_stmt.cpp` + llvm-19
- [x] **C1** (O3.3 binding policy): slot/sampler/stage/plain-uniform → `mgl_binding_policy.{h,c}`; `mgl_render.cpp` ~20470→~20165 (−305); Linux smoke `mgl_binding_policy.c`; `+Binding.m` not grown
- [x] **C1** (O3.2 format-class PSO): topology / format-class / blend·stencil·cull / viewport → `mgl_pso_format_class.{h,c}`; `mgl_render.cpp` ~20165→~19558 (−607); Linux smoke `mgl_pso_format_class.c`; `+Binding.m`/`+RenderPass.m` not grown
- [x] **C1** (O3.3 residual stage bind plan): V/F UBO·SSBO plan + helpers → `mgl_binding_stage.{h,c}`; `mgl_render.cpp` ~19558→~19484 (−74); `+BindingState.m` ~4675→~4523; `test-binding-stage`; `+Binding.m` not grown
- [x] **C1** (O3.3 residual attrib/texture/image): attrib plan → stage; sampled/storage → `mgl_binding_texture.*`; `mgl_render.cpp` ~19484→~19398 (−86); `+BindingState.m` ~4523→~4302 (−221); `test-binding-texture`; `+Binding.m` not grown
- [x] **C1** (O3.3 residual InSampler depth-recover): depth-recover plan → `mgl_binding_texture.*` + log ports; `+BindingState.m` ~4302→~4240 (−62); `test-binding-texture` depth cases; `+Binding.m` not grown; `mgl_render.cpp` unchanged
- [x] **C1** (O3.3 residual port collapse / apply-masks): warmup/snap/diag ports + shrink texture_log; `+BindingState.m` ~4152→~3855; honest combined −315; `+Binding.m` not grown
- [x] **C1** (O3.3 residual stage fallback unify): V/F fallback shared + residual snap + present-mask/inline; `+BindingState.m` ~3855→~3648 (−207); texture_log held 327; `+Binding.m` not grown
- [x] **C1** (O3.3 residual stage map-entry unify): V/F map apply shared + present-mask finalize + ClampMapCount/PostMtlUsable; `+BindingState.m` ~3648→~3476 (−172); texture_log held 327; `+Binding.m` not grown
- [x] **C1** (O3.3 residual sampled V/F stage unify): shared sampled stage + FINAL ports; `+BindingState.m` ~3476→~3313 (−163); texture_log held 327; `+Binding.m` not grown
- [x] **C1** (O3.3 residual stage-fill / emit ports): FillMapEntry/Attrib/GATE/COMPAT @C + Set*/Queue/Flush ports; `+BindingState.m` ~3313→~3094 (−219); texture_log held 327; `+Binding.m` not grown
- [ ] Future knives: BindingState residual / O3.1 pass plan toward &lt;300 ports; later expr facade; keep golden before large moves (ARCH); do not re-enable CI until Paravirt sorted; **do not grow texture_log.m**

