# MGL 全量架构审查（落地对照）

对照落地系列 `98f5ea2`..`42658c8`（代码）+ `938d6eb`/`70bd0aa`（文档）· 1,024 个 git 跟踪文件 · 第一方核心约 209k LOC。

规范基线：OpenGL 4.6 Core + GLSL 4.60。OpenGL ES 3.2 为第二阶段。Minecraft + Sodium/Iris 类路径是产品基线，与规范冲突时以 Khronos 为准。

本文件由审查画布整理，并按落地后仓库更新。审查阶段的「产品代码未改」已经过时。批次 0–3 已合入。批次 4：CommandIR 动态绑定为对象名；已删生产 Compat 符号与 MetalDrawExecutor；DrawArrays/DrawElements/MultiDraw*/Indirect 的 tess/GS/XFB→state→encode 由 C++ `mglIssueDraw*` 编排，公共 primitive encode 走 `mglEncodeDraw*ForRenderEncoderOwner`。GS 拓扑 gather / tess PATCHES 判定与 contract、native TES patch encode、GS passthrough rasterize、TCS ABI 槽位与 dispatch、TES patch item 计算在 C++（`mgl_draw_gs` / `mgl_draw_tess`）；TCS/TES texture bind、stage-in capture 与 AIR TES per-patch plan 仍是 ObjC host。批次 5：独立 ES 3.2 limits 表 + smoke/CTS 子集（limits、GLSL ES 3.20 link）；不是 Khronos GLES CTS。

每条发现保留审查时的域 / 置信 / 规范 / 动作，证据改为当前路径，并加落地状态。

## 结论

当前设计已经选对了方向（自研 GLSL→AIR、C 状态机、Metal-cpp 单一实现 TU、CompileArtifact）。`FrontendSession` 一次 parse 已覆盖 reflect/codegen/uniform seed；错误入队走 `mglDispatchError`（proxy 清错走 `mglClearCurrentError`）。生产 C 入口直接进 ObjC 薄 wrapper / C++ issue，Compat 符号已删除；`mglIssueDraw*` 拥有 tess/GS/XFB 与 Indirect 编排，公共 primitive encode 在 `mgl_draw_encode`。GS 拓扑、tess 判定/contract、native TES patch encode、GS passthrough、TCS ABI 槽位/dispatch、TES item 计算在 C++；TCS/TES texture bind 与 capture 仍在 ObjC。

不要把 Release 无线程 abort、缺 share context、PSO 无锁当成当前 P0——那些仍是 latent 或产品选择。

| 指标 | 审查稿 | 落地后 |
|------|--------|--------|
| P0 阻断项仍开放 | 3 | 0（F01 fail-closed；F02 误报；F03 已进 CI） |
| P1 仍开放 | 16 | F16（TCS/TES texture bind、stage-in capture 与 AIR TES per-patch plan 仍在 ObjC，尚未压到 layer/drawable/swap） |
| P2 仍开放 | 5 | 0（F20 share 显式失败；F21 PSO cache 加锁） |
| 已证实发现 | 22/24 | 不变；F23/F24 仍为 not-a-bug |

**保留**：前后端分离、自研 GLSL→MGLIR→AIR、`CompileArtifact` 原子发布、C 状态机、Metal-cpp 作为唯一 GPU 实现、薄平台壳。

**不要保留**：ObjC 巨型 category 当 encoder、Compat 四跳、seed 路径重复 parse、手写 `gl_core`/`dispatch` 当唯一真相、散落 `getenv` 的 MC 吞错。

信任链只保留三件事：`CompileArtifact` 原子发布、CommandIR 与 live 状态隔离、C ABI 不泄漏 `MTL`。其它内部协议、env 开关、binaryarchive 格式都可以重做。

## 仓库构成

跟踪文件分类。源：`git ls-files` @ 当前落地系列。

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 第一方 MGL | 214 | 改写面（不含 GLM / Khronos 头） |
| GLM + Khronos 头 | 432 | vendored |
| GLFW fork | 164 | 只审 MGL 集成与 fork diff |
| metal-cpp | 139 | 第三方，只审集成边界 |
| 测试 / 脚本 | 28 | `test_*`、`scripts/`、`spec_parser/`、`benchmark/` |
| 其它 | 47 | 根目录、golden、CI、本文件等 |
| **合计** | **1024** | 审查稿 1028；净删空壳 / enum_parser / stale spec 输出后为 1024 |

## 第一方源码规模

`wc -l` 等价格，不含 GLM/Khronos 头。单位：约千行。分层按文件名（`.m` → ObjC；`mgl_air*`/`mgl_glsl*`/`mgl_frontend*` → AIR；`gl_core`/`gl_es`/`dispatch` → ABI；其余 `.cpp` → Metal-cpp；其余 C → 状态机）。与审查稿分层不完全相同，总数仍约 209k。

| 层 | kLOC |
|----|------|
| C GL state | 78.9 |
| ObjC renderer | 58.5 |
| AIR + frontend | 35.3 |
| Metal-cpp | 24.2 |
| GL ABI | 10.4 |
| 其它第一方 | 2.1 |
| **合计** | **209.4** |

复杂度仍集中在 ObjC 编排和 AIR/Metal TU。目标是把 ObjC 压到平台壳，把编译器和 encoder 按域拆开，而不是继续横向加 category。

## 发现清单

置信度：`confirmed` 有路径与调用链；`hypothesis` 需运行复现；`not-a-bug` 为先前误报或产品选择。

落地：`landed` 有代码与门禁；`partial` 做了审查要求的一部分；`open` 未做；`misreported` 审查计数与仓库不符。

### P0

#### F01 DrawTransformFeedback* 静默 no-op — landed

- 域：GL ABI · 置信：confirmed
- 证据：`mgl_gl_extensions.c:2950–2983` 四个入口 `ERROR_RETURN(GL_INVALID_OPERATION)`。`test-arch-correctness` F01。
- 规范：GL 4.6 Core §13.2.3：`DrawTransformFeedback*` 等价于按捕获顶点数 `DrawArrays*`。
- 动作：未实现前应对非法/未实现路径报 `INVALID_OPERATION`；实现 Metal 捕获后再接通。禁止静默成功。
- 落地：fail-closed 已做。Metal 捕获后的真实 Draw 仍未接通。

#### F02 24 项 golden 中缺 17 个 TGA — misreported

- 域：Tests · 置信：confirmed（审查时计数错误）
- 证据：`test_regression/main.c` 23 个 `GOLDEN_TEST`；`MGL_Golden_Images/` 23 个对应 `Reg_*.tga`（git 已跟踪）。审查稿的 24/7 与仓库不符。
- 规范：非规范条款；这是大规模改造的像素门禁。
- 动作：补齐 golden，或把无基准项改为 self-check。CI 在 compare 模式下必须全绿。
- 落地：无需补文件。compare 模式不缺 TGA。

#### F03 CI 未安装 llvm@15 / googletest — landed

- 域：Build/CI · 置信：confirmed
- 证据：`.github/workflows/ci.yml` 安装 `llvm@15`、跑 `make gtest` 与 `make verify-gl-api`。`Makefile` `install-pkgdeps` / `LLVM_ROOT` 与 README 对齐。
- 规范：非规范条款；干净 runner 上构建/gtest 不可复现。
- 动作：CI 显式安装并 pin 版本；README 与 `install-pkgdeps` 对齐。
- 落地：已做。

### P1

#### F04 MSAA 以 2D array 仿真，sample_count=1 — landed（明确 emulation）

- 域：State · 置信：confirmed
- 证据：`MGLRenderer+Texture.m` 将 MS 纹理建成 `Texture2DArray`，`sample_count=1`，`array_length=GL samples`。原因：AIR 把 `sampler2DMS` / `image2DMS` 降成 `texture2d_array`，与 FBO 共用平面 backing。`glm_params.c` `max_image_samples` 与 `max_samples` 同为 4。`test-arch-correctness` F04：`TEXTURE_SAMPLES==4` 且 MS FBO complete。
- 规范：GL 4.6 Core §8.8 Multisample Textures；§9.4.2 完整性与 sample 数。
- 动作：query 与真实能力对齐，或明确 emulation 并让 CTS/光影走可验证路径。`max_image_samples` 与 `max_samples` 必须统一。
- 落地：query 统一为 4；Get 返回 GL sample 数；emulation 路径由 F04 探针锁定。不改成 Metal `texture2d_ms`（会与 AIR 降级冲突）。

#### F05 错误双轨：队列 vs STATE(error) 直写 — landed

- 域：GL ABI · 置信：confirmed
- 证据：`ERROR_RETURN` / `ERROR_CHECK_*`（`glm_context.h:79–82`）走 `mglDispatchError`。`mgl_unimplemented` 入队 `INVALID_OPERATION`。`error.c` 注释为 §2.3.1。`error.c` 之外仅 `mglClearCurrentError` 写 `STATE(error)`（proxy 探测清错）。比较 `STATE(error) ==` 保留。队列仍 16 槽 FIFO（规范下限，满则丢新错误）。
- 规范：GL 4.6 Core §2.3.1：记录第一个错误，后续错误不覆盖。16 槽 FIFO 是 QoI，不是规范下限。
- 动作：全部走 `mglDispatchError`。队列策略改成「首错误保留 + 可选额外 pair」。
- 落地：赋值已收口到 dispatch / `mglClearCurrentError`。16 槽 FIFO 保留（符合 §2.3.1 下限）。

#### F06 4.6 Core 的 Getn* 全 stub — landed（Core 范围）

- 域：GL ABI · 置信：confirmed
- 证据：`mglGetnUniform*` / `mglGetnTexImage` 接到现有 Get 并检查 `bufSize`（`mgl_gl_extensions.c:5707+`）。`GetnMap*` / `GetnPixelMap*` 仍 `mgl_unimplemented`。
- 规范：GL 4.6 Core §7.6 / §8.11：`GetnUniform*` 与 `GetnTexImage` 是核心命令。Map/PixelMap 属 Compatibility。
- 动作：接到现有 `GetUniform`/`GetTexImage` 并加上 `bufSize` 检查，不要留 unimplemented。
- 落地：Core 入口已接。Compatibility Getn* 保持 unimplemented → `INVALID_OPERATION`。

#### F07 Shader subroutine API 全 stub，但 limits 仍广告 — landed

- 域：GL ABI · 置信：confirmed
- 证据：`glm_params.c:554–555` `max_subroutines=0`、`max_subroutine_uniform_locations=0`。入口仍 unimplemented。
- 规范：GL 4.6 Core §7.9 Shader Subroutines。
- 动作：要么实现，要么 limits=0 且入口 `INVALID_OPERATION`，禁止广告不可用能力。
- 落地：选 limits=0 + `INVALID_OPERATION`。

#### F08 GLFW 扩展探测只认 11 条，glGetStringi 认 33 条 — landed

- 域：Platform · 置信：confirmed
- 证据：`external/glfw/src/mgl_context.m` `extensionSupportedMGL` 委托 `glGetStringi` + `GL_NUM_EXTENSIONS`。
- 规范：GLFW 契约，不是 Khronos 条款；会影响 `glfwExtensionSupported`。
- 动作：委托 `glGetStringi`，删除硬编码表。
- 落地：已做。

#### F09 variant 编译绕过 CompileArtifact — landed

- 域：Compiler · 置信：confirmed
- 证据：`mglCompileArtifactFromGLSLEx`（`mgl_compile_artifact.c`）携带 `air_flags` / `iface_peers`。`program.c:1566` `mglCompileCaptureVariant` 经 artifact `complete` 才发布 tess/cull/VS capture。
- 规范：内部信任链 R2，不是 Khronos 条款。失败半发布会破坏 link 原子性。
- 动作：所有 stage（含 tess/cull capture）只经 `CompileArtifact.complete` 发布。
- 落地：已做。

#### F10 link 路径重复 parse：reflect + codegen + uniform seed — landed

- 域：Compiler · 置信：confirmed
- 证据：`FrontendSession` 让 reflect + AIR codegen 共用一次 parse（gtest `FrontendSession.CompileReflectIsSingleParse`）。`CompileShader` 把 TU 存到 `Shader.frontend_tu`。`mglSeedUniformInitializers`（`program.c`）读该 TU，不再 `mglGLSLParse`。
- 规范：非正确性条款。MC/光影首次 link 延迟与峰值内存。
- 动作：`FrontendSession` 持有 TU+IR，reflect/codegen/seed 共用。
- 落地：reflect/codegen/seed 共用 compile TU。

#### F11 gl_ClipDistance/gl_CullDistance 用源码 strstr 推断宽度 — landed

- 域：Compiler · 置信：confirmed
- 证据：`mglFrontendBuiltinArrayCount`（`mgl_frontend_session.c:308`）：IR 符号优先，否则 AST 常量子下标。`mgl_air_backend.cpp` 调用该入口。源码启发式已删。
- 规范：GLSL 4.60 §7.1 / GL 4.6 §11.1.3：数组大小来自声明，注释/宏会误报或漏报。
- 动作：从 sema/IR 符号表取 compile-time size，删除字符串启发式。
- 落地：已做。

#### F12 legacy 翻译固定 len+2048，溢出静默跳过 — landed

- 域：Compiler · 置信：confirmed
- 证据：`mglFrontendRewriteLegacy`（`mgl_frontend_session.c:53`）动态缓冲；失败返回 compile log。`mglFrontendSessionBuild` 失败则编译失败。
- 规范：GLSL 1.x→4.60 兼容层。失败应变成 `COMPILE_STATUS=FALSE`。
- 动作：动态缓冲；失败返回明确 compile log。
- 落地：已做。

#### F13 MGL_MAX_TOKENS 未强制，ShaderSource 无长度上限 — landed

- 域：Compiler · 置信：confirmed
- 证据：`mgl_glsl_parser.c:48,84–90` 超 131072 token fail compile。`shaders.c:312–316` `ShaderSource` 超过 8MiB → `INVALID_VALUE`。
- 规范：实现定义资源上限。恶意/巨型 shader 可 OOM。
- 动作：超限 fail compile；与实现定义 MAX 对齐。
- 落地：已做。

#### F14 libmgl_es.dylib 零测试 — partial

- 域：Tests · 置信：confirmed
- 证据：`test_legacy_compat/test_es_smoke.c`；`make test-es-smoke`。探针含 ES 3.2 字符串、GLSL ES 3.20 compile/link、`CONTEXT_PROFILE_MASK`→`INVALID_ENUM`、GLES 3.2 Table 20.40 下限。`glm_params.c` `mglApplyES32Limits` 独立表。`gl_es.c` 仍缺大量 3.2 入口。无 Khronos GLES CTS。
- 规范：第二阶段 OpenGL ES 3.2。当前 Core/ES 同源双编译无完整回归网。
- 动作：ES 阶段前先加最小 context+`DrawArrays` smoke；现阶段不要假装 ES 已验收。
- 落地：独立 limits 表 + smoke/CTS 子集。不要用这些绿声称 ES 3.2 已验收。

#### F15 gl.xml codegen 已断开，API 层手工维护 — landed（验证层）

- 域：Build/CI · 置信：confirmed
- 证据：`MGL/generated/registry.lock` pin OpenGL-Registry `9cb90ca`。`make verify-gl-api` 对照 `gl_core.c` + overlay。overlay extra=361、missing=0。不生成 `mgl*` 实现体。`spec_parser/spec_parser.c` 保留；已删过期 `spec_parser/mgl.h` / `mgl_funcs.c`。
- 规范：覆盖度相对 OpenGL-Registry。漂移只能靠人工。
- 动作：pin `gl.xml` → 生成薄 ABI → overlay 实现；CI `verify-codegen` diff。不要全自动生成 `mgl*` 实现。
- 落地：验证层已做。手写实现仍 overlay。

#### F16 生产 draw 仍是四跳 Compat 桥，DrawExecutor 未完工 — partial

- 域：Metal · 置信：confirmed
- 证据：生产 `mglRendererDraw*` / `SwapBuffers` / `FlushDrawBuffer` / `ClearBuffer` / `BlitFramebuffer` / `DispatchCompute*` / texture transfer 直接进 ObjC。已删 `mgl_metal_draw_executor.c` 与 `mgl_renderer_compat_bridge.h` / `mglRendererCompat*`。Fake executor 仅 `test-arch-correctness` R4。CommandIR 动态 VB/纹理/EBO 存 `GLuint` 名。`mglIssueDraw*`（`mgl_draw_issue.cpp`）编排 tess/GS/XFB 与 Indirect CPU-expand/native → `mglEncodeDraw*`。ObjC `mtlDraw*` 是 issue 的一行端口。GS 输入拓扑与 gather 在 `mglDrawGs*`；tess PATCHES 判定 / GL 4.6 §10.5 TCS-without-TES / contract 填充在 `mglTessClassifyDraw` / `mglTessFillDrawContract`。native TES patch encode 在 `mglTessEncodeNativePatches`；GS passthrough rasterize 在 `mglDrawGsEncodePassthrough`。TCS ABI 槽位/dispatch 在 `mglTessAppendTCSCoreBindings`；TES patch item 在 `mglTessEvalItemsPer*`。Texture bind、stage-in capture 与 AIR TES per-patch plan 仍是 `MGLRenderer+Tessellation.m`。CommandIR replay 的 array 公共 encode 走 `mglEncodeDrawArraysForRenderEncoderOwner`（cull-distance split 仍在 ObjC）。
- 规范：内部架构。R4 名义边界未成为真相。
- 动作：删除生产 Compat 回退。ObjC 只留 pass 决策与平台壳；encode 下沉 C++。Fake executor 仅测试。不要补完 DrawExecutor 而不删 Compat。
- 落地：Compat 生产符号已删。DrawArrays/DrawElements/MultiDraw/Indirect 编排与公共 encode 已下沉。GS 拓扑 / tess 判定 / native TES patch encode / GS passthrough / TCS ABI dispatch / TES item 计算已在 C++。Texture bind、capture 与其余 ObjC category 仍未压到 layer/drawable/swap。

#### F17 广告 GL_KHR_debug 但无消息存储 — landed

- 域：GL ABI · 置信：confirmed
- 证据：`glm_context.h:147–161` 16 槽 debug ring。`DebugMessageInsert` / `GetDebugMessageLog` 可往返（`test-arch-correctness` F17）。
- 规范：GL 4.6 Core 第 20 章 Debug Output。
- 动作：实现最小 ring buffer，或从扩展串移除。
- 落地：ring buffer 已做。

#### F18 VertexAttrib1/2/3* 为 no-op，4 分量路径正常 — landed

- 域：GL ABI · 置信：confirmed
- 证据：1/2/3 分量接到 current-attrib，缺省补 `(x,0,0,1)` 等（arch F18）。`vertex_arrays.c`：`GetVertexAttrib* CURRENT_VERTEX_ATTRIB` 不要求绑定 VAO（§10.2）。
- 规范：GL 4.6 Core §10.2 Current Vertex Attributes。
- 动作：接到现有 current-attrib 路径，补默认分量。
- 落地：已做。

#### F19 program pipeline 的 GS/tess/compute 未被 batch retain — landed

- 域：Metal · 置信：confirmed
- 证据：`draw_command.h` 增加 `retained_geometry_program` / tess control / tess eval / compute。`draw_command.c` 与 VS/FS 同一套 `mglRetainBatchProgram`。
- 规范：内部生命周期。当前默认 replay 面未触发则为 latent。
- 动作：retain 集合与 replay 消费面用同一张表生成，禁止手写注释契约。
- 落地：retain 面已齐。

### P2

#### F20 无 share group / 共享上下文 — landed（显式失败）

- 域：State · 置信：confirmed
- 证据：`createGLMContext` 无 share 参数；每 context 独立 HashTable。`external/glfw/src/mgl_context.m` 对 `ctxconfig->share` 返回 `GLFW_INVALID_VALUE`。
- 规范：GL 4.6 Core §5.1.3 Shared Objects。单 context MC 主路径可延后。
- 动作：GLFW 收到 share 时显式失败；需要时再做 share group 表。不要静默建独立命名空间。
- 落地：GLFW share 显式失败。share group 表未做。

#### F21 air_loader PSO cache 无锁 — landed

- 域：Compiler · 置信：confirmed
- 证据：`mgl_air_loader.cpp` `psoCacheMutex()` 保护 create 与 shutdown。
- 规范：内部并发。启用异步编译后升 P0。
- 动作：合并进 `PipelineCacheOwner`，删第二套 cache。
- 落地：已加锁。未合并进 PipelineCacheOwner。

#### F22 空壳 TU 仍被 wildcard 链入 dylib — landed

- 域：Build/CI · 置信：confirmed
- 证据：已删空壳实现：`msl_patch_pipeline`、`mgl_toolchain`、`mgl_ir_postprocess`、`mgl_msl_compat.m`、`mgl_compute_pipeline_cache.m`，以及 `enum_parser/`。`mglGetOrCreateProgramComputePipeline` 仍由 `mgl_render.cpp` 实现，头文件保留。`Makefile:127` 仍 `wildcard MGL/src/*.c`。
- 规范：无。误导「还存在 MSL/SPIRV 路径」。
- 动作：删除空文件，Makefile 改为显式源列表。
- 落地：空壳 TU 已删。Makefile 未改为显式列表（空壳不在后无害）。

#### F23 DeleteShader(未知名) 报 INVALID_VALUE — not-a-bug

- 域：GL ABI · 置信：not-a-bug
- 证据：`shaders.c:190` `ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE)`。先前审查误判为应静默忽略。
- 规范：GL 4.6 Core §7.1：零静默忽略；非 shader/program 名应 `INVALID_VALUE`。
- 动作：保持现状。不要改成 silent ignore。
- 落地：保持现状。

#### F24 Release 下 GL 线程断言为空 — not-a-bug

- 域：Metal · 置信：not-a-bug
- 证据：`mgl_thread_affinity.h` Release 下检查编译掉。这是 documented 设计，不是实现漏检。
- 规范：GL 不要求实现 abort 跨线程调用。
- 动作：可选 `MGL_ENABLE_THREAD_CHECKS` 进 CI；不要当 P0 规范违规。
- 落地：保持现状。

## 当前架构 vs 目标

### 现在：三层方向已接通，ObjC 实现仍厚

```
gl* → dispatch → mgl* 状态
  → CommandIR（动态绑定为对象名）+ live/replay
    → C 入口（Draw/Swap/Flush/Clear/Blit/Compute/texture）
      → C++ mglIssueDraw*（tess/GS/XFB/Indirect 编排）
        → C++ mglDrawGs* / mglTess*（拓扑 gather、PATCHES 判定、contract、native patch encode、GS passthrough、TCS ABI dispatch）
        → ObjC host 端口（texture bind、capture、AIR TES per-patch plan、encoder recover）
        → mgl_draw_encode / mgl_render.cpp Metal-cpp encode
Compat 生产符号已删除。Draw* / MultiDraw* / Indirect 公共 encode 不在 `mtlDraw*` 内联。
```

编译：`FrontendSession` 一次 parse 供 reflect+codegen+uniform seed。capture/variant 走 `CompileArtifact`。

### 目标：三层 + 一次编译

```
生成式 C ABI（profile-aware）
  → GL 语义核心：校验、可变 live 状态、不可变 CommandIR
      → C++ Metal backend：资源 + encoder + PSO
          → 薄 ObjC 端口：layer / drawable / swap
  → FrontendSession 一次 → Reflect + Codegen + uniform seed 共用 IR
      → CompileArtifact.complete 才发布
```

## 模块处置

| 模块 | 规模 | 处置 | 落地备注 |
|------|------|------|----------|
| `gl_core` / `gl_es` / dispatch | ~10k | 局部重写 | 薄 ABI 由 pin 的 `gl.xml` 验证 + overlay。ES 有独立 limits 表 + smoke/CTS 子集。 |
| GLMContext / 资源状态 | ~79k | 保留并收口 | debug ring 进 context。GLFW share 显式失败；share group 表未做。ES limits 走 `mglApplyES32Limits`。 |
| GLSL frontend + MGLIR | ~35k | 保留并收口 | `FrontendSession` + seed 共用 TU。 |
| `mgl_air_backend.cpp` | 含在 AIR | 局部重写 | clip/cull 改 IR/AST；TU 仍未按域拆开。PSO cache 已加锁。 |
| CompileArtifact / reflect | 含在 AIR | 保留并收口 | variant/capture 已进同一门闩。 |
| `draw_command` 批处理 | 含在状态机 | 保留并收口 | GS/tess/compute retain 已齐。动态 VB/纹理/EBO 改为对象名。 |
| `MGLRenderer+*.m` | ~59k | 替换边界 | PSO miss 不再复用旧 PSO。Compat 符号已删。Draw*/MultiDraw*/Indirect 编排在 C++ `mglIssueDraw*`。GS 拓扑 / tess 判定 / native TES encode / GS passthrough 在 C++；TCS/TES compute 仍在 ObjC。 |
| `mgl_render.cpp` + backend | ~24k | 局部重写 | 已删 MetalDrawExecutor 与 Compat 桥。 |
| Platform shell + GLFW fork | 薄 | 保留并收口 | 扩展探测已委托 `glGetStringi`。share 显式失败。 |
| OpenGL ES 3.2 路径 | `gl_es.c` | 保留并收口 | 独立 limits 表 + smoke/CTS 子集。不要扩 ES 语义假装验收。 |
| `enum_parser` / stale spec_parser 输出 | — | 删除 | 已删。`spec_parser/spec_parser.c` 留下给 verify。 |
| 空壳 MSL/SPIRV TU | — | 删除 | 已删空壳实现 TU。compute pipeline cache 的 C++ 实现仍在。 |

## 不要做的事

- 不要把 DrawExecutor VTable 再补完一层而不删除 Compat。indexed stub 不影响生产正确性（已反证）。
- 不要为「完整 4.6」去实现 display list / immediate mode。Core profile 应停止导出或保持 `INVALID_OPERATION`。
- 不要在未补 golden/CI 依赖前拆 AIR/Metal 大 TU。没有门禁的重构不可逆（golden/CI 现已齐）。
- 不要把 Minecraft compat 做成默认吞错误。MC 走显式 profile 对象，CTS 保持 strict。
- 不要把 TES 含 SSBO 一律强制 `tess_eval_compute`（会破坏 `air_tessellation_resources` 像素）。

## 分阶段路线

| 批次 | 审查停止条件 | 落地 |
|------|----------------|------|
| 0 安全网 | 干净 clone 上 `make test-all` 绿；CI 装 llvm@15+gtest | 完成。golden 本已齐；CI/gtest/`verify-gl-api` 已加。 |
| 1 codegen | pin `gl.xml`，CI diff；删空壳 TU | 完成（验证层 + 删空壳）。手写 `mgl*` 实现仍 overlay。Makefile 仍 wildcard。 |
| 2 GL 语义 | XFB 失败关闭；Getn*/Attrib1-3/debug 与广告一致；错误统一入队；GLFW 委托 | 完成。F05 直写已收口。 |
| 3 编译链 | FrontendSession；clip/cull 改 IR；legacy 动态缓冲；variant 走 CompileArtifact | 完成。seed 共用 TU。 |
| 4 backend | CommandIR 深句柄；encode 下沉；删 Compat | **部分完成**。对象名句柄；已删 Compat 与 MetalDrawExecutor；Draw*/MultiDraw*/Indirect 编排与公共 encode 下沉。GS 拓扑 / tess 判定 / native TES encode / GS passthrough 在 C++。TCS/TES compute 仍在 ObjC。 |
| 5 ES 3.2 | 独立 profile 表 + 最小 GLES CTS 子集 | **部分完成**。`mglApplyES32Limits` + smoke（limits / GLSL ES 3.20 / DrawArrays）。不是 Khronos GLES CTS。 |

下一批应把 TCS/TES compute（`MGLRenderer+Tessellation.m`）与 tess/GS capture 移出巨型 ObjC category，压到 C++ backend + 薄 layer/drawable/swap 端口，而不是扩 ES。

## 验证矩阵

改造期间最低门禁。下列结果来自落地后本机跑通（macOS，Apple M4）：

| 门 | 命令 / 动作 | 通过标准 | 本次证据 |
|----|-------------|----------|----------|
| A 构建 | CI brew llvm@15 + gtest；`make lib` | 干净 macOS 14 runner 成功 | workflow 已写；本机 `make lib` 成功 |
| B 状态 | `test-arch-correctness` + `test-dirty-hash` | 与已落地的 §2.3.1 / attrib / debug 探针一致 | arch all probes passed；dirty-hash PASS |
| C 编译 | `test-mglair-gtest`；`verify-gl-api` | 单次 parse（含 seed TU）；reflection 槽位 assert | 51/51；`verify-gl-api: ok` |
| D 像素 | `test-regression` | 全 PASS 或 SKIP；golden 文件齐全 | **91 PASS / 0 FAIL / 2 SKIP / 93** |
| E 性能 | `test-benchmark` | P95 回退 ≤10% | 未在本次重跑；CI 仍有该 step |
| F 产品 | MC 1.21 + Sodium/Iris | 启动、世界、GUI、光影不花屏 | 未在本次验证 |
| ES | `test-es-smoke` | context + 3.2 字符串 + profile mask 拒绝 + GLES 3.2 limits 下限 + GLSL ES 3.20 link + DrawArrays 不崩 | `es-smoke: ok` |

`make test-all` 现含 `verify-gl-api` 与 `test-es-smoke`。不要用 ES smoke 绿来声称 ES 3.2 已验收。

## 规范来源

- [OpenGL 4.6 Core Profile](https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf)（2022-05-05）
- [GLSL 4.60.8](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [OpenGL ES 3.2](https://registry.khronos.org/OpenGL/specs/es/3.2/es_spec_3.2.pdf)
- [GLSL ES 3.20.8](https://registry.khronos.org/OpenGL/specs/es/3.2/GLSL_ES_Specification_3.20.pdf)

章节号以 PDF 为准。代码注释里的 §2.5 已过时，现行错误模型在 §2.3.1。

## 落地 commits

相对 `71a1db9`：

1. `98f5ea2` chore: drop unused MSL/SPIRV stubs and dead parsers
2. `396cb55` build: pin gl.xml and install llvm@15/gtest in CI
3. `9fc1616` fix(gl): fail closed XFB and align advertised 4.6 APIs
4. `590cc29` fix(glsl): share one FrontendSession and publish capture via CompileArtifact
5. `98964f1` fix(rt): retain pipeline stages and stop reusing a mismatched PSO
6. `42658c8` test(es): add a 3.2 context smoke against libmgl_es
7. `938d6eb` docs: record architecture-review landing status
8. `70bd0aa` docs: correct review landing status against HEAD

未 push。`build-audit/` 不入库。
