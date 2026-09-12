# MGL 全量架构审查（落地对照）

对照落地系列 `98f5ea2`..`42658c8`（代码）+ `938d6eb`/`70bd0aa`（文档）· 1,024 个 git 跟踪文件 · 第一方核心约 209k LOC。

规范基线：OpenGL 4.6 Core + GLSL 4.60。OpenGL ES 3.2 为第二阶段。Minecraft + Sodium/Iris 类路径是产品基线，与规范冲突时以 Khronos 为准。

本文件由审查画布整理，并按落地后仓库更新。审查阶段的「产品代码未改」已经过时。批次 0–3 已合入。批次 4：CommandIR 动态绑定为对象名；已删生产 Compat 符号与 MetalDrawExecutor；DrawArrays/DrawElements/MultiDraw*/Indirect 的 tess/GS/XFB→state→encode 由 C++ `mglIssueDraw*` 编排，公共 primitive encode 走 `mglEncodeDraw*ForRenderEncoderOwner`。GS 拓扑 gather / tess PATCHES 判定与 contract、native TES patch encode、GS passthrough rasterize、TCS ABI 槽位与 dispatch、TES patch item 计算、AIR TES per-patch dispatch plan、TCS/TES texture 槽位映射、TCS stage-in 默认值与顶点 pack、indexed TCS sparse compact、GS loc_map / XFB scatter 字段 / counts preset / 核心 compute bindings / compute layout / XFB prefix-sum 与 scatter plan、cull-distance array split plan/encode、VS capture POINT encode 与 restart sanitize、cull-distance attrib 扫描 / emu params / 槽 28/29 bind、VS capture 槽 28/29 bind 在 C++（`mgl_draw_gs` / `mgl_draw_tess` / `mgl_draw_encode`）；ObjC 仍物化 Metal texture/sampler 与 TCS attrib 源指针、VAO cull attrib resolve，并做 VS GPU capture host（processGLState / capture buffer 分配）。批次 5：独立 ES 3.2 limits 表 + smoke/CTS 子集（limits、GLSL ES 3.20 link）；不是 Khronos GLES CTS。

每条发现保留审查时的域 / 置信 / 规范 / 动作，证据改为当前路径，并加落地状态。

## 结论

当前设计已经选对了方向（自研 GLSL→AIR、C 状态机、Metal-cpp 单一实现 TU、CompileArtifact）。`FrontendSession` 一次 parse 已覆盖 reflect/codegen/uniform seed；错误入队走 `mglDispatchError`（proxy 清错走 `mglClearCurrentError`）。生产 C 入口直接进 ObjC 薄 wrapper / C++ issue，Compat 符号已删除；`mglIssueDraw*` 拥有 tess/GS/XFB 与 Indirect 编排，公共 primitive encode 在 `mgl_draw_encode`。GS 拓扑、tess 判定/contract、native TES patch encode、GS passthrough、TCS ABI 槽位/dispatch、TES item 计算、AIR TES per-patch plan、TCS/TES texture 槽位映射、TCS stage-in 默认值与顶点 pack、indexed TCS sparse compact、GS loc_map / XFB scatter 字段 / counts preset / 核心 compute bindings / compute layout / XFB prefix-sum 与 scatter plan、cull-distance array split、VS capture POINT encode、cull-distance attrib 扫描 / emu params / 槽 bind、VS capture 槽 28/29 bind 在 C++；Metal texture/sampler 物化、TCS attrib 源指针、VAO cull resolve、VS GPU capture host（processGLState / capture buffer 分配）仍在 ObjC。

不要把 Release 无线程 abort、缺 share context、PSO 无锁当成当前 P0——那些仍是 latent 或产品选择。

| 指标 | 审查稿 | 落地后 |
|------|--------|--------|
| P0 阻断项仍开放 | 3 | 0（F01 fail-closed；F02 误报；F03 已进 CI） |
| P1 仍开放 | 16 | 历史 F16（VS GPU capture 与 ObjC renderer 尚未压到 layer/drawable/swap）；续审新增 B01-B07 |
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
| ObjC renderer | 34.5（2026-09-12 实测；审查稿为 58.5，O1/O2/C1/O5/O7 已降 ~24k） |
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
- 证据：生产 `mglRendererDraw*` / `SwapBuffers` / `FlushDrawBuffer` / `ClearBuffer` / `BlitFramebuffer` / `DispatchCompute*` / texture transfer 直接进 ObjC。已删 `mgl_metal_draw_executor.c` 与 `mgl_renderer_compat_bridge.h` / `mglRendererCompat*`。Fake executor 仅 `test-arch-correctness` R4。CommandIR 动态 VB/纹理/EBO 存 `GLuint` 名。`mglIssueDraw*`（`mgl_draw_issue.cpp`）编排 tess/GS/XFB 与 Indirect CPU-expand/native → `mglEncodeDraw*`。ObjC `mtlDraw*` 是 issue 的一行端口。GS 输入拓扑与 gather 在 `mglDrawGs*`；tess PATCHES 判定 / GL 4.6 §10.5 TCS-without-TES / contract 填充在 `mglTessClassifyDraw` / `mglTessFillDrawContract`。native TES patch encode 在 `mglTessEncodeNativePatches`；GS passthrough rasterize 在 `mglDrawGsEncodePassthrough`。TCS ABI 槽位/dispatch 在 `mglTessAppendTCSCoreBindings`；TES patch item 在 `mglTessEvalItemsPer*`；AIR TES per-patch dispatch 在 `mglTessAppendEvalPerPatchDispatches`；TCS/TES image/sampler 槽位在 `mglTessCollectTextureBinds`（ObjC 只物化 `mtl_data`）。TCS stage-in 默认值在 `mglTessInitStageInDefaults`；indexed TCS sparse→continuous 在 `mglTessCompactSparseCapture`；TCS stage-in 顶点 pack 在 `mglTessPackStageInRecords`（ObjC 只填 attrib 源指针）。GS loc_map / XFB scatter 字段 / counts preset / 核心 compute bindings / compute layout / XFB prefix-sum / scatter plan 在 `mglDrawGsFillLocationMap` / `mglDrawGsFillXFBScatterParams` / `mglDrawGsPresetCounts` / `mglDrawGsAppendCoreBindings` / `mglDrawGsComputeLayout` / `mglDrawGsExclusivePrefixSum` / `mglDrawGsFillXFBScatterPlan`。cull-distance array split 在 `mglRenderFillCullDistanceArrayPrimitives` / `mglRenderCreateCullDistanceArrayPlan` / `mglEncodeCullDistanceArraySplitForRenderEncoderOwner`。cull-distance attrib 扫描 / layout / emu params / 槽 28/29 bind 在 `mglRenderCollectCullDistanceAttribs` / `mglRenderFillCullDistanceEmuParams` / `mglRenderBindCullDistanceEmuSlots`（ObjC 只做 VAO resolve 与 last-bound 记账）。VS capture POINT encode 与 restart sanitize 在 `mglTessEncodeCapture*` / `mglTessSanitizeRestartIndices`；槽 28/29 bind 在 `mglTessBindCaptureSlots`。host（processGLState、capture buffer 分配）仍是 `MGLRenderer+DrawSupport.m`。CommandIR replay 的 array 公共 encode 走 `mglEncodeDrawArraysForRenderEncoderOwner`。
- 规范：内部架构。R4 名义边界未成为真相。
- 动作：删除生产 Compat 回退。ObjC 只留 pass 决策与平台壳；encode 下沉 C++。Fake executor 仅测试。不要补完 DrawExecutor 而不删 Compat。
- 落地：Compat 生产符号已删。DrawArrays/DrawElements/MultiDraw/Indirect 编排与公共 encode 已下沉。GS 拓扑 / tess 判定 / native TES patch encode / GS passthrough / TCS ABI dispatch / TES item 计算 / AIR TES per-patch plan / TCS/TES texture 槽位映射 / TCS stage-in 默认值与 pack / indexed TCS sparse compact / GS loc_map / XFB scatter 字段 / counts preset / 核心 compute bindings / compute layout / XFB prefix-sum 与 scatter plan / cull-distance array split / VS capture POINT encode / cull attrib 扫描与 emu 槽 bind / capture 槽 bind 已在 C++。VS GPU capture host（processGLState / buffer 分配）与 VAO cull resolve 仍在 ObjC，renderer 尚未压到 layer/drawable/swap。

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
        → C++ mglDrawGs* / mglTess*（拓扑 gather、PATCHES 判定、contract、native patch encode、GS passthrough、TCS ABI dispatch、TES per-patch plan、texture 槽位映射、stage-in 默认值与 pack、sparse compact、GS loc_map / XFB scatter / counts / 核心 bindings / compute layout / XFB prefix-sum、cull-distance array split、VS capture POINT encode、cull attrib 扫描与 emu 槽 bind、capture 槽 28/29 bind）
        → ObjC host 端口（texture/sampler 物化、TCS attrib 源指针、VAO cull resolve、VS capture processGLState / buffer 分配、encoder recover）
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
| `MGLRenderer+*.m` | 34.5k（实测） | 替换边界 | PSO miss 不再复用旧 PSO；**PSO 键已含 `tessVertexRenderActive`**（TES-vertex 与 compute 两条栅格化路线不再互相命中，见 `docs/TESS_NATIVE_RENDER_VERTEX_PATH.md` §10.4）。Compat 符号已删。Draw*/MultiDraw*/Indirect 编排在 C++ `mglIssueDraw*`。GS 拓扑 / tess 判定 / native TES encode / GS passthrough / TCS ABI / TES per-patch / texture 槽位 / stage-in 默认值与 pack / sparse compact / GS loc_map / XFB scatter / counts / 核心 bindings / compute layout / XFB prefix-sum / cull-distance array split / VS capture POINT encode / cull attrib 扫描与 emu 槽 bind / capture 槽 bind 在 C++；VS capture processGLState 与 VAO cull resolve 仍在 ObjC。 |
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
| 4 backend | CommandIR 深句柄；encode 下沉；删 Compat | **部分完成**。对象名句柄；已删 Compat 与 MetalDrawExecutor；Draw*/MultiDraw*/Indirect 编排与公共 encode 下沉。GS 拓扑 / tess 判定 / native TES encode / GS passthrough / TCS ABI / TES per-patch / texture 槽位 / stage-in 默认值与 pack / sparse compact / GS loc_map / XFB scatter / counts / 核心 bindings / compute layout / XFB prefix-sum / cull-distance array split / VS capture POINT encode / cull attrib 扫描与 emu 槽 bind / capture 槽 bind 在 C++。VS capture processGLState 与 VAO cull resolve 仍在 ObjC。 |
| 5 ES 3.2 | 独立 profile 表 + 最小 GLES CTS 子集 | **部分完成**。`mglApplyES32Limits` + smoke（limits / GLSL ES 3.20 / DrawArrays）。不是 Khronos GLES CTS。 |

下一批应把 VS GPU capture 的 processGLState / buffer 分配与 VAO cull resolve 移出巨型 ObjC category，压到薄 layer/drawable/swap 端口，而不是扩 ES。array split、capture POINT encode、cull attrib 扫描与槽 bind 已下沉。

**ObjC 薄平台层拆解**：见 [`docs/OBJC_CATEGORY_DISMANTLE_TODO.md`](OBJC_CATEGORY_DISMANTLE_TODO.md)（O0 政策/度量；O1 draw/tess/GS 宿主；O2 batch path）。度量脚本 `scripts/objc_renderer_loc.sh`，目标 `MGLRenderer*.m` 合计 ≤ 8–12k。即时下一刀：见该文档 §5 第 29 条（当前为 O7.4 的 SPIRV 兼容清理残项与 O5.2 `+Compute.m` / O4.4 `+Blit.m` 下沉）。Batch O7 专管「SPIRV→LLVM IR 兼容层清理」：先用探针/逐条 diff 建 oracle，再删源码文本判定与名字启发式（已清 10 处文本判定 + 3 份 sampler 启发式）。

## 验证矩阵

改造期间最低门禁。下列结果来自落地后本机跑通（macOS，Apple M4）：

| 门 | 命令 / 动作 | 通过标准 | 本次证据 |
|----|-------------|----------|----------|
| A 构建 | CI brew llvm@15 + gtest；`make lib` | 干净 macOS 14 runner 成功 | workflow 已写；本机 `make lib` 成功 |
| B 状态 | `test-arch-correctness` + `test-dirty-hash` | 与已落地的 §2.3.1 / attrib / debug 探针一致 | arch all probes passed；dirty-hash PASS |
| C 编译 | `test-mglair-gtest`；`verify-gl-api` | 单次 parse（含 seed TU）；reflection 槽位 assert | 51/51；`verify-gl-api: ok` |
| D 像素 | `test-regression` | 全 PASS 或 SKIP；golden 文件齐全 | **91 PASS / 0 FAIL / 2 SKIP / 93**（本轮完整复跑；`compute_dispatch_ssbo` 已恢复） |
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

## 续审补充（2026-09-06）

对当前工作树继续审查后，仍需优先处理以下问题。它们不应被上表中已经落地的 A 项掩盖：

| 优先级 | 文件与位置 | 问题 | 处理方向 |
|---|---|---|---|
| P1 | `MGL/src/mgl_ir.c:54-60,131,163,183-189` | std140/std430 的对齐、stride、数组大小和结构体偏移使用未检查的 `uint32_t` 运算；大数组或深层嵌套可回绕，导致欠分配和错误 GPU ABI。单一 `type->layout` 也会被不同 layout standard 互相覆盖。 | 使用 checked 64 位/`size_t` layout engine；按 layout standard 保存不可变结果，溢出即拒绝编译。 |
| P1 | `MGL/src/mgl_air_reflect.c:260-320,527-565,987-997,1140-1144` | uniform/block 路径固定 192/208 字节并忽略 `snprintf` 截断；多维数组被压成一维；多处 `strdup/realloc/calloc` 失败后仍可能返回部分 metadata；非法 layout 被 fallback 为 size=4。 | 动态递归名称与维度元数据；reflection 与 codegen 通过 `CompileArtifact` 原子发布，任何分配/layout 失败都返回错误。 |
| P1 | `MGL/src/mgl_glsl_parser.c:64-100,982-984,3232-3234,466-533` | tokenize 失败泄漏 token/source；多个 `realloc` 未检查；固定常量/类型表超限时静默丢语义。 | 统一失败传播和清理，动态表或显式上限错误。 |
| P1 | `MGL/src/mgl_glsl_sema.c:4350-4407` | 跨阶段接口主要按名称和类型比较，未把 explicit location、component/index、patch/sample、插值和 matrix-major 纳入 ABI 检查。 | 生成版本化 `StageInterfaceRecord`，link 前完成完整契约校验。 |
| P1 | `MGL/src/mgl_air_loader.cpp:45-52,243-328` | PSO key 不含 device，且按 descriptor 原始字节（含 padding）序列化；全局 mutex 覆盖 PSO 编译和 archive IO。 | 按 device 分区、字段级 canonical key；锁外编译，成功后 double-check 插入。 |
| P1 | `MGL/src/mgl_renderer_backend.cpp` lease Begin/End + getter TLS | getter 解锁后返回未 retain 的 Metal 裸指针，destroy 随后释放；`GetDevice` 甚至无锁，teardown 竞态可形成 UAF。 | **已落地**：thread-local lease + destroy drain；borrowed Get* 要求本线程持有 lease。 |
| P2 | `test_regression/main.c:16400-16402,16494-16498`、`scripts/apitrace_capture.sh:73-81,150-159` | 测试和 capture 工具用 `system()`/未转义 `source` 处理路径，存在注入和截断风险。 | **已部分落地**：回归路径改为 checked libc mkdir/copy，capture state 改为受限 tab-separated 解析；仍需审查其余脚本外部命令边界。 |
| P2 | `test_mgl/main.cpp:211-223`、`test_regression/main.c:177-202` | 3D 纹理尺寸乘法无边界检查且 VM 内存未释放；TGA writer 忽略 I/O 错误。 | **已部分落地**：checked byte-size、`vm_deallocate`、逐次 TGA I/O 检查已加入；其他生成纹理调用仍需 RAII/ownership wrapper。 |

本次复跑中，`test-arch-correctness` 已通过。期间修复了 `mglRenderVertexAttribName` 缺少 `attrib_location_names` 回退的确定性回归，并删除了 `mgl_draw_gs.cpp:526` 的无意义溢出检查。`make lib` 成功但仍有 enum/static_assert 和 linker warning；已在本地 pinned registry 上直接运行 `python3 scripts/verify_gl_api.py` 并通过。网络同步脚本本身仍可能因 GitHub 不可用而阻断 `make test-all`，这不改变已验证的本地 API 对照结果。

## 续审补充（2026-09-06，compute 与 render-target）

### 已确认并修复：匿名 SSBO 成员的布局缓存断层

`layout(std430) buffer Out { int data[8]; };` 的匿名成员会被语义阶段 flatten 成独立 `MGLIRSymbol`。原先 `ir_type_clone()` 为避免跨标准污染而清空布局缓存，但发布成员符号前没有重新计算布局；AIR 地址生成读取到 `array_stride == 0`，所有 `data[gl_GlobalInvocationID.x]` 写入同一地址，表现为 `compute_dispatch_ssbo` 中 `data[1] == 0`。

现在由 `layout_standard_for_decl()` 统一解析块标准，flatten 成员克隆后立即调用 `mglIRComputeLayout()`；克隆仍保持清缓存语义，`std140/std430` 结果不会互相覆盖。`test_mglsema` 增加了 flattened SSBO stride 断言，`compute_dispatch_ssbo` 隔离复跑已通过。

### 已落地的 render-target 语义修复

- MSAA array backing 的物理切片按 `physical_slice = gl_layer * 8 + sample` 映射；普通 2D array 仍保持一对一层映射，pipeline cache key 包含该 stride。
- scaled blit 对 array/cube attachment 创建选定 level/slice 的 2D texture view，再交给 `texture2d` fragment path，避免把 array/cube 原纹理误绑定成 2D。
- blit 目标 attachment 即使已有 sampled-only Metal backing，也强制执行 RenderTarget usage transition 后再创建 blit encoder。

### Sync 生命周期：teardown barrier 已落地

`mglAcquireSync()` 与 `mglDeleteSync()` 使用 context-owned `sync_lock` 覆盖 pointer lookup、refcount 增加和 table detach；wait/status 在解锁后持有引用，避免常规 delete/wait 竞态。现在所有 fence API 先登记 context-level active operation，`destroyGLMContext()` 先设置 destroying gate、拒绝新进入，再等待 active operation 归零后释放 Sync、backend 和 context，覆盖 teardown 期间 waiter 使用已释放对象的 UAF 路径。`Sync.delete_status` 也已改为真正的 `_Atomic GLboolean`，不再把普通字段强转为原子类型。

这条从 **P1 open** 移为已落地。context 指针本身仍要求调用方不要在 `destroyGLMContext()` 返回后继续调用；进入 barrier 之后的新 fence API 会收到 `GL_INVALID_OPERATION`。

### 本轮续审状态（2026-09-06）

- **MGLIR layout：已落地。** std140/std430 结果按 layout standard 独立缓存；checked `uint64_t` 运算拒绝数组、结构体和嵌套偏移溢出；布局节点完整成功后才发布成员偏移；`ir_type_clone` 不复制旧 metadata；析构释放全部缓存。`test_mglir` 覆盖双标准缓存切换和 `UINT32_MAX` 数组溢出。
- **AIR reflection：已部分落地。** block flatten 的资源/成员扩容、slot 计数、纹理 binding、动态名称和分配失败传播已收口；失败不会发布不完整 block metadata。多维数组仍被压缩为单一 `gl_array_size`/`num_array_dims=1`，需要后续把维度数组同时接入 reflection、uniform 查询和 codegen。
- **GLSL parser：仍是 P1 partial。** token 上限和动态缓冲已有；tokenize 资源限制失败路径会清理 token/source，固定大小的常量、类型、数组和成员记录表在超限时返回显式 parse error，成员路径也拒绝 `snprintf` 截断。条件编译深度和宏定义数量超限的统一错误传播仍需补齐。
- **Renderer backend getter：已落地。** `mglRendererBackendBegin`/`End` 建立 thread-local lease；borrowed `Get*` 仅在调用线程持有匹配 lease 时返回对象；`Destroy` 在 `ReleaseOwnedState` 前 drain `active_leases`。ObjC `_device`/`_commandQueue` 宏与 `mglRenderer*` 入口、Lifecycle init owner 缓存纳入同一协议；`LeaseGetDevice`/`LeaseGetCommandQueue`/`LeaseGetOwner` 提供显式 lease 作用域 API。metalcpp smoke 覆盖无 lease 拒绝、lease 下借用，以及 getter 与 destroy 交错 barrier。
- **Sync teardown：已落地。** fence API 有 context-level active-operation barrier；destroy gate 先拒绝新进入、等待 active operation 归零，再释放 Sync/backend/context。`Sync.delete_status` 使用真正的原子字段。
- **Test/capture tooling：已部分落地。** 回归程序的输出目录创建、golden 更新和 TGA writer 已移除 shell 拼接，改为 checked libc 文件操作；apitrace capture 状态文件改为受限的 tab-separated 读取，不再 `source` 外部内容。`test_mgl` 的 3D 纹理生成加入 checked size arithmetic、VM 释放，并修正整数格式 helper 中把异或误写成幂运算的确定性错误。测试工具仍有若干长期持有的临时纹理分配，后续可用 RAII/ownership wrapper 继续收口。

### Renderer backend ownership / lease handoff（P1 open）

#### 问题模型

`mgl_renderer_backend.h` 的 `Get*` 约定目前是“backend 持有、调用方借用”。实现只在 `backend->mutex` 内读取指针，随后解锁并把裸 `void *` 返回给 ObjC/C++ 调用方。这个锁只保护读取动作，不覆盖调用方真正使用 Metal 对象或 opaque owner 的时间区间。

销毁路径 `mglRendererBackendDestroy()`（`MGL/src/mgl_renderer_backend.cpp:1483-1507`）先设置 `destroying`，调用 platform 回调和 `mglRendererBackendShutdown()`，再由 `mglRendererBackendReleaseOwnedState()` 释放 device、queue、纹理、buffer、sampler、cache 和 owner，最后 `delete backend`。因此下面的交错是合法的竞态：

```text
T1: GetDevice/Get* 在 mutex 内读出 p，解锁
T2: Destroy 设置 destroying，等待最后 submission，release(p)
T1: 调用 p->... 或把 p 传入后续 encoder/ObjC API
```

`GetDevice`、`GetCommandQueue` 和资源 getter 即使都加了 mutex，也不能阻止 T1 在解锁后使用已释放对象；`GetDevice` 目前也没有调用方持有的稳定引用。已落地的 sync teardown barrier 只覆盖 fence API 的 context-level active operation，不覆盖这些 backend getter、renderer category 方法或 backend handle 本身。

#### 影响范围

需要按三类 API 一起处理，不能只修 `GetDevice`：

1. **核心身份与队列**：`mglRendererBackendGetDevice()`、`mglRendererBackendGetCommandQueue()`，以及 `MGL/include/MGLRenderer_Private.h` 中 `_device`、`_commandQueue`、`_commandQueueOwner` 宏。它们在多个 ObjC category 中被隐式重复求值。
2. **backend-owned Metal 资源与缓存**：fallback/transient/default draw-buffer texture，fallback/cull/current-attrib/packed-attrib/size-constant buffer，blit sampler/depth state，passthrough function，sampler snapshot，tessellation/capture/XFB/TCS buffer，以及 fallback resource/sampled-texture cache。对应调用主要在 `MGLRenderer+RenderPass.m`、`+Blit.m`、`+BindingState.m`、`+Tessellation.m`、`+DrawSupport.m`、`+Texture.m`、`+BatchReplay.m`。
3. **opaque runtime owner**：`mglRendererBackendGetOwner()` 返回 command queue/buffer、render encoder/pass、query、recovery、binding owner 的 C++ 裸指针。`MGLRenderer+Lifecycle.m` 会把 binding/query/recovery owner 缓存到 ObjC 状态，`mgl_render.cpp` 还会转发 `GetOwner`；这些指针也必须受同一生命周期协议保护。

另外，`GetStageCopyBackResources()`、`GetFallbackSampledTexture()`、`GetTessFactorBuffer()` 和 `GetTessXfbDummyBuffer()` 通过 out 参数返回的对象仍是 borrowed；“返回命中状态”不改变对象的生命周期语义。任何把 getter 结果写入 ObjC ivar、C++ command structure、异步回调或跨线程队列的代码，都已经超出当前借用约定的安全范围。

#### 推荐的两层生命周期模型

推荐先建立统一的 **backend operation lease**，再决定少数跨 scope 对象是否需要额外 retain/shared ownership：

- lease 获取是 backend handle 的一次可失败操作。获取时检查 `destroying`/关闭 gate，并递增 active lease；销毁设置 gate 后拒绝新 lease，等待 active lease 归零，再释放 owned state 和 backend handle。
- 一个 lease 必须覆盖从 getter 取值到最后一次使用的完整区间，包括调用到 Metal-cpp、ObjC bridge、encoder 提交和 owner helper 的过程。getter 不再单独声称“锁内读安全”。
- 需要跨越 lease、存入长期状态或交给异步 command 的对象，使用显式 owned 返回：Metal-cpp 对象通过 `retain`/`release` 配对，或封装成项目内的 shared handle；C++ owner 使用引用计数/shared ownership，而不是把 `GetOwner` 裸指针缓存到 teardown 之后。
- 不建议只延长每个 getter 的 mutex 临界区。调用者的使用发生在 getter 返回之后，且长时间持锁会把资源创建、编码和销毁耦合在一把锁上。

可以先以 C ABI 形态落地，后续在 C++/ObjC 层包 RAII：

```c
typedef struct MGLRendererBackendLease {
    MGLRendererBackendHandle *backend;
    uint64_t generation;
} MGLRendererBackendLease;

int mglRendererBackendBegin(MGLRendererBackendHandle *,
                            MGLRendererBackendLease *);
void mglRendererBackendEnd(MGLRendererBackendLease *);
void *mglRendererBackendLeaseGetDevice(
    const MGLRendererBackendLease *);
```

实现时必须保证 lease 获取本身发生在 handle 仍然稳定可访问的 owner/context 保护下；否则调用方可能在读取 `backend` 指针前就遇到 `delete backend`。lease 生命周期之外不得保存其中的 borrowed pointer；跨 scope 必须调用明确的 `Retain`/`Release` 或 owned getter。还要检查 platform destroy 回调是否反向请求 lease，避免关闭 gate 后的回调死锁。

#### 建议的迁移顺序

1. 在 backend 内增加 active-lease 计数、关闭 gate、条件变量和 generation/debug 断言；保留旧 getter 以便分阶段迁移。
2. 先迁移 `_device`、`_commandQueue`、`_commandQueueOwner` 以及 `MGLRenderer+Lifecycle.m` 的 owner 缓存，建立 ObjC scope helper/C++ RAII lease。
3. 迁移 RenderPass、Blit、BindingState、Tessellation、DrawSupport、Texture、BatchReplay 中的 cache/fallback/tess getter；同一方法内的多次 getter 应共享一个 lease。
4. 最后迁移 `GetOwner`、stage-copy-back 和所有 cache out 参数；禁止新代码直接缓存 borrowed 返回值。
5. 过渡期把旧入口改名为内部 `*_BorrowedUnsafe` 或加静态检查标记；全部调用点迁移后删除旧 borrowed API，避免再次引入无保护调用。

#### 销毁协议

销毁顺序应固定为：

```text
close gate
detach backend from context/platform
reject new leases
wait active backend leases == 0
stop/wait command submission
destroy render-pass/runtime owners
release Metal-owned cache objects
delete backend handle
```

具体实现可把“阻止新 lease”和“等待 GPU 最后提交”分开，但不能在 active lease 仍存在时执行 `mglRendererBackendReleaseOwnedState()`。platform callback 必须只做 detach/通知，不能在 gate 关闭后同步取得新 lease；若确实需要访问 backend，应改成由 destroy 持有的内部 shutdown token。

#### 验收门禁

- 多线程 getter/use 与 destroy 交错 stress test；覆盖 device、queue、texture、buffer、cache out 参数和 owner 全类别。
- 重复 create/use/destroy loop，并加入 cache replacement、command-queue reset、platform callback 和 renderer `dealloc` 的交错。
- ASan/TSan（或 macOS 等价工具）运行上述测试；Metal retain/release 计数须无 UAF、double release 和泄漏。
- 静态审计所有 `mglRendererBackendGet*` 调用：每个调用必须位于 lease scope，或明确使用 owned getter；ObjC ivar、C++ command structure 和异步任务不得保存未说明的 borrowed 指针。
- 重新运行现有 regression、arch-correctness、指定 KHR-GL46 case，并在实现完成后用用户给出的逐 case runner 重跑相关 GL46 分组；CTS 的工作目录必须保持为 `.../external/openglcts/modules`。

#### 当前状态与接手边界

此项已为 **P1 landed**。backend 提供 `Begin`/`End` lease、generation 与 `active_leases` drain；borrowed `Get*` 与 `LeaseGet*` 要求调用线程持有匹配 lease；`Destroy` 在释放 owned state 前等待 lease 归零。ObjC 入口与 Lifecycle init 已纳入协议。跨 lease 的长期缓存仍应避免保存未说明的 borrowed 指针；后续可将剩余内部调用点继续收口到显式 `LeaseGet*` / RAII。

### 当前证据

- `make -j4 lib`：通过。
- `make test-mglsema`：67/67 通过（含 flattened SSBO stride 门禁）。
- `compute_dispatch_ssbo`：隔离运行 1/1 通过；完整套件同步为 91 PASS / 0 FAIL / 2 SKIP。
- `test_regression`：完整运行 91 PASS / 0 FAIL / 2 SKIP / 93。
- `test-es-smoke`：`es-smoke: ok`。
- `test-dirty-hash`：`dirty-hash batch regression: PASS`。
- `test-arch-correctness`：本轮此前已通过。
- KHR-GL46 `geometry_shader.layered_rendering.layered_rendering`：按指定 CTS modules 工作目录复跑 **1/1 Pass**。完整 GL46 CTS 尚未在本轮重跑。
- 本轮收口后的复验：`make -j4 lib`、`make test-mglparse`（74/74）、`make test-mglsema`（67/67）、`make test-arch-correctness`、`make test-es-smoke`、`make test-dirty-hash` 均通过；同一 KHR-GL46 case 再跑 **1/1 Pass**。
