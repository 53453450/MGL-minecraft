# MGL ObjC Category 拆解 TODO

> 目标：**ObjC 只保留薄平台层**（CAMetalLayer / drawable / present / view 几何 / 主线程同步 / 少量 `id` 物化端口）。
> 编排、策略、enum 映射、plan、encode、hazard、PSO 决策一律在 C / C++（`mgl_render*`、`mgl_draw_*`、`mgl_tess_*`）。
>
> 基线：`53453450/MGL-minecraft` @ `25b9338`（2026-09-08；docs/O0 合入时 HEAD）
> 对齐：`docs/ARCHITECTURE_REVIEW.md`（目标「薄 ObjC 端口：layer / drawable / swap」）
> 并列：`CTS_FIX_POLICY.md` / `CTS_REFACTOR_SPLIT.md`（禁则与域拆分不冲突；本文件管 **边界厚度**）

---

## 0. 薄平台层定义（Keep 清单）

ObjC **允许**保留的职责（理想终态每个入口 ≤ ~50–150 行）：

| 职责 | 现有落点 | 说明 |
|------|----------|------|
| `CAMetalLayer` / drawable 获取与 present | `MGLPlatformRendererShell`、Lifecycle / Swap | vsync、`displaySyncEnabled`、unlocked skip-present |
| View / window 几何与主线程 sync | `+Lifecycle` | KVO / notification → 回写 framebuffer size |
| Device / queue 生命周期 glue | `+Lifecycle` + lease | 只调 `mglRendererBackend*`；不缓存无 lease borrowed 指针 |
| GPU recovery **触发口** | `+GPURecovery` | 检测 + 调 C++ reset；策略在 C++ |
| `id` 物化端口（薄） | 若干 category 入口 | `createMTLTexture` / `createMTLSampler` / `newCommandBuffer` 等：**参数由 C++ plan，ObjC 只 new/retain** |
| C ABI → ObjC 一行转发 | `mtlDraw*` 等 | 已部分完成；继续压成一行 `mglIssue*` |

ObjC **禁止**再增长（与 ARCH「不要保留」一致）：

- 巨型 category 当 encoder / state machine
- GL enum ↔ Metal 映射、readback 分类、PSO format-class 选择
- tess/GS/XFB/cull plan、batch path 决策、hazard overflow 策略
- CTS-shaped 特判、环境变量污染 Core 语义（见 CTS 禁则 B1–B6）

**验收口径（整体）**

- `MGLRenderer+*.m` 合计从 ~59k → **目标 ≤ 8–12k**（平台壳 + 薄物化）
- 每个 domain category 要么删除，要么降到「调用 C ABI + 物化 `id`」
- 新增逻辑默认进 C/C++；ObjC PR 必须证明属于 Keep 清单

---

## 1. 现状库存（按厚度）

### 1.1 `MGLRenderer` categories（约 59k）

| 文件 | 约 LOC | 判定 | 终态 |
|------|-------:|------|------|
| `+Texture.m` | ~6990 | **厚** | 拆：upload/readback/fallback plan → C++；ObjC 只 `newTexture` / blit encode 端口 |
| `+RenderPass.m` | ~6959 | **厚** | 拆：`processGLState` / load-store / PSO desc 填表 → C++；ObjC 只 `MTLRenderPassDescriptor` 物化 |
| `+Blit.m` | ~4969 | **厚** | 拆：clip/format/DS unify plan → 已有 sink 方向；ObjC 只 blit encoder 端口 |
| `+BindingState.m` | ~4675 | **厚** | 拆：slot/stage/UBO/SSBO 表 → C++；ObjC 只 `setVertexBuffer` 等绑定口 |
| `MGLRenderer.m` | ~4473 | **厚** | 收口：删已迁走的死 `#pragma`；只留公共入口与少量 utility |
| `+DrawSupport.m` | ~350 | 薄 | O1.6：id 端口 → `mgl_draw_metal_port.m`；host ABI/cull/MS → StageHost；Support 仅 resolve/raster/polygon/ensure |
| `+DrawStageHost.m` | ~360 | 薄 | O1.4 residual2：capture/validate/cull encode → C++；HostOps → `mgl_draw_metal_port.m`；仅 bindCull/MS + 一行包装 |
| `mgl_draw_metal_port.m` | ~1950 | 薄端口+HostOps | O1.6 id 物化 + O1.4 residual2 StageHost HostOps/ABI |
| `+Batch.m` | ~2097 | **厚** | path 决策（MDI/stream/ICB）→ C；ObjC 只 enqueue/flush 端口 |
| `+Tessellation.m` | ~1766 | 中→薄 | O1.4：编排在 `mglTessRunPatchDraw`；ObjC 仅 dispatch/物化口 |
| `+BatchReplay.m` | ~1602 | **厚** | replay 绑定与 stage 分支 → C++；ObjC 一行 replay 口 |
| `+Buffer.m` | ~1575 | 中 | map/CoW/shadow plan → C++；ObjC 只 MTLBuffer 物化 |
| `+Compute.m` | ~1255 | 中 | dispatch plan → C++；ObjC 只 compute encoder 端口 |
| `+Lifecycle.m` | ~665 | **Keep 核心** | 压到 shell：init/bind/view/lease/dealloc |
| `+SwapDiagnostics.m` | ~555 | Keep/旁路 | 诊断可留 ObjC 或迁 trace；非热路径 |
| `+Draw.m` | ~511 | 薄 | O1.5：`mtlDraw*` 一行 → `mglIssue*` / MS guard |
| `+Binding.m` | ~408 | 薄化 | 与 BindingState 合并后删除 |
| `+GPURecovery.m` | ~350 | Keep 薄 | 触发 + 日志；reset 在 C++ |
| `+VertexLayout.m` | ~332 | 薄化 | descriptor plan 已在 sink；ObjC 删策略 |

### 1.2 其它 `.m`（非 category，但同边界）

| 文件 | 约 LOC | 判定 |
|------|-------:|------|
| `mgl_draw_encode.m` | ~1225 | **迁出**：应并入 / 对齐 `mgl_draw_encode` C++，ObjC 不留 encode |
| `hash_table.m` | ~854 | 平台资源表；可保留或 C++ owner |
| `MGLRenderPassManager.m` | ~523 | 并入 RenderPass 下沉 |
| `MGLPipelineCache.m` | ~445 | 并入 PSO builder（CTS Batch 4） |
| `MGLPlatformRendererShell.m` | ~229 | **Keep 样板** |
| `mgl_readback.m` 等 compat | 小 | 策略进 ReadbackPolicy；`.m` 变转发 |

---

## 2. 拆解批次 TODO（执行顺序）

原则：**先拔编排宿主，再拔物化策略，最后收口 category 文件删除**。与正在进行的 `refactor(*): sink` 同向，但要 **按域落文件**，避免只堆进 `mgl_render.cpp` 神文件。

### Batch O0 — 政策与度量（0.5–1d）

- [x] **O0.1** 本文件合入 `docs/`（或 PR 旁路）；在 `ARCHITECTURE_REVIEW.md` 链到本节
- [x] **O0.2** 加度量脚本 / CI 注释：`wc -l MGL/src/MGLRenderer*.m` + 阈值阈值；目标线写入 README 或 ARCH
- [x] **O0.3** 清单标注 `TECH_DEBT(objc-thick)`：凡 ObjC 内仍有 GL 映射 / plan / 特判的符号
- [x] **O0.4** 禁则（本边界）：
  - 禁止新逻辑只进 `MGLRenderer+*.m` 而不提供 C ABI
  - 禁止「sink 进 `mgl_render.cpp`」却不按域拆头文件（与 CTS Batch 2–4 对齐）
  - 禁止 ObjC 长期缓存无 lease 的 borrowed `id` / owner 指针

**O0.3 `TECH_DEBT(objc-thick)` 初标（仍厚，待后续 O# 清）**

| 符号 / 区域 | 文件 | 债因 |
|-------------|------|------|
| `runVertexCaptureSession` / `captureAIRVertexPositions*` | `mgl_draw_tess` + metal_port | O1.4 residual2：编排在 `mglTessRunVertexCapture*`；ObjC 一行包装 |
| `mglDrawHostGsExecuteMetalExpansion` | `mgl_draw_metal_port.m` | O1.4 residual：薄 HostOps 转发 → `mglDrawGsExecuteMetalExpansion` |
| `bindCullDistanceEmulationBuffers` VAO resolve 口 | `+DrawStageHost.m` | ObjC 只填 port 表；layout 已在 C++（O1.3）；cull encode 在 `mgl_draw_cull.cpp` |
| `scheduleDrawBatch` 物化口 | `+Batch.m` | 决策已在 `mgl_batch_select_path`；ObjC 填 inputs（O2.1） |
| `processGLState` / `processGLStateLocked` | `+RenderPass.m` | O1.1：编排在 `mgl_render_pass_plan`；ObjC 物化 MTL* |
| Texture/Blit/BindingState 巨型 category | 见 §1.1 | O3–O4 |

度量：`scripts/objc_renderer_loc.sh`（目标 `MGLRenderer*.m` 合计 ≤ 8–12k）。

### Batch O1 — Draw / Tess / GS 宿主清空（对齐 ARCH「下一批」）【P0】

目标：`+DrawSupport` / `+Tessellation` / `+Draw` 不再拥有 session 状态机。

- [x] **O1.1** `processGLState` / `processGLStateLocked` → `mglRenderProcessGLState` / `AfterDirty`（`mgl_render_pass_plan.*`）；ObjC 只物化 MTL*；`test-process-gl-state-plan`
- [x] **O1.2** VS GPU capture — **续完 session**：`mglTessRunCaptureSession` 拥有双遍 processGLState+bind；MTLBuffer 分配仍是 ObjC 物化口
- [x] **O1.3** VAO cull attrib resolve → C++（ObjC 只提供 VAO 指针表） — `mglRenderBuildCullDistanceLayoutFromPorts`；ObjC 只 resolve→port 表 + last-bound 记账
- [x] **O1.4** 双路径收口：删除 ObjC `handleTessellation*` / `handleGeometry*` / `handleVertexTransformFeedback*`；XFB+TES 编排在 `mglXfbRunVsOnlyDraw` / `mglTessRunPatchDraw`（HostOps）；GS 早段拓扑/gather/capture 在 `mglDrawGsRunDraw`；Metal 扩张在 `mglDrawGsExecuteMetalExpansion`（`mgl_draw_gs_metal.cpp`），ObjC 仅薄 HostOps
- [x] **O1.4 residual** GS Metal expansion HostOps：PSO/XFB scatter/passthrough encode 下沉；StageHost 2562→~1986
- [x] **O1.4 residual2** capture AIR cull/vertex + validate arrays → `mgl_draw_cull` / `mgl_draw_tess` / `mgl_draw_issue`+`mgl_draw_validate`；HostOps → metal_port；StageHost ~1986→~360；`test-validate-arrays-early`
- [x] **O1.5** `+Draw.m`：`mtlDraw*` 一行转发；Locked 删除；MS sample loop 进 `mglDrawHostGuardIssue*`
- [x] **O1.6** 验收：`+DrawSupport.m` &lt; 400 LOC（~350）：id 端口 → `mgl_draw_metal_port.m`；host ABI/cull/MS → `+DrawStageHost`；Support 仅 resolve/raster/polygon/ensure

### Batch O2 — Batch / Replay 决策下沉【P0】

- [x] **O2.1** `scheduleDrawBatch` 决策树（DIRECT / MDI / STREAM_MERGE / ICB）→ 纯 C `mgl_batch_select_path`（可单测，无 Metal） — `test_legacy_compat/test_batch_path.c` / `make test-batch-path`
- [ ] **O2.2** hazard overflow 策略（sticky vs flush-and-continue）→ C；ObjC 不设语义
- [ ] **O2.3** `+BatchReplay` stage/bind 展开 → C++；ObjC 只 `set*Bytes` / `draw*` 端口
- [ ] **O2.4** ICB：batch 与 `supportIndirectCommandBuffers` 门闩同层配置（避免再出现 ENABLE_BATCH / ENABLE_PIPELINES 分裂）
- [ ] **O2.5** 验收：`+Batch*.m` 合计 &lt; 600 LOC；MC 路径可用 env 金样 / benchmark 回归

### Batch O3 — RenderPass / PSO / Binding【P1】

- [ ] **O3.1** load/store / clear / attachment match → `mgl_render_pass_plan.*`
- [ ] **O3.2** `generatePipelineDescriptorState` → format-class PSO builder（CTS Batch 4）
- [ ] **O3.3** `+BindingState` / `+Binding` 合并下沉 slot 表；ObjC 绑定口 &lt; 300 LOC
- [ ] **O3.4** `MGLPipelineCache.m` → C++ LRU cache（兼 FPS 掉帧 P0）
- [ ] **O3.5** 验收：`+RenderPass.m` &lt; 800 LOC；PSO miss 行为有非 CTS 单测

### Batch O4 — Texture / Blit / Readback【P1】（对齐 CTS ReadbackPolicy）

- [ ] **O4.1** Y-flip / MSAA resolve / integer·depth pack → `mgl_readback_policy.*`（CTS Batch 2）
- [ ] **O4.2** upload dirty / 3D / array / texel buffer plan → C++；ObjC 只 `replaceRegion` / blit
- [ ] **O4.3** fallback sampled texture 选择 → format/type class 表，禁止散落 `if`
- [ ] **O4.4** `+Blit` 剩余 format/DS unify（延续 sink）→ `mgl_blit_plan.*`
- [ ] **O4.5** 验收：`+Texture.m`+`+Blit.m` &lt; 1.5k；readback 金样不依赖 CTS oracle

### Batch O5 — Buffer / Compute / VertexLayout / 杂项【P2】

- [ ] **O5.1** Buffer map / CoW / UBO isolate → C++；cap CoW（兼 FPS 方案）
- [ ] **O5.2** Compute binding expansion → C++；ObjC 只 dispatch
- [ ] **O5.3** `+VertexLayout` 删除或 &lt; 100 LOC
- [ ] **O5.4** `mgl_draw_encode.m` 迁空或删除
- [ ] **O5.5** compat `.m`（sampler/texture/state）变纯转发

### Batch O6 — Category 物理删除与壳收口【P2】

- [ ] **O6.1** 合并剩余端口到少量文件：
  - `MGLPlatformRendererShell.m`（layer/drawable/swap/view）
  - `MGLRenderer+Lifecycle.m`（create/bind/lease/dealloc）
  - `MGLRenderer+MetalPort.m`（可选：所有 `id` 物化一行口）
  - `MGLRenderer+GPURecovery.m`（薄）
- [ ] **O6.2** 删除空 category：`+DrawSupport` / `+BatchReplay` / `+Binding` / `+VertexLayout` / …
- [ ] **O6.3** `MGLRenderer.m` 降到入口表 + 文档化 C ABI
- [ ] **O6.4** 总 LOC 达标；ARCH 表格更新为「薄平台壳」

---

## 3. 与并行工作流的衔接

| 并行线 | 关系 |
|--------|------|
| 当前 `refactor(*): sink …` | **同向燃料**；本 TODO 要求 sink **落地到域文件**，不要只进 `mgl_render.cpp` |
| CTS Batch 1 TessDomain | **已基本落地**（`mgl_tess_domain*`）；O1 清 ObjC 宿主残留 |
| CTS Batch 2 ReadbackPolicy | **驱动 O4** |
| CTS Batch 3 LimitsTable | 不进 ObjC；C 状态机 |
| CTS Batch 4 PSO format-class | **驱动 O3.2 / O3.4** |
| MC FPS：hazard / PSO LRU / Y-flip / CoW | 分别挂在 O2.2 / O3.4 / O4.1 / O5.1 |

建议 PR 节奏：**O0 docs → O1 draw host → O2 batch →（CTS readback ∥ O4）→ O3 PSO/bind → O5 → O6 删文件**。

---

## 4. 每 PR 检查清单（短）

1. 改动的逻辑属于 Keep 清单吗？否则必须有新的/已有 C ABI。
2. 有无 Metal 金样或 property 测试吗？（禁止「只靠 CTS Fail 数」）
3. ObjC 文件 LOC 是否下降（或持平但策略行减少）？
4. 有无 lease 范围内使用 Metal 对象吗？
5. 有无 `TECH_DEBT(objc-thick)` / CTS 禁则冲突吗？

---

## 5. 即时下一刀（建议本周）

1. ~~**O1.1–O1.6**~~：DrawSupport &lt;400；XFB/TES HostOps 已下沉
2. ~~**O1.4 residual**~~：`mglDrawGsExecuteMetalExpansion` + `mgl_draw_gs_metal.cpp`
3. ~~**O1.4 residual2**~~：capture/validate/cull → C++；HostOps → metal_port；StageHost ~360（bindCull/MS + 一行包装）
4. ~~**O2.1**~~：`mgl_batch_select_path` + `test-batch-path` 已合入
5. ~~**O0**~~：本文与 `scripts/objc_renderer_loc.sh` 已挂进 `docs/` / ARCH / README
6. **下一刀**：O2.2 hazard overflow；O2.3 BatchReplay stage/bind；可考虑删除 StageHost（并入 metal_port / Support）

完成以上后，再大规模继续 sink 也不会失去「薄平台层」方向感。
