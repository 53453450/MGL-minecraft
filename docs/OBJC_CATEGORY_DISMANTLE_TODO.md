# MGL ObjC 清零 TODO

> **目标（2026-09-12 重定）：MGL 内 ObjC 清零。**
> 终态 = `MGL/` 下**不存在** `.m` / `.mm` 文件，也不存在 ObjC 语法与 ObjC 词汇
> （`@interface` / 消息发送 / `#import` / `__bridge` / `NS*` / `MTL*` ObjC 类型 / `BOOL`·`YES`·`NO`·`nil`·
> `NSUInteger`）；引擎全部为 **C 或 C++（Metal-cpp）**。
>
> 允许的**唯一**例外是平台壳（`NSWindow` / `CAMetalLayer` / drawable / 主线程同步）：若确实无法移出，
> 只允许存在**一个**平台壳 TU，且必须写明行数上限与移除路径（见 T5）；除此之外任何新代码不得引入 ObjC。
>
> 基线：`53453450/MGL-minecraft` @ `8e64afb`（2026-09-12；本目标重定前的 HEAD）
> 对齐：`docs/ARCHITECTURE_REVIEW.md`（层规模与依赖域）
> 并列：`CTS_FIX_POLICY.md` / `CTS_REFACTOR_SPLIT.md`（禁则与域拆分不冲突；本文件管 **ObjC 归零**）

---

## 0. 终态定义与分层（T0–T5）

「清零」不是一次性动作，按"改动风险 × 可验证性"分五层推进，每层都要过 §0.2 的验证口径。

| 层 | 内容 | 判据（DoD） | 规模（2026-09-12 实测） |
|---|---|---|---|
| **T0 ✅** | 空 TU 删除：只有注释的 `.m` | 文件消失；构建 + `make test-all` 通过 | 3 个文件 / 48 行（**已完成**） |
| **T1 ✅** | 仅 `#import` 算 ObjC 语法的 `.m` → `.c`（改 `#include`，去掉无用 `<Foundation/Foundation.h>`） | 全部为 `.c`；构建 + `make test-all` + hotspot 非通过集合 diff 为空 | 10 个文件 / 2,188 行（**已完成**） |
| **T2 ✅** | 无 ObjC 语法但有词汇：`BOOL`/`YES`/`NO`/`nil`/`NSUInteger`/`NSLog` 换成 C 等价物后改名 | 同上 + 该文件 ObjC 词汇清零 | 10 个文件 / 1,709 行（**已完成**） |
| **T3** | 决策下沉：category 里的 policy / plan / enum 映射 / 分类搬进 C 模块（**沿用 §2 的 Batch O1–O7**） | 每个 domain：决策在 C、有 golden harness、ObjC 只剩物化端口 | `MGLRenderer*.m` 34,610 → 逐批下降 |
| **T4** | 端口 C++ 化：`.m` 端口改 C++（Metal-cpp），`id` → `void*`，`MTL*` ObjC 类型 → Metal-cpp 类型 | `MGL/src` 内不再有 `.m`（平台壳除外） | 剩余 category + `MGLRenderer.m` + `mgl_draw_metal_port.m` 等 |
| **T5** | 平台壳：`NSWindow`/`CAMetalLayer`/drawable/present/主线程同步 | 二选一并记录：**(a)** 用 ObjC runtime C API（`objc_msgSend`）在 C++ 内实现，`MGL/` 内 0 个 `.m`；**(b)** 移交消费方，`MGL/` 内 0 个 `.m` | 现 `MGLPlatformRendererShell.m` 229 行 + `+Lifecycle` 665 行 |

**禁则（加严）**：除 T5 允许的那一个平台壳 TU 外，**任何新代码不得新增 ObjC**；不得以"先加后清"为由在 `.m` 里加逻辑；
新增策略/plan/映射一律进 C/C++ 模块并配 harness。

### 0.0 清零度量（可复现）

`scripts/objc_zero.sh` 输出四组数字：`.m/.mm` 文件数、空 TU 数、ObjC 语法出现次数、ObjC 词汇出现次数，
外加 `MGLRenderer*.m` 合计（历史指标，保留以便对照）。基线（2026-09-12 @ `8e64afb`）：

| 指标 | 基线 | 终态 |
|---|---|---|
| `MGL/` 内 `.m` 文件数 | **53** | 0（或 T5 的 1 个平台壳） |
| 其中空 TU | **3** | 0 |
| ObjC 文件行数 | **43,989** | ≈ 平台壳 |
| ObjC 语法出现次数（含 `#import`） | **2,268** | 0 |
| ObjC 词汇出现次数 | **4,353** | 0 |
| `MGLRenderer*.m` total | **34,604** | 0 |

**当前进度（2026-09-12，T0+T1+T2 完成后）**：文件 **53 → 30**、空 TU **3 → 0**、行数 **43,989 → 40,044**、
ObjC 语法 **2,268 → 2,239**、词汇 **4,353 → 4,266**。（T2 的 10 个文件词汇已清零，剩余词汇全在 30 个真 ObjC 文件里。）

---

## 0.1 文档地图（2026-09-12 整理）

| 文档 | 入库 | 作用 |
|---|---|---|
| [`OBJC_CATEGORY_DISMANTLE_TODO.md`](OBJC_CATEGORY_DISMANTLE_TODO.md)（本文） | ✅ | 薄 ObjC 边界：政策/度量/批次清单 + §5 落地日志（连续编号；最新可执行清单见第 29 条与 Batch O7） |
| [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) | ✅ | 总架构审查与分层规模；ObjC 厚度数字以本文度量为准 |
| [`C0_AIR_RENDER_DEP_MAP.md`](C0_AIR_RENDER_DEP_MAP.md) | ✅ | `mgl_air_backend.cpp` / `mgl_render.cpp` 的依赖与调用域地图 |
| `TESS_NATIVE_RENDER_VERTEX_PATH.md`（本地） | ❌ | TES-vertex / compute 双路线设计 + §10.4 的 PSO 键修复记录（2026-09-12）；按文件名引用，不入库 |
| `CTS_TESS_REMAINING_2026-09-10.md`（本地） | ❌ | tess 簇逐轮排查日志 + 单例 ground-truth 复现方法（顶部有当前状态横幅：139/1/0）；按文件名引用，不入库 |
| `GS_XFB_GL4_CTS_PROGRESS_2026-08-21.md`（本地） | ❌ | GS/XFB 簇进度切片 + 批跑命令模板；按文件名引用，不入库 |
| [`AIR_M3_CPP_TODO.md`](AIR_M3_CPP_TODO.md) · [`P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`](P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md) · [`GL46_GS_LAYERED_SPEC_AUDIT.md`](GL46_GS_LAYERED_SPEC_AUDIT.md) | ✅ | 专题记录 |
| 其余 `docs/*.md` | ❌（`.gitignore` 的 `/docs/*`） | 阶段性审计/审查稿与逐轮工作日志，**一律留在本地**；入库文档引用它们时只用文件名（标注"本地"），不随引用一起入库 |

度量脚本：[`scripts/objc_renderer_loc.sh`](../scripts/objc_renderer_loc.sh)（`MGLRenderer*.m` 合计、Batch 诚实簇、
Draw 簇；当前输出 `MGLRenderer*.m total: 34610`）。

## 0.2 验证口径（本周期每刀都按这三套语料报数）

| 语料 | 入口 | 通过口径 |
|---|---|---|
| 本地功能/回归 | `build/test_regression all`（94 项） | **92 PASS / 0 FAIL / 2 SKIP**（窗口/环境门控 2 项）；单例：`build/test_regression <name>` |
| 本地聚合门禁 | **`make test-all`（必须跑）** | 返回 0；含 `test-metalcpp` / `test-es-smoke` / `test-mcrepro` 等 smoke 目标——**2026-09-12 的教训**：O7.4 把 `mglProgramStageBuiltinMask` 搬进 `mgl_program_resource.c` 后，`test_metalcpp_smoke` 因缺该文件**链接失败**，而当时的验证清单只挑着跑 harness、没跑聚合目标，破损被漏掉数小时 |
| 本地单元 harness | `make test-all`（含 `test-tess-domain` / `test-tess-air` / `test-buffer-plan` / `test-reference-query` / `test-binding-*` / `test-batch-*` / `test-render-pass-clear-plan` …） | 各自 `ok` / `0 failure` |
| CTS tess 簇 | caselist `VK-GL-CTS-build-mgl-target/mgl-tess-cluster-cases.txt`（140 例） | **139 pass / 1 fail / 0 ns**；唯一失败为已归档的 FO-spacing CTS 期望矛盾 |
| CTS GS 簇 | caselist `…/mgl-gs-cluster-cases.txt`（136 例，由上一轮 GS run 的 `summary.tsv` 复刻） | **136 / 0** |
| GL46 hotspot | caselist `…/khr-gl46-fbo-hotspot.cases.txt`（1328 例） | 通过数可持平，但**必须给"非通过集合逐条 diff 为空"**（否则不算无回归） |

批跑模板（在 CTS build 目录执行；`--dyld-library-path` 指向仓库根——那里的 `libmgl.dylib` 是
`build/libmgl.dylib` 的软链，所以**先重建、再跑，跑的过程中不要再重建**，否则中途换库会让结果不可解释）：

```sh
python3 -u run_mgl_cts_cases.py \
  --glcts  $PWD/external/openglcts/modules/glcts \
  --caselist $PWD/mgl-tess-cluster-cases.txt \
  --workdir  $PWD/external/openglcts/modules \
  --outdir   $PWD/mgl-<tag>-$(date +%H%M%S) \
  --dyld-library-path /Users/<you>/MGL-minecraft \
  --timeout 45 --progress-every 70
```

非通过集合对比（hotspot 无回归的判据）：

```sh
awk -F'\t' 'NR>1 && $3!="pass" {print $2"\t"$3}' <outdir>/summary.tsv | sort > /tmp/nonpass_now.txt
diff /tmp/nonpass_baseline.txt /tmp/nonpass_now.txt   # 必须为空
```

单例复跑（定位单个 case，含打开 CTS 侧 trace 的方法）见
本地日志 `docs/CTS_TESS_REMAINING_2026-09-10.md` 的「复现 ground truth 的方法」。

## 1. 清零库存

### 1.0 ObjC 文件分层清单（2026-09-12 实测，脚本口径见 §0.0）

（口径：`scripts/objc_zero.sh`；`语法`含 `#import`，`词汇`＝`BOOL/YES/NO/nil/NSUInteger/NSLog/NS*/MTL*`）

| 层 | 文件数 / 行数 | 文件 |
|---|---|---|
| **T0 空 TU ✅ 已删** | 3 / 48 | ~~`MGLBindingSync.m`~~ · ~~`MGLQueryManager.m`~~ · ~~`MGLTextures.m`~~ |
| **T1 仅 `#import` ✅ 已改名** | 10 / 2,188 | `hash_table.m`(855) · `mgl_texture_compat.m`(331) · `mgl_sampler_compat.m`(324) · `mgl_sync.m`(108) · `mgl_rt_sync.m`(104) · `mgl_capability.m`(100) · `mgl_coordinate.m`(99) · `mgl_focus_program.m`(97) · `mgl_shader_resource.m`(94) · `mgl_state_log.m`(76) |
| **T2 有词汇无语法 ✅ 已改名** | 10 / 1,709 | `mgl_binding_texture_log.m`(328,v16) · `mgl_frame_activity.m`(288,v8) · `mgl_trace_strategy.m`(229,v16) · `mgl_state_compat.m`(184,v10) · `mgl_vertex_format.m`(147,v1) · `mgl_byte_hash.m`(142,v1) · `mgl_vertex_attrib_query.m`(133,v8) · `mgl_draw_buffer.m`(94,v5) · `mgl_blit_clip.m`(90,v9) · `mgl_buffer_query.m`(74,v8) |
| **T3/T4 真 ObjC（当前唯一剩余）** | 30 / 40,044 | `+RenderPass`(426 语法/558 词汇) · `+Texture`(312/1103) · `+Blit`(244/880) · `MGLRenderer`(172/282) · `+BindingState`(136/198) · `+Tessellation`(153/292) · `mgl_draw_metal_port`(116/100) · `+Compute`(90/104) · … |
| **T5 平台壳** | 1 / 229 | `MGLPlatformRendererShell.m`（归 T4/T5 处理） |

### 1.1 现状库存（按厚度）

### 1.1 `MGLRenderer` categories（2026-09-12 实测 **34.5k**；基线 ~59k）

| 文件 | 约 LOC | 判定 | 终态 |
|------|-------:|------|------|
| `+Texture.m` | 6981 | **厚** | 拆：upload/readback/fallback plan → C++；ObjC 只 `newTexture` / blit encode 端口（值类型/构造器已沉 `mgl_region_value.cpp`，O4.0） |
| `+RenderPass.m` | 7125 | **厚** | 拆：load-store / clear / attachment match → `mgl_render_pass_plan.*`（**O3.1 进行中**：clear-value、load/store 动作表、stale-clear 规则、**两半** attachment match 均已沉、各有 golden；**残量**＝attachment 物化/解析回调化以压 LOC）；ObjC 只 `MTLRenderPassDescriptor` 物化 |
| `+Blit.m` | 4941 | **厚** | 拆：clip/format/DS unify plan → `mgl_blit_plan.*`（**O4.4 已启动**：depth/stencil 三道门与拷贝矩形推导已沉 + 58 例 golden；**残量**＝format conversion/unify 与物化回调化）；ObjC 只 blit encoder 端口（值类型/构造器已沉 `mgl_region_value.cpp`，O4.0） |
| `+BindingState.m` | 2940 | **厚**（C1 多刀已从 ~4523 降下，见 O3.3） | 拆：attrib/texture/image apply 残留 → C；ObjC 只 `setVertexBuffer`/`set*Texture` 口（口仍 ≫300，O3.3 residual） |
| `MGLRenderer.m` | 4693 | **厚** | 收口：删已迁走的死 `#pragma`；只留公共入口与少量 utility |
| `+DrawSupport.m` | 350 | 薄 | O1.6：id 端口 → `mgl_draw_metal_port.m`；host ABI/cull/MS → StageHost；Support 仅 resolve/raster/polygon/ensure |
| `+DrawStageHost.m` | 362 | 薄 | A1：保留（非空）；bindCull/MS + 一行包装；GS 扩张已无策略 |
| `mgl_draw_metal_port.m` | 1970 | 薄端口+HostOps | A1：id 物化 + HostOps 表；`metal_ops` 嵌套 |
| `+Batch.m` | 170 | 薄 category / **簇仍厚** | O2.5 字面达标；encode 仍在 `mgl_batch_*_encode.m`（见 §1.2 / Track B） |
| `+Tessellation.m` | 2174 | 中→薄 | O1.4：编排在 `mglTessRunPatchDraw`；ObjC 仅 dispatch/物化口 |
| `+BatchReplay.m` | ~21 | 薄占位 | O2.5：dyn-bind → `mgl_batch_dyn_bind_encode.m`；待 O6 删空 category |
| `+Buffer.m` | 826 | 薄化中 | map/CoW/shadow plan → C++（O5.1：vertex-index + dirty-buffer 决策已沉 C 函数；vertex-attrib buffer map 已整段沉 `mgl_vertex_attrib_plan.*` + harness；**reflection fallback 已删——plan 成为唯一映射路径**）；ObjC 只 MTLBuffer 物化与逐 attribute resolve |
| `+Compute.m` | 1267 | 中 | buffer 绑定环复用 `mgl_binding_stage` plan 形态（PRE/POST + 三个 opt-in 开关 + C 侧判决表）；采样器级联与图形侧收敛为同一 port（复用 `mglBindingTexturePlanSamplerMaterialize`）；**剩余**＝整段纹理循环迁 C++（需物化回调 vtable）；ObjC 只 compute encoder 端口 |
| `+Lifecycle.m` | 665 | **Keep 核心** | 压到 shell：init/bind/view/lease/dealloc |
| `+SwapDiagnostics.m` | 555 | Keep/旁路 | 诊断可留 ObjC 或迁 trace；非热路径 |
| `+Draw.m` | 511 | 薄 | O1.5：`mtlDraw*` 一行 → `mglIssue*` / MS guard |
| `+Binding.m` | 488 | 薄化（实测较基线 +80；与 BindingState 合并后删除） | 与 BindingState 合并后删除 |
| `+GPURecovery.m` | 350 | Keep 薄 | 触发 + 日志；reset 在 C++ |
| `+VertexLayout.m` | 203 | 薄化（O5.3 本刀：`generateVertexDescriptorState` 已沉 C 函数 `mglRenderGenerateVertexDescriptorState`，ObjC 仅薄转发） | `updateBlendStateCache` 写 `_pipelineCache`（ObjC 物化，留）；`bindFramebufferAttachmentTextures` 实为 FBO 绑定，应归 RenderPass 域 |

### 1.2 其它 `.m`（非 category，但同边界）

| 文件 | 约 LOC | 判定 |
|------|-------:|------|
| `mgl_draw_encode.cpp` | ~1187 | **已迁出 ObjC**（O5.4 DONE）：原 `.m` 整文件重命名为 `.cpp`，剥除 6 处 `__bridge`，经 Makefile `wildcard MGL/src/*.cpp` 自动纳 non-ARC C++；draw-encode 决策不再属 ObjC 边界 |
| `mgl_batch_flush_restore_encode.m` | ~381 | **Batch 簇残量**：flush/restore/stream；`flush_run_batches` / check / trace-skip 已接线；**已呈 ops-callback 薄形**（C++ driver + ObjC 回调接线） |
| `mgl_batch_dyn_bind_encode.m` | ~366 | **Batch 簇残量**：dyn-bind/sampler；`mgl_batch_mtl_bind_dyn_*` / `apply_sampler_snapshot` 已接线 |
| `mgl_batch_issue_encode.m` | ~217 | **Batch 簇残量**：issue/direct；MDI+direct loops → `mgl_batch_mtl_issue_mdi_batch` / `mgl_batch_issue_direct_batch` |
| `mgl_batch_replay_trace.m` | ~270 | **Batch 簇残量**：trace；FS-slot POD 已抽；**禁止再扩** |
| `mgl_batch_icb_mdi_encode.m` | ~177 | **Batch 簇残量**：ICB/stream-MDI；whole loops → `mgl_batch_mtl_issue_*_batch` |
| `mgl_batch_rt_mark_port.m` | ~139 | **Batch 簇残量**：RT-mark；draw-attachments → `mgl_batch_rt_run_draw_attachments` |
| `hash_table.m` | ~854 | 平台资源表；可保留或 C++ owner |
| `MGLRenderPassManager.m` | ~523 | 并入 RenderPass 下沉 |
| `MGLPipelineCache.m` | ~445 | **已薄端口（O3.4 实质完成）**：LRU/archive 策略在 C++ owner（`mglRenderLookupPipeline`/`StorePipeline`）；ObjC 仅 `id`↔`void*` 桥 + archive URL 路径；可按 O3.4 收口标记 [x] |
| `MGLPlatformRendererShell.m` | ~229 | **Keep 样板** |
| `mgl_readback.m` 等 compat | 小 | 策略进 ReadbackPolicy；`.m` 变转发 |

> **实测（2026-09-12）**：`MGLRenderer+*.m` categories 合计 **34.5k** LOC（基线 ~59k；O1/O2/C1 已降 ~24.5k）；距目标 ≤8–12k 仍差 ~3×。`MGLRenderer+Texture.m`+`+RenderPass.m`+`+Blit.m` 三厚块 = **19.0k**（6981+7058+4945），仍是 O4 主体。`MGLPipelineCache`/`+VertexLayout`/`mgl_batch_*_encode` 已呈薄端口/ops 形，不应再计入「待沉厚代码」。
>
> 本周期新增回落（同日多刀，见 §5 第 25–28 条）：`+Buffer.m` 1479→**826**（vertex-attrib buffer map 沉 `mgl_vertex_attrib_plan.*` 且删掉 544 行 reflection fallback）、`+RenderPass.m` 退役 6 处源码文本扫描、3 份 sampler 启发式实现合并为 1 个共享谓词。度量：`scripts/objc_renderer_loc.sh`（当前输出 `MGLRenderer*.m total: 34610`，`+Buffer.m` 822）。

**Batch ObjC 诚实合计（Track B）**：categories ~190 + encode/trace/port ~1521 = **~1711**（`scripts/objc_renderer_loc.sh`）。A3 本刀 1923→~1711（−212；direct-submit C 决策树、trace fill helpers、binding helpers→`+Binding`、sampled resolve gate）；**勿宣称 cleanup done**（残量仍 ~1.7k）。

---

## 2. 拆解批次 TODO（执行顺序）

> 本节 O0–O7 是 **T3（决策下沉）** 的批次清单：把 category / port 里的决策搬进 C 模块。
> T0–T2（文件级清零）在 §0 的层表里逐项勾选；T4/T5（端口与平台壳 C++ 化）在 O6 之后单独立项。

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
| ~~`mglDrawHostGsExecuteMetalExpansion`~~ | — | **A1 DONE**：已删除；`RunDraw` 经嵌套 `metal_ops` 直调 `mglDrawGsExecuteMetalExpansion`；fragment clear 在 C++ |
| `bindCullDistanceEmulationBuffers` VAO resolve 口 | `+DrawStageHost.m` | ObjC 只填 port 表；layout 已在 C++（O1.3）；cull encode 在 `mgl_draw_cull.cpp` |
| `scheduleDrawBatch` 物化口 | `mgl_batch_flush_restore_encode.m` | 决策+batch flags fill 在 C；ObjC 仅 cull/os/env（A3 fold） |
| ~~`flushDrawBufferLocked` 编排环~~ | `mgl_batch_flush_restore_encode.m` | A3 续：`flush_run_batches` + trace-skip；MTL issue 仍 ObjC ops |
| ~~`bindDynamic*Directly` MTL 物化环~~ | `mgl_batch_dyn_bind_encode.m` | A3 续：`mgl_batch_mtl_bind_dyn_*` + `apply_sampler_snapshot`；ensure/resolve 仍 ObjC |
| `processGLState` / `processGLStateLocked` | `+RenderPass.m` | O1.1：编排在 `mgl_render_pass_plan`；ObjC 物化 MTL* |
| Texture/Blit/BindingState 巨型 category | 见 §1.1 | O3–O4 |

度量：`scripts/objc_renderer_loc.sh`（目标 `MGLRenderer*.m` 合计 ≤ 8–12k；Batch 簇按 Track B 诚实口径，含 encode/trace）。

### Batch O1 — Draw / Tess / GS 宿主清空（对齐 ARCH「下一批」）【P0】

目标：`+DrawSupport` / `+Tessellation` / `+Draw` 不再拥有 session 状态机。

- [x] **O1.1** `processGLState` / `processGLStateLocked` → `mglRenderProcessGLState` / `AfterDirty`（`mgl_render_pass_plan.*`）；ObjC 只物化 MTL*；`test-process-gl-state-plan`
- [x] **O1.2** VS GPU capture — **续完 session**：`mglTessRunCaptureSession` 拥有双遍 processGLState+bind；MTLBuffer 分配仍是 ObjC 物化口
- [x] **O1.3** VAO cull attrib resolve → C++（ObjC 只提供 VAO 指针表） — `mglRenderBuildCullDistanceLayoutFromPorts`；ObjC 只 resolve→port 表 + last-bound 记账
- [x] **O1.4** 双路径收口：删除 ObjC `handleTessellation*` / `handleGeometry*` / `handleVertexTransformFeedback*`；XFB+TES 编排在 `mglXfbRunVsOnlyDraw` / `mglTessRunPatchDraw`（HostOps）；GS 早段拓扑/gather/capture 在 `mglDrawGsRunDraw`；Metal 扩张在 `mglDrawGsExecuteMetalExpansion`（`mgl_draw_gs_metal.cpp`），ObjC 仅薄 HostOps
- [x] **O1.4 residual** GS Metal expansion HostOps：PSO/XFB scatter/passthrough encode 下沉；StageHost 2562→~1986
- [x] **O1.4 residual2** capture AIR cull/vertex + validate arrays → `mgl_draw_cull` / `mgl_draw_tess` / `mgl_draw_issue`+`mgl_draw_validate`；HostOps → metal_port；StageHost ~1986→~360；`test-validate-arrays-early`
- [x] **A1** 抽干 `mglDrawHostGsExecuteMetalExpansion`：嵌套 `metal_ops`；`RunDraw` 直调 C++；fragment clear 下沉；StageHost 保持 ~363（&lt;800；bindCull/MS 非空故不删）；禁新厚 category；Draw 簇进 `objc_renderer_loc.sh`
- [x] **O1.5** `+Draw.m`：`mtlDraw*` 一行转发；Locked 删除；MS sample loop 进 `mglDrawHostGuardIssue*`
- [x] **O1.6** 验收：`+DrawSupport.m` &lt; 400 LOC（~350）：id 端口 → `mgl_draw_metal_port.m`；host ABI/cull/MS → `+DrawStageHost`；Support 仅 resolve/raster/polygon/ensure

### Batch O2 — Batch / Replay 决策下沉【P0】

- [x] **O2.1** `scheduleDrawBatch` 决策树（DIRECT / MDI / STREAM_MERGE / ICB）→ 纯 C `mgl_batch_select_path`（可单测，无 Metal） — `test_legacy_compat/test_batch_path.c` / `make test-batch-path`
- [x] **O2.2** hazard overflow 策略（sticky vs flush-and-continue）→ C；ObjC 不设语义 — `mgl_batch_hazard_*` + `test-batch-hazard`；`draw_command` 只执行 action；默认 sticky，`MGL_HAZARD_OVERFLOW_FLUSH_CONTINUE` 选 flush-and-continue
- [x] **O2.3** `+BatchReplay` stage/bind 展开 → C++；ObjC 只 `set*Bytes` / `draw*` 端口 — `mgl_batch_replay.*`（dynamic VAO / UBO·texture override / resource binding collect / attrib can-bind）
- [x] **O2.4** ICB：batch 与 `supportIndirectCommandBuffers` 门闩同层配置 — `mgl_batch_icb_config` / `mgl_batch_icb_support_indirect_command_buffers`；ObjC Batch/Blit + `mgl_air_loader` 同用；`test-batch-icb`；legacy ENABLE_ICB_BATCH|PIPELINES / DISABLE_ICB(_BATCH) 仍识别
- [x] **O2.5 / A3** 验收（**字面口径**）：`MGLRenderer+Batch*.m` 合计 &lt; 600 — **~292**（271+21）。flush/restore/check/stream → `mgl_batch_flush_restore_encode.m`；dyn-bind/sampler/simple → `mgl_batch_dyn_bind_encode.m`；FBO/index/sampler POD + `test-batch-restore`/`test-batch-issue`。MC env 金样 / benchmark 仍建议补跑
  - **Metric arbitrage / Track B**：同域 ObjC **未清完**。诚实 Batch ObjC 簇（categories + `mgl_batch_*_encode.m` + `mgl_batch_replay_trace.m` + `mgl_batch_rt_mark_port.m`）= **~1711**（encode/trace 仍 ~1.5k）。**勿宣称 ObjC cleanup done**。度量：`scripts/objc_renderer_loc.sh`（B 已改簇定义）

### Batch O3 — RenderPass / PSO / Binding【P1】

- [ ] **O3.1** load/store / clear / attachment match → `mgl_render_pass_plan.*`
  - [x] **clear-value 首刀**（O3.1 启动 + 回归保护就位）：`mglRenderPassPlanClearValues` 沉入 plan 层（`mgl_render_pass_plan.c`），配套新头 `mgl_render_pass_clear.h`（**必须**与 `mgl_render_pass_plan.h` 分开——后者被 `glm_context.h:96` 引入，若再 include `mgl_render.h` 会成环）；`mglRenderPassAttachmentClass` / `mglRenderPassColorAttachmentIndexValid` 从 `mgl_render.cpp` 迁入 plan 层（纯值谓词，C linkage，顺带压薄 monolith），使 plan 层自包含、**harness 可独立链接不拖 Metal/LLVM**；`+RenderPass.m` 的 `mglRenderPassClearValuesFor` 变薄转发（只取 persistent state 再转发）
  - [x] **load/store + attachment match 首刀（`d52f334`）**：`configureUserFBOLoadStoreActionsLocked:` 的三段
    动作决策（颜色 / 深度 / 模板）与"未附着却挂着 clear 位"的清理规则移入 plan：
    `mglRenderPassPlanLoadStore()`（输入＝kind / 是否附着 / 是否有待清位 / 是否有 texture / dontcare 开关 /
    本帧首次使用 / 是否开混合；输出＝load action + 是否改写 store + store action）与
    `mglRenderPassDropsStaleColorClear()`；`shouldUseDontCareLoadForColorTexture:firstUseThisFrame:` 整个谓词删除
    （其判据成为 plan 输入）。同时把**默认帧缓冲**那一半的附着匹配移入
    `mglRenderPassAttachmentsMatch()`：输入是**逐附着条目**（identity + 每条目的纹理对 / required 缺失即不匹配 /
    可选 subresource 三元组比对），配套 `mglRenderPassFillMatchEntry()` 填充口。**两半都接了**——默认帧缓冲
    （3 条目：color0 / depth / stencil，深度与模板的 required 由 caps 决定）与用户 FBO（每个 draw slot 一条目 +
    深度/模板条目，逐条目调用以保持原短路顺序）；`mglRenderPassSnapshotAttachmentMatchesSubresource` 因此失效删除
    （subresource 规则进 plan，由 harness 固化）；subresource/scan-stop 中 scan-stop 本就是 C 谓词。
    golden＝新增 `make test-render-pass-load-store`（**53 例**，纯 C：颜色 8 组决策、深度/模板 5 组、
    stale clear 4 组、匹配含 subresource 与填充口 14 组，含"未参与的条目不比较"、"required 缺失即不匹配"、
    "两侧无纹理时不比 subresource"）；golden 是**先照 ObjC 现有语义写死**再改实现（文档要求的"先补 golden"）。
    规模（如实）：`+RenderPass.m` 7058 → **7125（+67）**，`MGLRenderer*.m` 34525 → 34617。
    这一段 LOC 反增的原因是：该半段的实质是**解析**（texture / subresource 取值）而非决策，解析必须留在 ObjC，
    规则合并后调用点反而更长；要真正压 LOC 需把 texture 解析做成回调注入（仿 O5.1 的 resolver），列入下一刀。
    验证：本地 92/0/2；`test-render-pass-load-store` 44/44；`test-render-pass-clear-plan`、
    `test-binding-stage`、`test-buffer-plan`、`test-reference-query` 30/30、`test-per-vertex-signature` 23/23、
    `test-legacy-compat` 193/193、`test-frontends`、`test-tess-domain`、`test-tess-air`(180)；
    hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空；tess 139/1/0；GS 136/0；
    refq 223 例 164/54/5 且逐例 diff 为空；piq 30 例 17/12/1 且逐例 diff 为空。
  - [x] **clear-value 回归保护 harness**：`make test-render-pass-clear-plan`（`test_legacy_compat/test_render_pass_clear_plan.c`，已挂 `test-all`）；golden 覆盖 color[0]/color[3]/depth/stencil 取值、非法 color index(8/100)、未知 attachment kind、NULL state、NULL out params；**已做变异测试验证能捕获回归**（故意交换 blue/alpha → harness exit 2、2 failures）
  - [ ] **残量**：① 用户 FBO 那一半的 attachment match（`mglRenderPassMatchesFramebufferImpl:` 的循环与 subresource 比对、`mglRenderStopColorAttachmentScan` 交织；plan 侧 `mglRenderPassAttachmentsMatch()` 已就绪待接）；
    ② attachment 物化（`configureUserFBOAttachmentsLocked` / `configureDefaultFramebufferAttachmentsLocked` 的 texture 解析与 MS 平面）；
    ③ `mglRenderPassActionsFor` 的持久状态读写口（现已全为薄转发）。
    前两项仍需先补 golden 再动手（勿无 oracle 下刀）。
- [ ] **O3.2** `generatePipelineDescriptorState` → format-class PSO builder（CTS Batch 4）
  - [x] **C1 本刀**：topology / tess modes / format-class / blend·stencil·cull / scissor·viewport → `mgl_pso_format_class.*`（render ~20165→~19558）；`+Binding.m`/`+RenderPass.m` 未增厚；残量 generatePipeline apply + BindingState
- [ ] **O3.3** `+BindingState` / `+Binding` 合并下沉 slot 表；ObjC 绑定口 &lt; 300 LOC
  - [x] **C1 本刀**：slot/sampler/stage/plain-uniform policy → `mgl_binding_policy.*`（render ~20470→~20165）；`+Binding.m` 未增厚；残量 BindingState apply（仍厚）
  - [x] **C1 续刀**：stage UBO/SSBO/UC/atomic bind plan + helpers → `mgl_binding_stage.*`；V/F map 环 → plan@C + 薄 set*Buffer 口；fallback 资源类型表；`test-binding-stage`；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；残量 attrib/texture/image BindingState 仍厚（口远未 &lt;300）
  - [x] **C1 本刀**：attrib plan + helpers → `mgl_binding_stage.*`；sampled/storage/image-view plans + helpers → `mgl_binding_texture.*`；V sampled GATE/COMPAT/FINAL + shared RT copy helper；storage V/F 合并 stage 环；`test-binding-texture`；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~4523→~4302；残量 depth-recover / 厚 logging / 口仍 ≫300
  - [x] **C1 本刀**：InSampler / depth-RT recover plan → `mgl_binding_texture.*` + `mgl_binding_texture_log.m`；ObjC = plan@C + thin bind/apply + log ports；`test-binding-texture` 扩 depth-recover；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~4302→~4240；残量 Y-flip/sampler materialize + 口仍 ≫300
  - [x] **C1 本刀**：Y-flip RT ports + sampler materialize + sampled-diag gates → `mgl_binding_texture.*`；TBIND/sample-detail/RT log ports 扩既有 `mgl_binding_texture_log.m`（不新裂 log 壳）；ObjC = plan@C + thin set*/createMTLSampler；`test-binding-texture` 扩 sampler/RT/diag；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~4240→~4152；残量口仍 ≫300
  - [x] **C1 本刀**：port collapse / apply-masks — sampler warmup + V/F snapshot emit + sampled diag ports@C；`texture_log.m` 合并 depth/RT/fallback（**shrink** 345→~327，禁再涨）；诚实度量 BindingState+texture_log；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~4152→~3855；残量口仍 ≫300
  - [x] **C1 本刀**：stage fallback unify / residual snap — V/F fallback → shared stage 环 + `PlanFallbackSlot`/`InlineBytesSrc`/`BuildPresentMask`；VATTR/VFB/VPS/FFB 残量宏塌到 `MGL_BIND_SNAP_*`；texture_log **hold** 327；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~3855→~3648 (−207)；残量口仍 ≫300
  - [x] **C1 本刀**：stage map-entry unify / present-mask finalize — V/F map 环 → shared `bindStageBufferMapEntriesForStage` + `ClampMapCount`/`PostMtlUsable`；present-mask/sparse/diag → `finalizeStageBufferPresentMask`；resource ordinal helper；texture_log **hold** 327；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~3648→~3476 (−172)；残量口仍 ≫300
  - [x] **C1 本刀**：sampled V/F stage unify / FINAL ports — shared `bindSampledTexturesForStage` + `FillSampledFinalInput`/`ForceDefaultSampler`；SNAP/EMIT macros → headers；dead CreateBuffer/RequiredBytes/LevelView 删；texture_log **hold** 327；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~3476→~3313 (−163)；残量口仍 ≫300
  - [x] **C1 本刀**：stage-fill / emit ports / attrib SELECT — FillMapEntry/Attrib/GATE/COMPAT @C；Set*/Queue/Flush + STAGE emit → Draw_Private；attrib emit helper；inline V/F fallback wrappers；texture_log **hold** 327；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~3313→~3094 (−219)；残量口仍 ≫300
  - [x] **C1 本刀**：depth-RT wire / diag emit / stage POST+storage fills — FillDepthRecover*/SampledRT/Sampler/Storage/DiagEmit @C；EmitSampledDiagPorts + WriteFragTrace + EmitMipDiag；FillMapEntryPostMtl；**禁**再堆 Draw_Private；texture_log **hold** 327；`+Binding.m` 未增厚；**未**灌进 `mgl_render.cpp`；BindingState ~3094→~2937 (−157)；残量口仍 ≫300
- [ ] **O3.4** `MGLPipelineCache.m` → C++ LRU cache（兼 FPS 掉帧 P0）
- [ ] **O3.5** 验收：`+RenderPass.m` &lt; 800 LOC；PSO miss 行为有非 CTS 单测

### Batch O4 — Texture / Blit / Readback【P1】（对齐 CTS ReadbackPolicy）

- [x] **O4.1** Y-flip / MSAA resolve policy / integer·depth pack → `mgl_readback_policy.*`（CTS Batch 2）；**C1 已落** IntegerReadback + CopyRows/depth pack/GetTexImagePlan/MSAA stride；**残量** Metal `EncodeMultisampleResolve*` + flip-aware format convert 仍在 monolith
- [x] **O4.0** value-geometry 类型（MGLSizeValue/MGLOriginValue/MGLRegionValue）+ 构造器去重 → `mgl_region_value.{h,cpp}`（C++-safe，纯 C 无 Metal/ObjC）；原 `MGLRenderer_Private.h`/`+Texture_Private.h` 重复 typedef 与 `+Texture.m`/`+Blit.m` 重复 static 簇（`mglTexture*`/`mglBlit*`）删除，改 `static inline` 别名转发 `mglRegionOrigin/Size/1D/2D/3D`；**unblock** O3/O4 plan 层（须纯 C++ 构造 `MGLRegionValue` 而不引 ObjC 头）
- [ ] **O4.2** upload dirty / 3D / array / texel buffer plan → C++；ObjC 只 `replaceRegion` / blit
- [ ] **O4.3** fallback sampled texture 选择 → format/type class 表，禁止散落 `if`
- [ ] **O4.4** `+Blit` 剩余 format/DS unify（延续 sink）→ `mgl_blit_plan.*`
  - [x] **首切片：depth/stencil 三道门（`1359764`）**：新建纯 C `mgl_blit_plan.{h,c}`——
    `mglBlitPlanDepthStencil()` 判定 `blitFramebufferDepthStencil:` 的三条路径（① MSAA resolve：格式相同 /
    读多重采样 / 写单采样 / 两侧 level 与 depth plane 为 0 / 原点 0 / 不缩放 / 落在两张纹理内，**slice 允许不同**；
    ② 同尺寸拷贝：格式相同 / 双单采样 / 不缩放 / 源尺寸为正，并把 **GL scissor 裁剪后的目标矩形**与随之平移的
    源原点一起算好（含落在两侧纹理内的校验）；③ 缩放渲染：双单采样 / 缩放 / GL_NEAREST / 两张都是 2D 且
    level·slice·depth plane 全 0），以及 resolve 的 depth/stencil 两个 arm（stencil 需 packed DS 源）；
    配套 `mglBlitFillDS{Texture,Subresource,Rect,Mask}Input()` 分组填充口（仿既有 plan 模块的 Fill\*Input 风格，
    texture 那个负责清零、须先调）。ObjC 侧改为**算一次纹理信息**（原先该函数内重复调用 `mglBlitTextureInfo`
    ~20 次）＋ 4 次分组填充 ＋ 读 `dsPlan.*` 分派，三个门条件与裁剪/边界推导整段删除。
    golden 先行：`make test-blit-plan`（`test_legacy_compat/test_blit_plan.c`，**58 例**：同尺寸拷贝含 scissor
    裁剪与越界拒绝、MSAA resolve 的 7 个边界（slice 允许不同/level·plane·原点·缩放·越界各自阻断、packed 才有
    stencil arm、unpacked stencil 无 arm）、缩放路径的 5 个阻断条件、填充口与 NULL 容错），已挂 `make test-all`。
    规模：`+Blit.m` 4945 → **4941（−4）**，`MGLRenderer*.m` 34608 → 34610（plan 输入填充的开销与门的缩减基本抵消；
    **决策面已移出**，进一步压 LOC 需把 blit 物化也做成回调/vtable，列入续刀）。
    验证：本地 92/0/2；`test-blit-plan` 58/58；`make test-all` 返回 0（含 smoke / es-smoke / legacy-compat 193/193）；
    CTS hotspot 非通过集合 diff 为空；tess/GS/refq/piq 见下。
- [ ] **O4.5** 验收：`+Texture.m`+`+Blit.m` &lt; 1.5k；readback 金样不依赖 CTS oracle

### Batch O5 — Buffer / Compute / VertexLayout / 杂项【P2】

- [ ] **O5.1** Buffer map / CoW / UBO isolate → C++；cap CoW（兼 FPS 方案）
  - [x] **O5.1 本刀**：`getVertexBufferIndexWithAttributeSet:`（纯决策：VAO 解析 + `vertex_buffer_map_list` fallback）下沉为 C 函数 `mglRenderVertexBufferIndexForAttribute(ctx, state, attribute, where)`（`MGLRenderer.m` 兄弟函数，声明于 `MGLRenderer+Draw_Private.h:84`）；ObjC 方法仅 `MGL_STATE(ctx)` 解 `GLMState *` + 一行转发。C1 模式范本，零 Metal 耦合；`+Buffer.m` 该入口 −33 行；唯一调用方 `+BindingState.m:356` 不变。
  - [x] **O5.1 续刀3（本刀）**：删掉 `mapShaderBufferResourcesToBufferMap:stage:` 里 **544 行 reflection fallback**（4 类资源逐 draw 重算 metal/client binding、plain-uniform struct packing、全局 fallback），plan 成为唯一映射路径。**先测量后下刀**：临时探针挂在 fallback 入口（reason/stage/program/四类资源计数），本地全量 `test_regression`（92 项）与 GL46 hotspot（1328 例）**一次都没触发**。删除后可用性显式化：无 program → 无需映射；plan/stage plan invalid → 强制一次 `mglBufferBindingPlanBuild` 重试（`EnsureBuilt` 只在 vertex stage invalid 时重建）；仍 invalid 即分配失败 → 日志 + **拒绝该 draw**（调用方保持 dirty 并重试，瞬态失败下一帧自愈）——这是唯一的行为变化（旧行为是 OOM 时静默走慢路径）。`+Buffer.m` 1342→826（本 session 内 1479→826），`MGLRenderer*.m` 35102→34586。
  - [x] **O5.1 续刀2（本刀）**：`mapVertexAttributeBuffersToBufferMap:vao:stageInputCount:stage:` 整段（207 行）沉为纯 C `mglRenderPlanVertexAttribBuffers`（`mgl_vertex_attrib_plan.{h,c}`）：候选遍历、同流分组（buffer name/target + stride/divisor）、slot 分配（`kMGLVertexAttribBufferBase` 起）、容量/索引溢出守卫、mismatch 诊断全部在 C；逐 attribute 的 GL 状态解析留在 ObjC，经 `MGLVertexAttribResolveFn` 回调注入（`+Buffer.m` 传 `mglResolveVertexAttribForPlan`）。配套把 `MGLResolvedVertexAttribBinding` + resolver 原型从 ObjC 私头 `MGLRenderer+Draw_Private.h` 提到 C 头 `mgl_vertex_attrib_binding.h`，并把纯值谓词 `mglRenderMappedBufferCountOK` 从 `mgl_render.cpp` 迁入 plan TU（沿用 O3.1 的"纯值谓词迁 plan 层、harness 不拖 Metal/LLVM"口径）。`+Buffer.m` 1479→1342（−137），`MGLRenderer*.m` 35239→35102。新增 `make test-buffer-plan`（`test_legacy_compat/test_buffer_plan.c`，11 组 golden：单 attribute/同流合并/stride 分裂/divisor 分裂/多 buffer/不可解析跳过/空候选/既有条目保留/容量守卫/mismatch 只告警/混合候选，已挂 `test-all`；变异测试：删掉 stride 兼容判据、slot 不自增两次都被 harness 捕获）。
  - [x] **O5.1 续刀**：`checkForDirtyBufferData:` / `updateDirtyBaseBufferList:`（纯 `BufferMapList` 迭代 + dirty 上传决策，零 Metal）下沉为 `mglRenderCheckForDirtyBufferData` / `mglRenderUpdateDirtyBaseBufferList`（`MGLRenderer.m` 兄弟函数，`MGLRenderer+Draw_Private.h` 声明）；ObjC 方法仅一行转发。原 `[self updateDirtyBuffer:]` 调用改为直调 `mglRenderUpdateDirtyBuffer`，逻辑等价；所有调用方（`+RenderPass`/`+Compute`/`+Binding`/`mgl_batch_dyn_bind_encode`）经 ObjC 薄壳不变。`+Buffer.m` 两入口合计 −75 行。`test-regression` [01]–[15] 全 PASS（[16] `air_geometry_resources` 仍是已知上游 SIGSEGV，与本改动无关）。
- [x] **O5.2 compute buffer 绑定环复用 stage-binding plan（`ad7f18e`）**：
  `bindBuffersToComputeEncoder:…:executionPlan:temporaries:` 的逐条目决策不再由 ObjC 自己拼（旧形态是
  `mglRenderResolveMappedBufferSlot` + `mglTessPlanIsolatedBinding` + 手写 "Metal backing 太小就重建" 三套并存），
  改为与图形路径**同一个 plan**：PRE 阶段 `mglBindingStageFillMapEntryInput` → `mglBindingStagePlanMapEntry`
  （NEED_MTL）→ 物化 → `mglBindingStageFillMapEntryPostMtl` → POST 阶段（ISOLATE / BIND），ObjC 只剩物化与快照编码。
  plan 输入新增三个**默认关闭**的开关（零初始化即旧行为，harness 已固化）：`no_inline`（compute 没有 set*Bytes
  路径 ⇒ plain-uniform 槽必须落成真 Metal buffer）、`iso_storage_exhausted` + `storage_remaining`（GL 存储耗尽也要
  隔离，compute/tess 口径）、`iso_empty_visible`（可见 backing 为空也要隔离）；另加 C 侧
  `mglBindingStageMapEntryDisposition()`（失败/跳过/继续的判决表）、`mglBindingStagePlanReasonName()`、
  `mglBindingStageIsolateFallbackLength()`（隔离副本长度下限 `max(required,4)`），把判决表与下限从 ObjC 移出。
  等价性 oracle＝**决策级 A/B 探针**（物化后同点复算旧公式与 plan 判决，逐条目比较 slot / 隔离判决 / copy-back
  判决 / fallback 长度）：本地 94 项 21 次决策、CTS refq+tess+gs+hotspot 537 次决策，**0 分歧**。逐点核对：
  slot 解析与 `mglRenderResolveMappedBufferSlot` 同算法；required 字节以 `visible_range=0 && min_stage_bytes=0`
  复现旧 `mglTessRequiredBindingBytes` 的结果；`allow_isolate_when_gpu=1` 复现"compute 作为写入者也要隔离"；
  copy-back 判定与旧 `mglTessIsolatedNeedsCopyBack` 一致。
  **CTS 抓到一处探针没覆盖的差异（如实记录）**：plan 在 ISOLATE 时把 `bind_offset` 置 0（隔离副本总绑 0），
  而 copy-back 的**目标**偏移必须取调用方 map 的 offset；第一版误用 `plan.bind_offset`，
  `KHR-GL46.geometry_shader.api.max_shader_storage_blocks`（GS 写 16 个 SSBO、binding 带非零 offset）立刻报
  "Value read from Shader Storage Buffer [9] … is not equal to expected value"（GS 簇 135/1）。修正后恢复；
  头文件写明该字段语义、harness 增断言 `plan.bind_offset == 0`、探针补上 copy-back 判决这一维。
  规模：`+Compute.m` 1251 → 1276（+25：PRE/POST 两阶段与显式失败语义写在 ObjC 侧，决策面已移出），
  顺带删掉该文件既有的未使用 helper `mglComputeCreateTextureLevelView`（编译期 warning 消失）。
  **续刀（纹理侧，`ef37534`）**：`bindTexturesToComputeEncoder:` 的**采样器级联**原本手写了两份
  （主循环 + 采样纹理数组元素），与图形侧 `materializeSampledSamplerForTexture:` 第三份互不相同。
  现在三处收敛到同一实现：compute 直接调用图形侧那个 port（`MGLRenderer+Binding_Private.h` 公开声明），
  其内部仍走 `mglBindingTexturePlanSamplerMaterialize` 决策表；配套新增纯 C 日志标签
  `mglBindingTextureSamplerStageTag()`（"vert"→VERT / "comp"→COMP / 其它→FRAG）。
  **一处有意的行为变更**：数组元素路径过去**漏了 dirty 采样器的释放**（会继续用旧 Metal 对象），
  走共享 port 后与其它路径一致；oracle＝**新造的 compute 专项语料**（152 例：`compute_shader` 42 /
  `shader_image_load_store` 51 / `-cs` SSBO 59）做旧实现 A/B，**逐例 diff 为空**（两边同为 113 pass / 38 fail /
  1 ns）——即该一致性修正在这批语料上不可观测，按"有意变更 + harness 固化 plan 判决"记录。
  golden 先行：`test-binding-texture` 补 compute 形状的 materialize 用例（GL sampler 无 Metal ⇒ recreate + 清 dirty；
  已物化且干净 ⇒ 复用；tex params 为空 ⇒ 仍选 tex params；顶点形状 ⇒ KEEP）与标签 helper 用例。
  `+Compute.m` 1276 → 1267（−9），`MGLRenderer*.m` 34617 → 34608。**剩余**：纹理单元的解析（`textureForSampledResource:` /
  `textureUnitForSampledResource:`）是共享解析函数、决策已是 C 谓词，不再重复下沉；
  把整段纹理循环迁 C++ 需要物化回调 vtable，仍留作 O5.2 续刀。
- [ ] **O5.3** `+VertexLayout` 删除或 &lt; 100 LOC
  - [x] **O5.3 本刀**：`generateVertexDescriptorState:` 整段 plan 装配（读 GL VAO/Program + 写 `MGLRenderPipelineDescriptorState`，零 Metal/`id`）沉为 C 函数 `mglRenderGenerateVertexDescriptorState(ctx, state, nativeTESActive, nativeTESProgram, tcsOutputStride, absoluteVertexBindingOffsets, where)`（`MGLRenderer.m` 兄弟函数，声明于 `MGLRenderer+VertexLayout_Private.h`）；ObjC 方法仅提取 `_tessellation`/`_batching` 两 ivar 标量 + 一行转发。逻辑逐行等价（`NSLog`→`fprintf(stderr,...)`）。`+VertexLayout.m` 332→~193。`test-frontends` 67/67；`test-regression` [01]–[15] PASS（[16] 已知上游 SIGSEGV 不变）。**剩余**：`updateBlendStateCache`（写 `_pipelineCache` ObjC 物化，留）、`bindFramebufferAttachmentTextures`（FBO 绑定，归 RenderPass/O6）。
- [ ] **O5.4** `mgl_draw_encode.m` 迁空或删除
  - [x] **O5.4 本刀**：纯 C 的 indirect-skip 谓词 `mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled` / `mglSkipIndirectDrawWhenPolygonPointEmulationNeeded`（零 Metal、零 ObjC，仅被 `mgl_draw_issue.cpp` 调用）从 `mgl_draw_encode.m`（1225→1186，−39）迁至 `mgl_draw_issue.cpp`；声明仍留 `mgl_draw_encode.h` 不变，`mgl_draw_issue.cpp` 加 `#include "mgl_draw_mode.h"`；`MGLRenderer.m` 旧注释同步修正。仍含 `mglEncode*ForRenderEncoderOwner` 等带 `MGLDrawMetalHandle`/`__bridge` 的薄端口，需将 handle 改 `void*` 并在调用方 bridge 后才能整文件迁 C++。
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

### Batch O7 — SPIRV→LLVM IR 兼容层清理【P0，2026-09-12 新开】

**口径**：SPIRV/MSL 文本时代，内建、资源类型与 sampler 归属都不在反射里，ObjC/C 只能靠
「扫源码文本 / 猜名字 / 合成 location」作答。IR 链（`mgl_air_reflect` + `fillStageInfo`）现在对这些
问题有**精确事实**，因此本批任务是：把事实发布出来，删掉猜测。**禁则**：不得用「CTS 没跑到」当作
删除依据——每一刀必须先建立 oracle（探针计数 / 逐条 diff / golden），再删。

- [x] **O7.1 内建使用掩码（`a6cc8ce`）**：`MGLAIRStageInfo.builtin_mask` +
  `Program.air_builtin_mask[stage]` + `mglProgramStageBuiltinMask()` / `mglProgramStageUsesBuiltin()`；
  退役 6 处 `strstr(shader->src, "gl_X")`（GS `gl_PointSize`/`gl_PrimitiveID`/`gl_ClipDistance`、
  FS+GS `gl_Layer`/`gl_ViewportIndex`、TES `gl_ClipDistance`）；`mglRenderVSWritesLayer()` 改收
  `Program`。顺带删除死代码 `mglRendererFindMSLEntryParameterClose()`（生成 MSL 入口参数表解析器，
  AIR 链不再生成 MSL 文本，全树零调用）。
- [x] **O7.2 vertex-id / frag-coord / sample（`64d127d`）**：掩码扩到 VERTEX_ID / FRAG_COORD /
  NUM_SAMPLES / SAMPLE_ID / SAMPLE_POSITION / SAMPLE_MASK / INTERPOLATE_AT_SAMPLE /
  INTERPOLATE_AT_OFFSET / SAMPLE_INTERPOLATION；frontend 谓词补「调用名匹配」（内建函数）与
  `mglFrontendStageUsesSampleInterpolation()`（`sample in`）。退役 `program.c` 的
  vertex/primitive-id 扫描与 FS frag-coord/sample 扫描；删除 `mglRenderShaderSourceUsesSampleParams()`；
  `mglRenderFragmentNeedsPerSampleMSValues()` 改收 `Program`。
- [x] **O7.3 sampler 名字/位置启发式（`84fe4c4`）**：删掉 `_UNIFORM_CONSTANT_RES` 分支上的
  `uniform_location >= 0x4000` 与「名字含 `Sampler` / 等于 `CloudFaces`」两条子句，三份重复实现合并为
  `mglRenderResourceLooksSamplerLike(res_type, image_dim)`；oracle＝探针在 CTS 1328 例 9602 次判定 +
  本地全量 721 次判定中**零次**由这两条子句决定。
- [x] **O7.4.1 反向 sampler 查表去掉 plain-uniform 遍（`591aa11`）**：
  `mglFindSamplerResourceForMetalBinding`（`mgl_sampler_compat.m`）原按 5 类资源找 slot 归属，
  第一遍 `_UNIFORM_CONSTANT_RES` 是 SPIRV 时代兼容（当年 sampler 可能落在 plain-uniform 列表里）。
  IR 链下该列表只放 plain-uniform 聚合（`binding` 是 **buffer** 槽、`image_dim == 0`，opaque leaves 已
  抽到 `_SAMPLED_IMAGE_RES`），那一遍只可能按数值巧合命中 texture 槽——正是 sampler-like 过滤在掩盖的隐患。
  oracle：探针记录每次调用（结果 / 四表精确查找 / 命中资源数）→ 本地全量 **12 次调用全部
  compatHit=0、diverged=0、matches=1**；**CTS 语料完全不触达该函数**（1328 例 hotspot 探针文件为空），
  因此这里以本地套件为 oracle、CTS 只作回归护栏（文档如实标注）。
- [x] **O7.4.2 属性位置的声明序兜底删除（`de69b78`）**：`mglRendererProgramUsesVertexAttrib` 与
  `mglRendererProgramVertexAttribResource`（`mgl_vertex_attrib_query.m`）末尾各有
  `location == 0xffffffff && i == attribute` 的兜底（无 location 时按声明序猜）。oracle：探针记录所有
  `location == UINT32_MAX` 的 stage input 及兜底是否命中 → 本地全量 94 项 + CTS hotspot 1328 例
  **零观察**；结构上 `assignStageVarSymLocations` 保证每个 VS stage input 都有 location（explicit →
  `glBindAttribLocation` 名字表 → 声明序），`applyVertexInputLocations` 只覆盖被绑定名字。两处一起删。
- [x] **O7.4.3 三个"跳过该 stage 资源"谓词全部退休（`5372d0c`）**：
  `mglShouldSkipStageTextureResource` / `mglShouldSkipStageSamplerResource` 早已是恒 `false` 死桩，
  连调用点一起删（sampled-texture / storage-image plan 现传 `skip=0`；binding-state、draw_tess、compute、
  batch-replay 的 `continue` 守卫消失）；最后一条真规则 `mglShouldSkipStageBufferResource`
  （`_UNIFORM_CONSTANT_RES` 且 sampler-like）先用探针测量：本地全量 94 项 + CTS hotspot 1328 例
  **零次 TRUE**，与 sampler 启发式那刀的结论一致（plain-uniform 资源在 IR 链下不带 texture dim），
  随后删除并简化 5 处调用点。`MGL_BP_FLAG_SKIP` 因此失去生产者，标志位、`mglRenderBufferPlanEntrySkip()`
  与 buffer-map fast path 的守卫一并退休（plan 层仍保留 skip action 供 harness 使用）。
- [x] **O7.4.4 引用查询从源码文本扫描迁到 TU 访问路径（`f089cd9`）**：
  `mgl_gl_extensions.c` 的三个文本扫描器——`mgl_program_stage_source_body`（`strstr(src, "void main")`
  截断取"主体"）、`mgl_program_source_name_is_referenced`（叶子标识符整词匹配）、
  `mgl_program_qualified_member_referenced`（`<instance>[...].<member>` 文本匹配）加转发包装
  `mgl_program_stage_source_references`，四个函数共 123 行——删除，
  改由 `mgl_frontend_session.c` 的 `mglFrontendStageReferencesName()` / `mglFrontendStageReferencesMember()`
  回答（`mgl_program_stage_references_name/member` 只做 `Program` → `Shader::frontend_tu` 转发）。
  语义（= 新的**规格**，由 `make test-reference-query` 固化）：
  1. 只走**函数体**：声明、注释、预处理行都不算引用；反过来 main 之前定义的 helper 算引用
     （旧扫描在第一个 `void main` 处截断，位于其上的 helper 一律看不见）；
  2. 表达式压成组件路径 `name[idx][idx]...`（VAR_REF/MEMBER 追加名字，INDEX 给最后一个组件追加下标；
     非字面量下标记 `?`），query（`colors[0]`、`a[0].b[0].d[0]`）按同一形状解析，两边的组件必须**对齐成连续一段**；
  3. 于是：选得更深算引用（query `colors[0]` vs 访问 `colors[0].xyz`）、query 少写尾部下标算引用
     （`colors` vs `colors[0]`）、两个**字面量**下标必须相等（`colors[1]` 不匹配 `colors[0]`）、`?` 是通配；
  4. 被索引对象的"丢下标前缀"（`blk.colors` 之于 `blk.colors[0]`）不参与匹配，否则同族元素会互相冒充
     （这一条是 harness 逼出来的：第一版匹配器把 MEMBER 子表达式路径也算引用，`colors[1]` 会命中 `colors[0]`）；
  5. 旧扫描按**叶子**文本找（`a[0].b[0].d[0]` 只找 `d[0]`），所以同名叶子会假阳性——这是本次唯一有意的语义收紧。
  oracle：① 新增 `make test-reference-query`（30 例，纯 C，不需要 Metal/ObjC：`mglGLSLParse` + sema 后直接查 TU）；
  ② A/B 双库对照：223 例 `KHR-GL46.{program_interface_query, shader_atomic_counters,
  shader_atomic_counter_ops_tests, shader_storage_buffer_object, layout_binding}.*`，切换前（`79269da` 的三个文件）
  与切换后各跑一次，**逐例结果 diff 为空**（两边都是 164 pass / 54 fail / 5 crash）；③ 标准三套语料见下。
  注：这三套语料几乎不触达该路径（caselist 内 `atomic|program_resource|interface_query|shader_storage` 命中
  tess 1 / GS 4 / hotspot 0 例），所以真 oracle 是 ①②，CTS 只作回归护栏——文档如实标注。
- [x] **O7.4 残条：源码文本判定的逐条判定（`7383fc2`）**：把全树剩余的 `->src` / 源码文本判定逐条过了一遍
  （`grep -rn "->src"` + `grep -rn "strstr("` 全命中 + 调用者追踪），按"可删 / 解析器本体需要"分类。
  **本刀删除（每条都有 oracle）**：
  1. `mgl_uniform_reflection.c` 的 `mglSamplerUniformLocationFromReflection()` +
     `mglFindExplicitUniformLocation()`（共 102 行）：前者**全仓无调用者**（`mgl_air_reflect.c` 直接用
     `s->location`，拿不到才用 `mglSyntheticSamplerUniformLocation()`），后者是它内部"从名字位置向前找
     `layout(...)` 再 `strtoul` 解析 `location=N`"的源码扫描器。oracle＝静态事实：`grep` 全仓（含 ObjC/C++/
     harness/头文件）零引用，且无 harness 桩引用它。
  2. `mgl_gl_extensions.c` 的 `_UNIFORM_CONSTANT_RES` 无 TU 兜底（按本 stage 资源表里同名 `query_name`
     猜"被引用"）。oracle＝探针 `MGL_O74B_LOG`：本地 94 项 + CTS piq 30 例 + refq 223 例共 **32 次命中，
     全部 `no_tu=1` 且 `result=0`**，而无 TU 时 TU 查询本身也返回 0 ⇒ 该兜底从未改变过答案。
  3. `mgl_gl_extensions.c` 的 `.d[0]` 数组长度特判（`query_name` 以 `[0]` 结尾且含 `.d[0]` ⇒ 返回 2）。
     oracle＝同一探针：命中 2 次，`query=TrickyBlock.a[2].b[0].d[0]`，**`reflected_size` 本身已经是 2**
     ⇒ 返回值与删除后的 `res->ubo_member->size` 完全相同（零行为差异）。
  4. `mgl_gl_extensions.c` 两处残留 `->src` 门（block 成员引用查询、plain-uniform 引用查询）：查询已改走 TU，
     门只剩"有没有 src"的旧含义；删掉后由 TU 判空回答（stage 缺失/未编译 ⇒ 0）。
  5. `mgl_program_reflection.c` 的 `mglProgramPerVertexSignature()` + `mglShaderSourceHasToken()`：
     `strstr(src,"gl_PerVertex")` → 花括号配对 → 整词找成员。改为读 TU 的块声明
     （`decl->type->name == "gl_PerVertex"`，成员取 `struct_members`；实例名无关）。oracle＝新增
     `make test-per-vertex-signature`（23 例，纯 C，构造 Program/Shader + `mglGLSLParse`）；A/B＝同 harness
     编到旧实现上 **21/23**，差异恰好是旧扫描的两处假阳性（注释里的块、`struct gl_PerVertexLike`）。
  **保留并标注（不属兼容残留）**：`mgl_frontend_session.c` 的 `strstr(src,"#version")`
  （`mglFrontendGLSLVersionOf`：legacy 重写需要版本号，发生在解析**之前**，此处无 TU 可用）与 legacy 翻译
  幂等标记注释（同样是重写阶段的唯一通道）；`program.c` 的 `mglShaderInterfaceCheck` /
  `mglShaderTessInterfaceCheck` 与 `mglAirReflectGLSLStageInfo(shader->src,…)`（三者都是**自己 parse+sema**
  再用 IR 比较/填充，不是文本判定——链接期重复解析属 O5 类去重候选，另计）；
  `program.c`/`shaders.c` 把 `->src` 当编译输入与空值判断；`mgl_air_backend.cpp` 的
  `strstr(name,"Proj"/"Lod"/"Grad"/"Offset")`（GLSL 内建名→AIR 行为分派；子串匹配偏松，属代码质量项）。
  **留待下一批**：`mgl_uniform_reflection.c` / `mgl_gl_extensions.c` 的"名字→类型/location"启发式
  （`Color`/`UV`/`Normal`/`Position`…）与 `mglDefaultAttribLocationForName()`——属 O7.3 名字启发式家族，
  需先探针测 `gl_type == 0` / 默认 location 的命中率。
- [ ] **O7.4 剩余候选（按收益排序）**：
  1. 名字启发式一批（见上条末段）；2. 链接期重复 parse 去重（`mglShaderInterfaceCheck` 复用
     `frontend_tu`，属 O5 类）。
- [ ] **O7.5 验收口径**：每刀必须给 `local 全量（94 项）` + `CTS tess 140 / GS 136 / hotspot 1328`
  三套数字，hotspot 要求**非通过集合逐条 diff 为空**；退役的判据要在文档里留下 oracle 说明（探针名 +
  样本量 + 结论），否则不得删。

---

## 3. 与并行工作流的衔接

| 并行线 | 关系 |
|--------|------|
| 当前 `refactor(*): sink …` | **同向燃料**；本 TODO 要求 sink **落地到域文件**，不要只进 `mgl_render.cpp` |
| CTS Batch 1 TessDomain | **已基本落地**（`mgl_tess_domain*`）；O1 清 ObjC 宿主残留 |
| CTS Batch 2 ReadbackPolicy | **驱动 O4** |
| CTS Batch 3 LimitsTable | 不进 ObjC；C 状态机 |
| CTS Batch 4 PSO format-class | **驱动 O3.2 / O3.4** |
| SPIRV→IR 兼容清理（O7） | 与 CTS Batch 2/3/4 同向：凡是「CTS 时代为绕开反射缺口而猜」的写法，先建 oracle 再删；本周期已清 10 处文本判定 + 3 份 sampler 启发式 |
| MC FPS：hazard / PSO LRU / Y-flip / CoW | 分别挂在 O2.2 / O3.4 / O4.1 / O5.1 |

建议 PR 节奏：**O0 docs → O1 draw host → O2 batch →（CTS readback ∥ O4）→ O3 PSO/bind → O5 → O7 兼容清理 → O6 删文件**。

---

## 4. 每 PR 检查清单（短）

1. 改动的逻辑属于 Keep 清单吗？否则必须有新的/已有 C ABI。
2. 有无 Metal 金样或 property 测试吗？（禁止「只靠 CTS Fail 数」）
3. ObjC 文件 LOC 是否下降（或持平但策略行减少）？
4. 有无 lease 范围内使用 Metal 对象吗？
5. 有无 `TECH_DEBT(objc-thick)` / CTS 禁则冲突吗？

---

## 5. 落地日志与即时下一刀

> 按时间顺序连续编号（2026-09-12 整理：修掉重复编号并把新增刀按序归位）。每条末尾给出该刀的
> **下一刀候选**与**禁则**；最新的可执行清单见文末第 29 条与 §2 的 Batch O7。

1. ~~**O1.1–O1.6**~~：DrawSupport &lt;400；XFB/TES HostOps 已下沉
2. ~~**O1.4 residual**~~：`mglDrawGsExecuteMetalExpansion` + `mgl_draw_gs_metal.cpp`
3. ~~**O1.4 residual2**~~：capture/validate/cull → C++；HostOps → metal_port；StageHost ~360（bindCull/MS + 一行包装）
4. ~~**O2.1**~~：`mgl_batch_select_path` + `test-batch-path` 已合入
5. ~~**O0**~~：本文与 `scripts/objc_renderer_loc.sh` 已挂进 `docs/` / ARCH / README
6. ~~**O2.2 / O2.3**~~：`mgl_batch_hazard` + `mgl_batch_replay` stage/bind
7. ~~**O2.4**~~：`mgl_batch_icb_*` 统一 batch path + `supportIndirectCommandBuffers`；`test-batch-icb`
8. ~~**A1**~~：删 `mglDrawHostGsExecuteMetalExpansion`；`metal_ops` 嵌套；StageHost ~363 保留
9. ~~**A3 先前刀**~~：`mgl_batch_restore` + stream/dyn-vertex/sampler/hash；diag 拆出；3529→2849；`test-batch-restore`
10. ~~**A3 中刀**~~：`mgl_batch_rt_mark` / `mgl_batch_issue`；ICB/MDI/direct encode；Batch 簇 →~1636
11. ~~**A3 / O2.5 本刀**~~：flush/restore/check/stream → `mgl_batch_flush_restore_encode.m`；dyn-bind/sampler/simple → `mgl_batch_dyn_bind_encode.m`；FBO fold + stream-index + stable-sampler POD；Batch*.m **1636→~310**（&lt;600 字面）；`test-batch-restore`/`test-batch-issue`
12. ~~**B（metrics）**~~：`scripts/objc_renderer_loc.sh` Batch 簇改为含 `mgl_batch_*_encode.m` + `replay_trace` + `rt_mark_port`；诚实合计 **~3352**。O2.5 字面达标 ≠ ObjC cleanup done
13. ~~**C0**~~：`docs/C0_AIR_RENDER_DEP_MAP.md` — 仅 `mgl_air_backend.cpp` / `mgl_render.cpp` 的 includes/callers/domains（mermaid）；**无 monolith 编辑**
14. ~~**A3 encode-fold 本刀**~~：plan-in-C + `mgl_batch_mtl_encode` 一行 MTL 口；flush path stats / restore dirty / select fill / scratch / ICB array params / dyn offset·sampler slot / RT cross·diag；诚实簇 **3352→~3087**。禁扩 metal_port / replay_trace / `mgl_render.cpp`。**勿宣称 cleanup done**
15. ~~**A3 encode-fold 续（flush/dyn 重心）**~~：`mgl_batch_mtl_encode_buffer_binds` / `resource_binds`；restore dirty/skip/oracle/finish + flush path-perf/cmd-stats；dyn mtl-ptr/slot/sampled gates；trace FS-slot POD（**shrink** replay_trace）；诚实簇 **3087→~2910**。禁扩 metal_port / trace / `mgl_render.cpp`。**勿宣称 cleanup done**
16. ~~**A3 encode-fold 本刀（issue/ICB whole loops）**~~：`mgl_batch_mtl_issue_mdi_batch` / `stream_mdi_batch` / `icb_batch` / `simple_replay`；`mgl_batch_issue_direct_batch` + `apply_dyn_bindings`（C，ObjC 未全接线）；issue 514→~315、icb_mdi 347→~180；诚实簇 **2910→~2509**（−401）。禁扩 metal_port / trace / `mgl_render.cpp`。**勿宣称 cleanup done**
17. ~~**A3 encode-fold 续（dyn/flush/trace）**~~：接线 `mgl_batch_issue_apply_dyn_bindings`；`mgl_batch_mtl_bind_dyn_vertex/uniforms/sampled`；`mgl_batch_flush_run_batches` + `mgl_batch_check_should_execute`；stream_merged 接线；trace/RT POD formatters（**shrink** replay_trace 456→~374）。诚实簇 **2509→~2366**（−143）。禁扩 metal_port / `mgl_render.cpp`。**勿宣称 cleanup done**
18. ~~**A3 encode-fold 本刀（dyn/flush 重心 → ≤2k）**~~：`mgl_batch_mtl_apply_sampler_snapshot`；`mgl_batch_flush_trace_skip_commands` / `mgl_batch_bind_active_textures`；`mgl_batch_rt_run_draw_attachments` + `copy_state_to_rt`；`mgl_batch_restore_apply_from_key`；dyn/flush/issue/rt/Batch 残体压薄；修 `rt_mark_port` 断体。诚实簇 **2366→~1923**（−443）。禁扩 metal_port / trace / `mgl_render.cpp`。`test-batch-issue`/`test-batch-restore` 扩展。**勿宣称 cleanup done**
19. ~~**C1**~~：`mgl_readback_policy.{h,c}`（O4.1/CTS）— IntegerReadback Convert/Source/Packed/Classify 出 `mgl_render.cpp`（~21043→~20599）；`mgl_render.h` include 域头；禁堆回 monolith
19b. ~~**C1 O4.1 residual**~~：Y-flip (`CopyRows`) / depth pack (`CopyDepth*`/`DepthReadbackPlan`/`RepackDepthPlanes`) / `GetTexImagePlan` / `MSAAArrayLayerStride` → 同 TU（render ~20599→~20470）；Metal MSAA encode 残量留 monolith；禁堆回
19c. ~~**C1 binding-policy (O3.3)**~~：slot/sampler/stage/plain-uniform → `mgl_binding_policy.{h,c}`（render ~20470→~20165，−305）；禁增厚 `+Binding.m`；BindingState apply 残量留 monolith（仍厚）
19d. ~~**C1 format-class PSO (O3.2)**~~：NeedsExplicitTopology / PrimitiveTopologyClass / format-class / Blend·Stencil·Cull / scissor·viewport → `mgl_pso_format_class.{h,c}`（render ~20165→~19558，−607）；禁增厚 `+Binding.m`/`+RenderPass.m`；generatePipeline apply 残量留 monolith
20. ~~**A3 encode-fold 本刀（submit/trace/Binding）**~~：`mgl_batch_issue_submit_direct_*` 决策树；trace fill/gate helpers（**shrink** replay_trace 374→~254）；binding dedup/sync → `+Binding.m`；sampled resolve gate；restore/teardown/sync C drivers + tests。诚实簇 **1923→~1711**（−212）。禁扩 metal_port / trace / `mgl_render.cpp`。**勿宣称 cleanup done**
21. **下一刀**：残量 `dyn`(~364) / `flush`(~379) / `trace`(~254) / `issue`(~215)；C1 已开但勿掩盖 Batch 残量；**C1b–C1g done**（见上）；Batch 诚实簇仍~1711、`mgl_render`~19.6k 勿宣称 done；BindingState 仍厚；继续 ensure/resolve/submit 与 flush callback 压薄；O3 / O6 可并行
    - **禁止**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`

22. **O5.1 本刀**：`getVertexBufferIndexWithAttributeSet:` → C 函数 `mglRenderVertexBufferIndexForAttribute(ctx,state,attribute,where)`（`MGLRenderer.m` 兄弟函数，`MGLRenderer+Draw_Private.h:84` 声明）；ObjC 仅 `MGL_STATE(ctx)` + 一行转发；零 Metal 耦合，C1 模式范本；`+Buffer.m` −33 行、唯一调用方 `+BindingState.m:356` 不变
    - **O5.1 续刀（已落）**：`checkForDirtyBufferData:` / `updateDirtyBaseBufferList:` → C 函数 `mglRenderCheckForDirtyBufferData` / `mglRenderUpdateDirtyBaseBufferList`（`MGLRenderer+Draw_Private.h` 声明）；ObjC 两入口合计 −75 行。`test-regression` [01]–[15] 全 PASS。
    - **下一刀候选**（O5）：`+Buffer.m` 的 map/CoW/shadow plan、`+Compute.m` dispatch plan（O5.2）、`mgl_draw_encode.m`（O5.4）迁空；**O3.1（`+RenderPass` load/store/clear）风险高，留待带 clear-value 回归保护时再做**
    - **禁止**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`

23. **O5.4 本刀（已落）**：`mgl_draw_encode.m` 的两枚纯 C indirect-skip 谓词迁至 `mgl_draw_issue.cpp`（O5.4 第一步：`mgl_draw_encode.m` −39 行、零 Metal/ObjC 逻辑移出 ObjC TU）。`test-regression` [01]–[15] 仍全 PASS；[16] `air_geometry_resources` 已知上游 SIGSEGV 不变。
    - **下一刀候选**（O5）：`mgl_draw_encode.m` 余下 `mglEncode*ForRenderEncoderOwner` 需把 `MGLDrawMetalHandle` 改 `void*` + 调用方 `__bridge` 后方可整文件迁 C++；或转 O5.2（`+Compute.m` dispatch plan）。
    - **禁止**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`

24. **上游 SIGSEGV 已修（6f53544，2026-09-09）**：[16/93] `air_geometry_resources` 的 `EXC_BAD_ACCESS`（`objc_retain` ← AGX `setBuffer_impl` ← `mglDrawGsExecuteMetalExpansion`）根因是 **GS compute-binding 计划的 temporaries 生命周期**：`mglGsMetalFillComputeBindings` 用局部 `NSMutableArray` 保活 plan 里的借用 MTL 指针，ARC 在函数返回时释放数组，而 C++ 侧在其后才 encode/dispatch。修复：`fill_compute_bindings` 新增 `void **temporaries_out` 出参，以 +1 CF 引用交还 keep-alive 集，C++ 侧用既有 `OwnedList` 追踪（作用域覆盖 expansion + XFB scatter 两次 execute）。compute-dispatch 与 tessellation 路径 temps/execute 同作用域，无同类问题。
    - **回归基线变化**：`test-regression` 首次跑完 93 项：**PASS 80 / FAIL 11 / SKIP 2**。11 个失败为此前被 [16] 崩溃掩盖的既有失败：10× `air_tessellation_*`（accumulation、isolines_point_mode/variants/indexed/multidraw/rasterdiscard/tripoint_instanced/xfb、factors_spacing、cull_distance，均 rc=1）+ `legacy_glsl_frontend`（rc=31）。**这些不在本次修复引入范围内，需独立排查**（嫌疑：tess-domain 重构 5d81d98 行为变化）。
      - **2026-09-12 更新**：上述 10× `air_tessellation_*` 已由后续 tessellation 修复（`b6adea0` 三项 ABI 风险 + `06ab844` PSO 键）清掉，`legacy_glsl_frontend` 亦不在当前失败列表；本地全量基线现为 **94 项：92 PASS / 0 FAIL / 2 SKIP**。
    - **另**：`test-mglair` `TCS_FACTOR_FAIL (patch=0 factor=5 bits=0x0000)` 经 stash 基线法确认为既有失败（与 GS temporaries 修复无关）。
    - **诊断教训**：旧 "+Compute.m temporaries" 定位是陈旧结论（早期文本检索转义差异导致静默漏配、误判"符号已不存在"）；崩溃定位应优先取系统崩溃报告里的真实调用栈，并用**泄漏对照实验**（临时延长可疑对象生命周期看崩溃是否消失）确认根因后再改码。


25. **O5.1 续刀2（本刀，vertex-attrib buffer map 沉 C）**：`mapVertexAttributeBuffersToBufferMap:` → `mglRenderPlanVertexAttribBuffers`（新 TU `mgl_vertex_attrib_plan.c`，故意与 `mgl_buffer_plan.c` 分开——后者的 plan-build 依赖 shader/program resource 层，会拖 ObjC/Foundation 进 harness）。seam：`MGLVertexAttribResolveFn` 回调 + `MGLResolvedVertexAttribBinding` 类型提到 C 头。`MGLResolvedVertexAttribBinding` 的原私头声明改为 include 转发。`+Buffer.m` −137；`make test-buffer-plan` 11 组 golden + 变异测试；`test-frontends`/`test-regression` 子集/CTS 复验见提交信息。
    - **下一刀候选**：同域剩余两大环 `mapShaderBufferResourcesViaPlan:`（fast path，保留）与 `mapShaderBufferResourcesToBufferMap:stage:`（~541 行 reflection fallback），可用同一 seam 思路（plan@C + 回调解析），但需要先补 plan/fast-path 一致性的 golden；或转 O5.2（`+Compute.m` dispatch/binding 环）与 O4.4（`+Blit` format/DS unify）。
    - **禁止**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`

26. **O5.1 续刀3（reflection fallback 已删）**：证据链＝探针（`MGL_PLANFB_TRACE`，测后已删）在 92 项本地全量 + 1328 例 GL46 hotspot 上零命中；此后 plan 缺失只可能是分配失败，改为"重试一次 + 拒绝该 draw + 速率限制日志"。**注意语义变化**：OOM 下旧代码会静默走慢路径完成绑定，现在会拒画（调用方 dirty 重试）；若日后要恢复 OOM 韧性，应该让 plan 分配不可失败（例如 Program 内联存储），而不是把慢路径搬回来。
    - **下一刀候选**：`+Compute.m`（1255）的 compute binding 环仍厚 —— 可复用 `mgl_binding_stage`/`mgl_binding_texture` 的 plan 形态；或 `+Blit.m`（4945）O4.4 format/DS unify；或 `+BindingState.m`（2940）残量。
    - **禁止**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`

27. **SPIRV 时代兼容写法清理（对象：ObjC 里的源码文本判定）**：SPIRV/MSL 工具链时代，内建与资源类型都不进反射列表，ObjC 只能 `strstr(shader->src, "gl_X")`。LLVM IR 链下 frontend 本来就知道每个内建（要生成对应 store/load），所以改为**发布精确事实**，分两刀落地：
    - **#1（`a6cc8ce`）**：`MGLAIRStageInfo.builtin_mask`（POINT_SIZE / PRIMITIVE_ID / CLIP_DISTANCE / CULL_DISTANCE / LAYER / VIEWPORT_INDEX / TESS_LEVEL）+ `Program.air_builtin_mask[stage]` + 访问器 `mglProgramStageBuiltinMask()` / `mglProgramStageUsesBuiltin()`；退役 `+RenderPass.m` 的 GS `gl_PointSize`/`gl_PrimitiveID`/`gl_ClipDistance`、FS+GS `gl_Layer`/`gl_ViewportIndex`、TES `gl_ClipDistance` 共 6 处扫描；`mglRenderVSWritesLayer()` 由 `const char *src` 改为读 `Program`。顺带删掉死代码 `mglRendererFindMSLEntryParameterClose()`（解析生成 MSL 的入口参数表，AIR 链不再生成 MSL 文本，全树无调用）。
    - **#2（`64d127d`）**：掩码扩到 VERTEX_ID / FRAG_COORD / NUM_SAMPLES / SAMPLE_ID / SAMPLE_POSITION / SAMPLE_MASK / INTERPOLATE_AT_SAMPLE / INTERPOLATE_AT_OFFSET / SAMPLE_INTERPOLATION；frontend 谓词补两项能力——`mglFrontendNoteBuiltinUse` 现在也匹配**调用名**（内建函数 interpolateAt*），新增 `mglFrontendStageUsesSampleInterpolation()`（`sample in` 限定符，走 IR symbol / TU decl 的 `MGL_AST_Q_SAMPLE`）。退役 `program.c` 的 vertex-id/primitive-id 扫描与 FS frag-coord/sample 扫描，删除 `mglRenderShaderSourceUsesSampleParams()`，`mglRenderFragmentNeedsPerSampleMSValues()` 改为接收 `Program`（注意两套 bit 集合刻意不同：`uses_sample_params` 含 NUM_SAMPLES、`needs_per_sample_ms` 含 INTERPOLATE_AT_OFFSET，与旧扫描一致）。
    - **语义严格更准**：`strstr` 连注释、字符串字面量、`gl_LayerFoo` 这类更长标识符都会命中；掩码只认真正的内建符号。
    - **验证（两刀同口径）**：本地全量 92 PASS / 0 FAIL / 2 SKIP；`test-legacy-compat` 193/193；`test-frontends` 67/67；`make test-buffer-plan`；GS 簇 136/0；tess 簇 139/1/0（同一 FO-spacing 归档例）；hotspot 1270/52/4 ns/1 crash，**非通过集合与上一轮逐条 diff 为空**。

    **同族待清理清单（已定位，按收益排序）**：
    | 位置 | SPIRV 时代写法 | IR 链下的替代 |
    |---|---|---|
    | ~~`mgl_binding_policy.c:236` `mglRenderResourceLooksSamplerLike`~~ | **#3 已删（`84fe4c4`）**：`uniform_location >= 0x4000` + 名字启发式（`Sampler`/`CloudFaces`） | 实测 9602 次判定（CTS 1328 例）+ 721 次（本地全量）**从未由这两条子句决定**；合成 location 只发给 `_SAMPLED_IMAGE_RES`/`_STORAGE_IMAGE_RES`，sampler 声明与 plain struct 的 opaque leaves 也都进 `_SAMPLED_IMAGE_RES`（带 `image_dim`）。三份重复实现合并为单一 `mglRenderResourceLooksSamplerLike(res_type, image_dim)` |
    | `mgl_sampler_compat.m:163` `mglFindSamplerResourceForMetalBinding` | 按 metalBinding 遍历 5 类资源找 sampler（判定已改走共享谓词） | 仍可进一步改成用反射的 `binding` 直查（本刀未动其遍历结构） |
    | `mgl_vertex_attrib_query.m:56` `mglRendererProgramUsesVertexAttrib` | `location == 0xffffffff && i == attribute` 的"无 location 时按声明序"兜底 | IR 链每个 stage input 都有 location；该分支只剩兼容意义 |
    | `mgl_program_resource.c` `mglShouldSkipStageTextureResource` / `mglShouldSkipStageSamplerResource` | 恒返回 `false` 的桩（说明当年的跳过启发式已死） | 直接删桩与调用点判断 |
    | `mgl_gl_extensions.c:1428` | `strstr(shader->src, "void main")` | 查询/校验路径，需单独评估 |
    | `mgl_legacy_compat.c` 多处 | 旧 GLSL 文本改写 | **设计如此**（兼容层本体），不在清理范围 |
    | `mgl_frontend_session.c:42/60` | `#version` 解析 / legacy 翻译标记 | 解析器本体需要，保留 |

28. **SPIRV 时代兼容写法清理 #3（sampler 名字/位置启发式，`84fe4c4`）**：sampler-like 判定原有**三份重复实现**（`mgl_binding_policy.c` / `uniforms.c` / `mgl_uniform_reflection.c`），规则是"类型是 sampler/image，或 `_UNIFORM_CONSTANT_RES` 且（`image_dim != 0` ∥ `uniform_location >= 0x4000` ∥ 名字含 `Sampler` ∥ 名字 == `CloudFaces`）"。后两条是 SPIRV 时代兼容：旧反射可能把 sampler 放在 plain-uniform 列表里且类型不可靠，于是用"合成了 uniform location"和 Minecraft 专属名字当证据。IR 链下两条都不可能命中（合成 location 只发给 sampled/storage image；sampler 声明与 plain struct 的 opaque leaves 都进 `_SAMPLED_IMAGE_RES`）。
    - **先测后删**：临时探针在两个 live 副本上记录每次 `_UNIFORM_CONSTANT_RES` 判定（子句 + 结论）——CTS hotspot 1328 例 **9602 次判定、本地全量 721 次**，**全部 `clause=none decision=0`**，无一次由 synthetic-location 或名字子句决定。
    - 删除后可合并为单一共享谓词 `mglRenderResourceLooksSamplerLike(res_type, image_dim)`；`uniforms.c` 查询路径与 uniform-reflection 的 sampler 匹配改调它。顺带删掉 `mglRenderSamplerNameLooksSamplerLike()`、两份 `mglUniformNameLooksSamplerLike()`、`mglRendererSamplerNameLooksSamplerLike()`、未使用的 `MGL_SYNTHETIC_SAMPLER_LOCATION_BASE` 定义。
    - 验证：本地全量 92/0/2；`test-legacy-compat` 193/193；`test-frontends` 67/67；GS 簇 136/0；tess 簇 139/1/0；hotspot 1270/52/4 ns/1 crash，非通过集合与上一轮 diff 为空。
    - **下一刀候选**：`mglFindSamplerResourceForMetalBinding` 的 5 类资源遍历 → 反射 `binding` 直查；`mglRendererProgramUsesVertexAttrib` 的无 location 兜底；`mglShouldSkipStageTexture/SamplerResource` 恒 false 死桩；`mgl_gl_extensions.c:1428` 的 `"void main"` 扫描。

29. **O7.4.1 反向 sampler 查表清理（`591aa11`）**：`mglFindSamplerResourceForMetalBinding` 去掉
    `_UNIFORM_CONSTANT_RES` 那一遍（SPIRV 时代兼容：当年 sampler 可能落在 plain-uniform 列表），并去掉循环内
    已冗余的 sampler-like 过滤（类型列表本身即分类）。保留"slot 落在 `[binding, binding+元素数)` 即归属该资源"
    的语义，与 `mgl_air_reflect.c` 发放 slot 的方式一致。
    - **oracle（如实标注覆盖缺口）**：临时探针记录每次调用的结果 / 四表精确查找 / 命中资源数 —— 本地全量
      **12 次调用，全部 `compatHit=0 diverged=0 matches=1`**；**CTS hotspot 1328 例完全不触达该函数**
      （探针文件为空），所以本刀以本地套件为 oracle、CTS 只作回归护栏。结构上那一遍只能"数值巧合"命中
      texture 槽（plain-uniform 的 `binding` 属 buffer 槽、`image_dim==0`），删掉同时消除了该隐患。
    - **验证**：本地全量 92/0/2；`test-legacy-compat` 193/193；`test-frontends` 67/67；
      `make test-buffer-plan`；GS 簇 136/0；tess 簇 139/1/0；hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空。

30. **O7.4.2 属性位置声明序兜底删除（`de69b78`）**：`mgl_vertex_attrib_query.m` 两处
    `location == 0xffffffff && i == attribute` 兜底一并删除（`mglRendererProgramUsesVertexAttrib` /
    `mglRendererProgramVertexAttribResource`）。oracle：探针记录所有 `location == UINT32_MAX` 的 stage input
    与兜底命中情况 → 本地全量 94 项 + CTS hotspot 1328 例**零观察**；结构上 `assignStageVarSymLocations`
    对每个 VS stage input 都发 location（explicit → 名字表 → 声明序），`applyVertexInputLocations` 只覆盖
    绑定名字，故该分支不可达。验证：本地 92/0/2；`test-legacy-compat` 193/193；`test-frontends` 67/67；
    `make test-buffer-plan`；GS 簇 136/0；tess 簇 139/1/0；hotspot 1270/52/4 ns/1 crash 且非共识集合 diff 为空。

31. **O7.4.3 三个 stage-resource skip 谓词退休（`5372d0c`）**：两个恒 `false` 死桩
    （`mglShouldSkipStageTextureResource` / `mglShouldSkipStageSamplerResource`）连调用点删除；
    `mglShouldSkipStageBufferResource` 先探针测量（本地 94 项 + CTS hotspot 1328 例零次 TRUE，与
    "plain-uniform 资源在 IR 链下不带 texture dim"一致）再删，5 处调用点简化。`MGL_BP_FLAG_SKIP`
    失去生产者 → 标志位 + `mglRenderBufferPlanEntrySkip()` + buffer-map fast path 守卫一并退休。
    验证：本地 92/0/2；`test-legacy-compat` 193/193；`test-frontends` 67/67；`make test-buffer-plan`；
    GS 簇 136/0；tess 簇 139/1/0；hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空。

32. **O7.4.4 引用查询迁到 TU 访问路径（`f089cd9`）**：`mgl_gl_extensions.c` 的三个文本扫描器加转发包装
    （`mgl_program_stage_source_body` / `mgl_program_source_name_is_referenced` /
    `mgl_program_qualified_member_referenced` / `mgl_program_stage_source_references`，共 123 行）删除，
    改由 `mglFrontendStageReferencesName()` / `mglFrontendStageReferencesMember()` 走 TU 函数体 +
    组件化访问路径（详见 Batch O7 的 O7.4.4 条）。
    顺带修掉旧扫描的两个结构性问题：main 之前的 helper 看不见（截断），以及按**叶子**文本匹配导致同名叶子假阳性。
    新增持久 oracle `make test-reference-query`（30 例，纯 C；`make test-all` 已纳入）。
    验证：本地 92/0/2；`test-reference-query` 30/30；`test-legacy-compat` 193/193；`test-frontends`；
    `test-buffer-plan` / `test-tess-domain` / `test-tess-air`(180) / `test-binding-*` / `test-render-pass-clear-plan`；
    A/B 双库 223 例逐例 diff 为空（164/54/5 两边相同）；GS 簇 136/0；tess 簇 139/1/0；
    hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空。

33. **O7.4 残条：源码文本判定逐条判定 + 五处删除（`7383fc2`）**：全树 `->src` / `strstr` 命中逐条过筛
    （Batch O7 有分类表），本刀删五处、换一处：
    ① `mglSamplerUniformLocationFromReflection()`（**全仓无调用者**）+ 其内部 86 行"向前找 `layout(...)` 再
    `strtoul` 取 `location=N`"的源码扫描器 `mglFindExplicitUniformLocation()`，共 102 行；
    ② `_UNIFORM_CONSTANT_RES` 的"无 TU 时按同名成员猜被引用"兜底（探针 32 次命中全 `result=0`，且无 TU 时新
    查询同值）；
    ③ `.d[0]` 数组长度特判（探针 2 次命中，`reflected_size` 本身已是 2 ⇒ 零行为差异）；
    ④ 两处残留 `->src` 门（查询已走 TU，门只剩旧含义）；
    ⑤ `mglProgramPerVertexSignature()` 的 `strstr(src,"gl_PerVertex")`+花括号配对+整词找成员，换成读 TU 块声明
    （新增 `make test-per-vertex-signature` 23 例；同一 harness 编到旧实现是 **21/23**，差异恰好是旧扫描的两处
    假阳性：注释里的块、`struct gl_PerVertexLike`）。
    保留并标注的：frontend 的 `#version` 读取与 legacy 翻译幂等标记（都发生在解析**之前**，此处没有 TU 可用）、
    `mglShaderInterfaceCheck`/`mglShaderTessInterfaceCheck`/`mglAirReflectGLSLStageInfo`（自己 parse+sema，
    不是文本判定）、`->src` 作编译输入；名字启发式一批（`Color`/`UV`/`Normal`…、`mglDefaultAttribLocationForName`）
    留待下一批。
    验证：本地 92/0/2；`test-reference-query` 30/30、`test-per-vertex-signature` 23/23、`test-legacy-compat`
    193/193、`test-frontends`、`test-buffer-plan`/`test-tess-domain`/`test-tess-air`(180)/`test-binding-*`/
    `test-render-pass-clear-plan`；piq 30 例 17/12/1 且逐例 diff 为空；refq 223 例 164/54/5 且逐例 diff 为空；
    pp 语料（可分程序管线 5 例）新旧库一致 1/3/1 ns；GS 簇 136/0；tess 簇 139/1/0；
    hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空。

34. **O5.2 compute buffer 绑定环复用 stage-binding plan（`ad7f18e`）**：compute 的逐条目绑定决策
    （slot / required / 隔离 / 偏移 / copy-back / 判决）并入图形路径同一个 `mglBindingStagePlanMapEntry`，
    plan 输入加三个默认关闭的开关（`no_inline` / `iso_storage_exhausted` + `storage_remaining` /
    `iso_empty_visible`）与三个 C 侧 helper（判决表 / reason 名 / 隔离长度下限）。
    决策级 A/B 探针（旧公式 vs plan 判决，含 copy-back 维度）：本地 21 次 + CTS 537 次决策 **0 分歧**；
    但 **CTS 仍抓到一处探针漏掉的差异**——ISOLATE 的 `plan.bind_offset` 是 0（隔离副本总绑 0），
    copy-back 目标偏移必须取调用方 map 的 offset，第一版误用后者导致
    `KHR-GL46.geometry_shader.api.max_shader_storage_blocks` 数据错（GS 135/1），修正后恢复。
    `+Compute.m` 1251 → 1276（+25，决策面已移出；顺带删掉既有未使用 helper）。
    验证：本地 92/0/2；`test-binding-stage`（新开关 / 判决表 / 隔离偏移断言）、`test-buffer-plan`、
    `test-reference-query` 30/30、`test-per-vertex-signature` 23/23、`test-legacy-compat` 193/193、
    `test-frontends`、`test-tess-domain`、`test-tess-air`(180)、`test-render-pass-clear-plan`；
    refq 223 例 164/54/5 且逐例 diff 为空；piq 30 例 17/12/1 且逐例 diff 为空；tess 簇 139/1/0；
    GS 簇 136/0（含回归单例 `KHR-GL46.geometry_shader.api.max_shader_storage_blocks` 1/0）；
    hotspot 1270/52/4 ns/1 crash 且非通过集合 diff 为空。

35. **O3.1 load/store + attachment match 下沉（`dbfe9d9`）**：三段动作决策（颜色/深度/模板）与
    "未附着却挂着 clear 位"的清理规则进 `mglRenderPassPlanLoadStore()` / `mglRenderPassDropsStaleColorClear()`，
    `shouldUseDontCareLoadForColorTexture:` 谓词删除；附着匹配的两半（默认帧缓冲 + 用户 FBO）统一走
    `mglRenderPassAttachmentsMatch()` 的**逐附着条目**（纹理相等 / required 缺失即不匹配 / 可选 subresource
    三元组比对），配套 `mglRenderPassFillMatchEntry()` 填充口；objc 侧失效谓词
    `mglRenderPassSnapshotAttachmentMatchesSubresource` 删除。golden 先行：`make test-render-pass-load-store`
    53 例（颜色 8 组、深度/模板 5 组、stale clear 4 组、匹配含 subresource 与填充口共 14 组）。
    **LOC 如实记录**：`+RenderPass.m` 7058 → 7125（+67）。原因是这半段的实质是**解析**（texture / subresource
    取值）而非决策，分辨率代码无法下沉，规则合并后调用点反而更长；若要继续压 LOC，方向是把 texture 解析做成
    回调注入（仿 O5.1 的 resolver 形态），另开一刀。
    验证：本地 92/0/2；`test-render-pass-load-store` 53/53 及其余 harness 全绿；hotspot 非通过集合 diff 为空；
    tess 139/1/0；GS 136/0；refq/piq 逐例 diff 为空。

36. **当前下一刀（按推荐顺序）**：
    1. **O5.2 续刀**：`+Compute.m` 的纹理/采样器环（约 470 行，含 "late binding" 启发式）复用
       `mglBindingTexturePlanSampled`；整段迁 C++ 需物化回调 vtable。
    2. **O3.1 续刀**：attachment 物化（texture 解析 / MS 平面）做成回调注入以真正压 LOC；
       `configureUserFBOAttachmentsLocked` / `configureDefaultFramebufferAttachmentsLocked`。
    3. **O7.4 残条（下一批）**：名字→类型/location 启发式（`gl_type == 0` 命中率需探针）；链接期重复 parse 去重
       （`mglShaderInterfaceCheck` 复用 `frontend_tu`，属 O5 类）。
  - **禁则（不变）**：扩 `mgl_draw_metal_port.m`、扩 `mgl_batch_replay_trace.m`、新开厚 category、堆进 `mgl_render.cpp`；不得以「CTS 没跑到」代替 oracle。

37. **smoke 死桩与 stdio/stdlib include 清理（`12a2671`）**：三件事：
    ① **修好 `test_metalcpp_smoke` 的链接**——O7.4 把 `mglProgramStageBuiltinMask` 移到
    `mgl_program_resource.c`（头里是 `static inline mglProgramStageUsesBuiltin` 的转发），该 harness 编译
    `mgl_render.cpp` 却没编这个文件 ⇒ 链接失败。按"补真实现而非补桩"的原则把
    `mgl_program_resource.c` 加进它的编译集（该 TU 零外部依赖，只导出 `mglProgramStageBuiltinMask` /
    `mglShaderStageName`）。
    ② **删 smoke 死桩**：判据＝"引擎里已无任何声明的符号，其桩要删"——桩会把未来重新引入的调用**静默接住**，
    而缺桩会**链接失败**（更强的护栏）。据此删 `mtlBindBuffer` / `mtlFlushBufferRange` / `mtlBindProgram`
    （引擎零引用）与 `mglShouldSkipStageBufferResource`（O7.4.3 已删），并清掉随之失效的观测点；
    保留 `mtlBufferSubData` / `mtlMapUnmapBuffer` / `mtlFlush`（引擎仍在引用 ⇒ 桩有承载）与
    `mglSyntheticSamplerUniformLocation`（`test_mglair*` / `test_mcrepro` 的链接确实需要它——**实证**：删掉后
    链接失败，恢复后通过）。
    ③ **清未使用的 stdio/stdlib include**（10 个文件，逐个以构建验证）：`mgl.h`、`gl_core.c`、
    `mgl_draw_encode.cpp`、`mgl_legacy_compat.c`、`mgl_program_reflection.c`、`mgl_program_resource.c`、
    `mgl_texture_transfer.c`、`test_legacy_compat/main.c`、`test_mcrepro.mm`、`test_mglir.c`。
    影响面实测（`-H` 追 include 链 + 逐头探针）：`mgl.h` 仍经 `glm_context.h` 提供 **stdio**，被切断的只有
    **stdlib**；仓库内没有 TU 依赖这条传递链（`program.c`/`mgl_gl_extensions.c`/`rendering.c` 自带
    `<stdlib.h>`；`state.c` 的"free"只是注释）。唯一靠传递拿 stdlib 的是 `draw_buffers.c`（`calloc`/`getenv`），
    它走 `mgl_trace_log.h` → `objc/objc.h` → `<stdlib.h>` 的 **ObjC 链**——**不是本次切断的**、但属既有脆弱点，
    故按"谁用谁 include"给它补上显式 `#include <stdlib.h>`（`-H` 复核 stdlib 改为深度 1 由自身提供）。
    验证：**`make test-all` 返回 0**（含 smoke 系列：`SMOKE_DONE`、`es-smoke: ok`、legacy-compat 193/193、
    test_regression 92/0/2、各 plan harness 全绿）；CTS hotspot 非通过集合 diff 为空。

38. **目标重定为「ObjC 清零」+ T0/T1 首批（`c5429c2`）**：把本文的目标从"薄平台层（≤8–12k）"改为
    **`MGL/` 内 ObjC 清零**，并给出可复现度量 `scripts/objc_zero.sh`（文件数 / 空 TU / 行数 / ObjC 语法次数 /
    ObjC 词汇次数 / `MGLRenderer*.m` 合计）与基线（53 文件 · 43,989 行 · 语法 2,268 · 词汇 4,353）。
    分层 T0–T5（空 TU → 仅 `#import` → 有词汇无语法 → 决策下沉（沿用 O1–O7）→ 端口 C++ 化 → 平台壳），
    并加严禁则：除 T5 允许的**唯一**平台壳外，任何新代码不得引入 ObjC。
    **首批 T0/T1 已落地**：删掉 3 个只有注释的空 TU（`MGLBindingSync.m` / `MGLQueryManager.m` / `MGLTextures.m`），
    10 个"仅 `#import` 是 ObjC 语法"的文件改名为 `.c`（`hash_table` / `mgl_texture_compat` / `mgl_sampler_compat` /
    `mgl_sync` / `mgl_rt_sync` / `mgl_capability` / `mgl_coordinate` / `mgl_focus_program` / `mgl_shader_resource` /
    `mgl_state_log`），`#import` → `#include`，并去掉 `mgl_shader_resource.c` / `mgl_sampler_compat.c` 里**未被使用**的
    `<Foundation/Foundation.h>`；Makefile 中 `test_metalcpp_smoke` 对 `mgl_sync.m` 的两处显式引用同步改为 `.c`。
    度量：文件 **53 → 40**、空 TU **3 → 0**、ObjC 行数 **43,989 → 41,753**、ObjC 语法 **2,268 → 2,254**。
    验证：两个库构建无错；**`make test-all` 返回 0**（smoke / es-smoke / legacy-compat 193/193 / test_regression
    92/0/2 / 各 plan harness）；CTS 整轮见下（hotspot 非通过集合 diff 为空）。
    下一批：**T2**（10 个"有词汇无语法"文件：`BOOL/YES/NO/nil/NSUInteger/NSLog` → C 等价物后改名）。

39. **T2 落地：10 个"有词汇无语法"文件转 C（`38455bd`）**：`BOOL→bool`（头里 `<objc/objc.h>`→`<stdbool.h>`）、
    `YES/NO→true/false`、`NSUInteger→size_t`、`NSLog(@"…")→fprintf(stderr, "…")`、
    `NSString *label→const char *label`（`mglDumpBytesToLog`，其 `mglTraceLogNSString` 调用改 `mglTraceLog`）、
    去掉 `#import <Foundation/Foundation.h>` 与 `#import`；`mgl_frame_activity` 的 `os_log_t`/`OS_LOG_DEFAULT`
    改由 `<os/log.h>`（本就是 C API）提供；注释里残留的 YES/NO/nil 措辞一并改写（度量把注释计入）。
    涉及文件：`mgl_buffer_query` / `mgl_blit_clip` / `mgl_draw_buffer` / `mgl_vertex_attrib_query` /
    `mgl_vertex_format` / `mgl_trace_strategy` / `mgl_state_compat` / `mgl_frame_activity` /
    `mgl_binding_texture_log` / `mgl_byte_hash`，全部改名 `.c`。
    **解依赖**：`mgl_trace_log.h` 的 `BOOL`→`bool`（`NSString` 声明本就有 `#ifdef __OBJC__` 保护，C TU 可安全包含）；
    Makefile 为 `mgl_binding_texture_log.c` 加同形状 C 规则（`METALCPP_C_SRC/_OBJ`，smoke 目标仍以 `-x none` 链接）。
    **顺带清掉一处潜在传递依赖**：去掉若干头的 Foundation 后，`mgl_gl_extensions.c` 失去间接的 `<stdlib.h>`
    （`calloc`/`free`/`strtoul`），按"谁用谁 include"补显式 include——清零工作正在把这些隐患逐个逼出来。
    度量：文件 **40 → 30**、行数 **41,753 → 40,044**、语法 **2,254 → 2,239**、词汇 **4,353 → 4,266**。
    验证：两个库构建无错；**`make test-all` 返回 0**（smoke / es-smoke / legacy-compat 193/193 /
    test_regression 92/0/2 / 各 plan harness）；CTS 整轮（hotspot 非通过集合 diff 为空）。
    下一批：**T2′**（12 个 <20 处 ObjC 语法的文件：`mgl_trace_log` / `+Buffer` / `+VertexLayout` /
    `mgl_readback` / batch 端口等），随后进入 **T4**（端口 C++ 化）。
