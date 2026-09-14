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
>
> **联合审计（已签，docs-only 吸收）**：[`OBJC_LLVM_JOINT_AUDIT_2026-09-14.md`](OBJC_LLVM_JOINT_AUDIT_2026-09-14.md)
> （coding · DXMT 讲述者 · DannyFeng-bot；审计基线 tip `4ed1d54`＝**23** 文件；交付 2026-09-13，截止由周一改到 **周日 08:00 Asia/Shanghai**）。
> 当前 tip 见文首（`objc_zero` ＝ **21**；`+Batch.m` 与 `mgl_batch_flush_restore_encode.m` 删除后）。BindingState stash@{0} **仍暂停、禁止 pop/入库**。
> **禁「薄平台 ≤8–12k 即终态」**——终态是清零（T5 唯一平台壳除外）；文件数降权，三厚块 LOC + BindingState + metal_port + air LOC + shim wrapper Δ 升权。

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
| **T5** | 平台壳：`NSWindow`/`CAMetalLayer`/drawable/present/主线程同步 | **唯一平台壳 TU**（`MGLPlatformRendererShell` + `+Lifecycle` **合并**；O6.1 双文件歧义结束）。另可二选一：**(a)** ObjC runtime C API（`objc_msgSend`）在 C++ 内实现使 `MGL/` 0 个 `.m`；**(b)** 移交消费方。**禁止**把 AppKit 沉进 `mgl_render.cpp` | 现 Shell **230** + Lifecycle **666** → 合并为一 TU |

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
| `MGLRenderer*.m` total | **34,604** | 0（当前 **32,218**） |
| **shim 端口数 / 行数**（§0.04 记账面） | 43 / 511 | 0（当前 **16 个端口**；实现面集中在唯一壳 TU，560 → 629 行） |

**当前进度（2026-09-14，T0–T2′ + T4 切片 + **P0-1 八十九刀** + trace 清零 后；第 68–119 轮见 §0.24/§0.26–§0.76；
**第 112 轮第四次尝试采样簇仍被 CTS 拦下并回滚；第 113–119 轮（第八十三～八十九刀）从 `+Blit.m` 推进并已落地**）**：
文件 **53 → 6**（**第一个 category 整文件消失**）、空 TU **3 → 0**、行数 **43,989 → 22,079**、
ObjC 语法 **2,268 → 1,278**、词汇 **4,353 → 2,326**；**第 103/104/112 三轮的采样绑定刀均被 CTS 拦下并回滚
（度量与 `520691f` 相同），第 105 轮起改从 `+Blit.m` 推进**；
**shim：43 → 35 个端口 / 唯一壳 TU 2,078 行 / 298 语法**（第 100 刀退役 1 个端口、新增 4 个纹理物化端口，按 §0.04 该刀只算 P0-1 结构收益、不算 T4 端口净减；**第 101/102 两刀各退役 0/1 个端口、0 新增**；第 103/104/105 三刀按 T5 依次把 `MGLPipelineCache`、纹理绑定入口、renderer 生命周期并入壳，端口均不变；第 106 刀把 `MGLRenderPassManager` 类转成 C struct；第 107 刀把 host-ops 的 25 个 `id` 门面改成 `void *`；**第 125 刀 0 退役 0 新增**——它把 `+Tessellation.m` 的绑定规划簇整块搬进 C，用的是既有端口；**第 126 刀净退役 1 个端口**——`mglRendererDispatchTessControlShaderPort` 随其目标方法转 C 一起删除，C 侧改直调；**第 127 刀再净退役 1 个端口**——`mglRendererDispatchAIRTessEvalVertexRenderPort` 同理；**第 128 刀退役 1、新增 1（T4 净减 0，如实记账）**——AIR TES compute 端口退役，但新方法内部仍要调 `+RenderPass.m` 里的 `ensureAIRTessEvalPassthroughFunctionForProgram:`，故补了一个随它退役的端口；**P0-1 第八十三刀（`+Blit.m` 的 `mtlCopyTexSubImageViaTextureBlit:` 转 C，见第 143 条）0 退役 0 新增**——入口全部复用既有端口与 twin；**P0-1 第八十四刀（两个 copyImageSubData 叶子转 C，见第 144 条）0 退役、新增 1**——`synchronizeRenderPassForTextureReadback:` 尚无 C 入口，故新增该端口（T4 如实记 +1），`_capability` 靠 `areas.core->capability` 零结构改动解决；**P0-1 第八十五刀（后置回读叶子 + copyImageSubData dispatcher 一起转 C，见第 145 条）0 退役、新增 1**——唯一新增的是 `endRenderPassIfFramebufferChangedForNonDraw:` 的端口，另外两个桥（`ctx = glm_ctx` 的 `mglPlatformShellSetContext`、`bindMTLTexture` 的 `mglRendererBindMTLTexture`）都是**既有**入口；**P0-1 第八十六刀（blitFramebuffer 附着解析叶子转 C，见第 146 条）0 退役、新增 3**——`_drawable` 是 property 宏（`self.drawable`），没有 core 字段可借，故必须补 `mglNextDrawable` / `mglDrawableTexture` / `mglEnsureLayerDrawableSizeAtLeastWidth` 三个端口；**P0-1 第八十七刀（scaled color blit 叶子转 C，见第 147 条）0 退役 0 新增**——并顺带补齐 12 个 render-encoder twin；**P0-1 第八十八刀（`mtlBlitFramebuffer:` dispatcher 转 C，见第 148 条）0 退役 0 新增**——被调方法全在 C 里，`+Blit.m` 只剩 68 语法；**P0-1 第八十九刀（`mtlCopyTexSubImage:` 转 C，见第 149 条）0 退役、新增 2**——`mtlReadDrawable` 与 `copyTextureUploadWithDedicatedCommandBuffer` 两个桥，端口头首次引入 `mgl_region_value.h`；`MGLRenderer*.m` **34,604 → 24,779**）。
（已建 C 端口面 `mgl_renderer_ports.*` + 单一 ObjC 端口 shim `mgl_renderer_port_shim.m`；
`mgl_readback` / `mgl_batch_rt_mark_port` / `mgl_trace_log` / `mgl_batch_issue_encode` / `mgl_batch_replay_trace` /
`mgl_batch_icb_mdi_encode` / `mgl_batch_dyn_bind_encode` 七个 TU 已转入 C，
并**整文件删除** `MGLRenderer+Batch.m` 与 `mgl_batch_flush_restore_encode.m`（成员按域落 C，ObjC 壳进 shim）——
**Batch 簇的 `mgl_batch_*_encode` 已全部不是 ObjC**；
`mglTraceLogNSString`（429 行 ObjC 面）已彻底移除；**当前 `objc_zero.sh`＝21 个 `.m`**（联合报告基线 tip `4ed1d54` 为 **23**；勿再写「26 文件」）。）

**口径提醒**：上表数字全部由同一脚本在同一天测得，但 **2,223 那一格是端口 shim 建立之前**的数字
（shim 新增 176 行 / 19 语法 / 3 词汇），本行以重测为准：`2,223 + 19(shim) + 3(桥接) − 14(issue_encode 转 C) − 1(shim 收窄) = 2,230`；
replay_trace 一刀：`2,230 − 16(文件转 C) − 2(MGLRenderer.m 去 NSString) − 1(+BindingState.m 去 NSString)
+ 3(shim 端口) + 4(flush_restore 桥接) = 2,218`；
icb_mdi 一刀：`2,218 − 17(文件转 C) + 3(shim 的 @try/@catch 端口 + processBuffer 端口) + 2(flush_restore 桥接) = 2,207`；
dyn_bind 一刀：`2,207 − 22(文件转 C) − 6(退掉三个已无用的 shim 端口) + 30(12 个新 shim 包装 × 桥接+消息发送) ≈ 2,209`；
批次 shell 一刀：`2,209 − 16(+Batch.m 删除) − 4(三处调用点改直调) + 15(shim 的 category + bindMTLTexture 端口) = 2,204`；
flush_restore 一刀：`2,204 − 34(文件转 C) + 14(shim 的 14 个新端口 × 桥接+消息发送) + 6(调用点桥接) = 2,190`。
**按联合报告 §0.04 的 T4 硬规（无 shim 净减＝拒收），dyn_bind 与 flush_restore 两刀都不算 T4 进度**：
它们把被转走的消息发送搬进了 shim（26 → 40 个包装），语法只小幅下降、文件/行数才是收益。
shim 现在是 **40 端口 + 5 方法 / 511 行 / 61 语法**，**净减它才是下一刀的验收条件**（见第 49 条）。
shim 是**唯一**允许新增的 ObjC 面（逐函数一行包装，把 batch/draw 端口集中到一处），它随实现文件逐个转 C
而缩小，终态删除或并入 T5 平台壳。

**T4 硬规（联合报告 §1/§2 P0-2，已签）**：**无 shim 净减 = 拒收**。下一 Batch PR 必须净减 Port wrappers；
「语法持平 / rename-only / 只转扩展名」= 假进度。`flush_restore→C` **alone** 退休 **0** 个 Port（flush_restore-only bucket = 0）——禁止当进度。
优先杀 multi/shared 桶；全 encode/trace 停用 Port → 26/26 退休。

### 0.05 联合分类总表（Delete / Rewrite / Keep-thin / Keep-product-dual / Keep-A/B-temporary）

> 源：[`OBJC_LLVM_JOINT_AUDIT_2026-09-14.md`](OBJC_LLVM_JOINT_AUDIT_2026-09-14.md) §2（三方已签）。审计 tip `4ed1d54`；当前 tip 见文首。

#### Delete（可立即排队，小）

| 项 | 证据 | 动作 |
|---|---|---|
| `shouldUseDontCareLoadForColorTexture:` 死声明 | `_Private.h`；无实现；`MGLRenderer.m` 过期注释 | 删声明+注释 |
| 文档过期计数（「26 文件」等） | `objc_zero.sh`：联合报告时 **23**，当前 tip **21** | **已对齐脚本；禁再写 26** |
| 「薄平台 ≤8–12k 即终态」旧目标句 | 掩护三厚块 | **已禁**；改清零叙事（T5 唯一壳除外） |
| `esrc` 文本侧 ×24（oracle-equal 后） | `mgl_air_backend.cpp` ~10457–10552 | 删 strstr，只留 mask/IR |
| `MGL_USE_METALCPP` 生产 A/B | 树内无生产读取 | **死透 — 禁止复活** |

#### Rewrite（真债）

| Pri | 项 | 要点 |
|---|---|---|
| **P0-0** | air `esrc`→`builtin_mask`/IR | LLVM 特异最高优先；`emitTessBlock*` 外提；**禁再胀** `mgl_air_backend.cpp`（尤其为 tess） |
| **P0-1** | 三厚块 materialize/upload | **禁止整文件 Delete**；抽 C++ 域+金样；禁新增 ObjC 行 |
| **P0-2** | T4 纪律 | 每 Batch PR **净减 shim wrappers**；否则拒收 |
| **P1** | BindingState 二次决策 | 一口 plan（stash `PlanAttribSelect` **仅作形状参考**）；`spirvBinding` 改名；≪300 DoD 仍开 |
| **P1** | `mgl_draw_metal_port` HostOps | 假薄 ~1982；禁扩；HostOps 外迁 |
| **P1** | 名字启发式 | `DefaultAttribLocation*` / `LooksLikeSampledColor2D`：探针 `gl_type==0`/命中率后 Delete 或收表 |
| **P1** | TES ObjC 编排 | 下沉 `mgl_draw_tess`；**路径本身 Keep-product-dual** |
| **P2** | Compute 绑定环 | 等 BindingState 端口定型再共享 apply |

#### Keep-thin

StageHost / Draw 一行 issue / DrawSupport / PipelineCache id 桥 / GPURecovery 触发口。
**T5 终态钉死：** `MGLPlatformRendererShell` + `+Lifecycle` **合并为唯一平台壳 TU**。禁止把 AppKit 沉进 `mgl_render.cpp`。

#### Keep-product-dual（非临时）

**TES compute expansion vs TES-vertex render** — AIR/Metal ABI 分叉；删任一路径 = 产品错误。只允许编排下沉，不允许「选边删除」。

#### Keep-A/B-temporary（必须带退休条件）

| 双轨 | 退休门禁 |
|---|---|
| **ICB / MDI / DIRECT env** | `make test-batch-icb`；`MGL_ENABLE_ICB=1` 下 regression **92/0/2**（现状 **82/10/2**）；日志 **0** 条 `Fragment/Vertex shader cannot be used with indirect command buffer`；tess/GS/hotspot **非通过集 diff 空**；窗口内禁 `MGL AGX RECOVERY/ERROR` / sustained recovery；无 OOM。达标后 env 收成单一 plan 输入。 |
| **C driver + ObjC shim** | 每 PR wrapper 数净减；全 encode/trace 停用 Port → **26/26** 退休 |
| **`esrc` vs mask** | 临时至 Delete 文本侧（P0-0），非开放式 A/B |
| 诊断旗（`MGL_SKIP_SAME_KEY_ORACLE` 等） | Keep with debt note；**不是**行为分叉。`MGL_ENABLE_DONTCARE_LOAD` = plan **输入**，Keep |

#### Do-not-delete-yet

三厚块 / BindingState / TES 双路径 / ICB flags（门禁前）/ shim（短期）/ Compute 环 / **未审 BindingState stash**。

### 0.06 过誉 / 过程债（联合报告 §5 — 不得软化）

1. **plan@C ≠ 域清完**（贴皮）：O3.1 / O4.4 / O3.3 切片把决策碎片沉进 C，但 materialize/upload/二次编排仍在 ObjC；LOC 几乎不动。
2. **T4 shim 会计魔术**：转走的消息发送搬进 shim，语法可持平；**无 shim 净减 = 拒收**。
3. **O7 `esrc` 未阻断**：清了 ObjC/program 源扫描，codegen 文本侧 ×24 未关——审计共谋过誉；现升 **P0-0**。
4. **格式串机械门禁缺失**：`mglTraceLog` 改 `vsnprintf` 后 `mgl_byte_hash` 的 `%@` 漏网（`da51b1c` 才修）。
5. 度量政策：**文件数降权**；三厚块 LOC + BindingState + metal_port + **air LOC** + **shim wrapper Δ** 升权。tip 盯梢通知应附 `shimΔ/airLOC/triadLOC`。

### 0.07 报告后 backlog（联合 §7，与 §2 标签对齐）

| Pri | Item |
|---|---|
| **P0-0** | air `esrc`→mask（oracle-equal 后删 24 strstr）+ `emitTessBlock*` 外提；**停止**为 tess 胀 `mgl_air_backend` |
| **P0-1** | 三厚块 materialize 设计+金样起跑（禁止整文件 Delete） |
| **P0-2** | 下一 Batch 刀：**必须** shim 净减（优先 multi/shared）；flush_restore alone 不够 |
| **P1** | BindingState 一口 plan；名字启发式探针；metal_port 禁扩+HostOps 外迁 |
| chore | DontCare 死声明；Lifecycle→唯一壳；计数对齐 `objc_zero`；tip 通知附 `shimΔ/airLOC/triadLOC` |

**暂停期诚实选项**：只出设计/spike 笔记，或等解除暂停后再动刀；**禁止**再开 rename-only T4。联合建议：先 **P0-0** 设计/spike（LLVM 特异、不碰三厚块巨石），Batch 仅在能证明 wrapper Δ&lt;0 时动。

---

## 0.1 文档地图（2026-09-12 整理）

| 文档 | 入库 | 作用 |
|---|---|---|
| [`OBJC_CATEGORY_DISMANTLE_TODO.md`](OBJC_CATEGORY_DISMANTLE_TODO.md)（本文） | ✅ | ObjC 清零：政策/度量/批次清单 + §0.05 联合分类 + §5 落地日志；最新 backlog 见 §0.07 |
| [`OBJC_LLVM_JOINT_AUDIT_2026-09-14.md`](OBJC_LLVM_JOINT_AUDIT_2026-09-14.md) | ✅ | 已签联合报告（Delete/Rewrite/Keep-* 政策；tip `4ed1d54` 基线） |
| [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) | ✅ | 总架构审查与分层规模；ObjC 厚度数字以本文度量为准；**禁 ≤8–12k 终态句** |
| [`C0_AIR_RENDER_DEP_MAP.md`](C0_AIR_RENDER_DEP_MAP.md) | ✅ | `mgl_air_backend.cpp` / `mgl_render.cpp` 的依赖与调用域地图 |
| `TESS_NATIVE_RENDER_VERTEX_PATH.md`（本地） | ❌ | TES-vertex / compute 双路线设计 + §10.4 的 PSO 键修复记录（2026-09-12）；按文件名引用，不入库 |
| `CTS_TESS_REMAINING_2026-09-10.md`（本地） | ❌ | tess 簇逐轮排查日志 + 单例 ground-truth 复现方法（顶部有当前状态横幅：139/1/0）；按文件名引用，不入库 |
| `GS_XFB_GL4_CTS_PROGRESS_2026-08-21.md`（本地） | ❌ | GS/XFB 簇进度切片 + 批跑命令模板；按文件名引用，不入库 |
| [`AIR_M3_CPP_TODO.md`](AIR_M3_CPP_TODO.md) · [`P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`](P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md) · [`GL46_GS_LAYERED_SPEC_AUDIT.md`](GL46_GS_LAYERED_SPEC_AUDIT.md) | ✅ | 专题记录 |
| 其余 `docs/*.md` | ❌（`.gitignore` 的 `/docs/*`） | 阶段性审计/审查稿与逐轮工作日志，**一律留在本地**；入库文档引用它们时只用文件名（标注"本地"），不随引用一起入库 |

度量脚本：[`scripts/objc_renderer_loc.sh`](../scripts/objc_renderer_loc.sh)（`MGLRenderer*.m` 合计、Batch 诚实簇、
Draw 簇；当前输出 `MGLRenderer*.m total: 34555`）。

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
| **T3/T4 真 ObjC（当前唯一剩余）** | 21 / 38,099 | `+RenderPass`(426 语法/556 词汇) · `+Texture`(316/1103) · `+Blit`(244/880) · `MGLRenderer`(170/280) · `+BindingState`(135/197) · `+Tessellation`(153/292) · `mgl_draw_metal_port`(117/100) · `+Compute`(90/104) · … |
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
| ~~`+Batch.m`~~ | 0 | **已删**（T4）：成员按域落 C（`mglBatchRecord*DrawSubmitted` → `mgl_batch_rt_mark_host.c`；`mglBatchBindActiveTexturesToMTL` → `mgl_batch_replay.cpp`；`mglBatchRestoreStateFromKey` → `mgl_batch_restore_host.c`），dual-proxy / 锁 / 异常壳进 shim |
| `+Tessellation.m` | 2174 | 中→薄 | O1.4：编排在 `mglTessRunPatchDraw`；ObjC 仅 dispatch/物化口 |
| `+BatchReplay.m` | ~21 | 薄占位 | O2.5：dyn-bind → `mgl_batch_dyn_bind_encode.m`；待 O6 删空 category |
| `+Buffer.m` | 826 | 薄化中 | map/CoW/shadow plan → C++（O5.1：vertex-index + dirty-buffer 决策已沉 C 函数；vertex-attrib buffer map 已整段沉 `mgl_vertex_attrib_plan.*` + harness；**reflection fallback 已删——plan 成为唯一映射路径**）；ObjC 只 MTLBuffer 物化与逐 attribute resolve |
| `+Compute.m` | 1267 | 中 | buffer 绑定环复用 `mgl_binding_stage` plan 形态（PRE/POST + 三个 opt-in 开关 + C 侧判决表）；采样器级联与图形侧收敛为同一 port（复用 `mglBindingTexturePlanSamplerMaterialize`）；**剩余**＝整段纹理循环迁 C++（需物化回调 vtable）；ObjC 只 compute encoder 端口 |
| `+Lifecycle.m` | 665 | **Keep-thin → T5 唯一壳** | 与 `MGLPlatformRendererShell` **合并为唯一平台壳 TU**（init/bind/view/lease/dealloc）；O6.1 双文件歧义结束 |
| `+SwapDiagnostics.m` | 555 | Keep/旁路 | 诊断可留 ObjC 或迁 trace；非热路径 |
| `+Draw.m` | 511 | 薄 | O1.5：`mtlDraw*` 一行 → `mglIssue*` / MS guard |
| `+Binding.m` | 488 | 薄化（实测较基线 +80；与 BindingState 合并后删除） | 与 BindingState 合并后删除 |
| `+GPURecovery.m` | 350 | Keep 薄 | 触发 + 日志；reset 在 C++ |
| `+VertexLayout.m` | 203 | 薄化（O5.3 本刀：`generateVertexDescriptorState` 已沉 C 函数 `mglRenderGenerateVertexDescriptorState`，ObjC 仅薄转发） | `updateBlendStateCache` 写 `_pipelineCache`（ObjC 物化，留）；`bindFramebufferAttachmentTextures` 实为 FBO 绑定，应归 RenderPass 域 |

### 1.2 其它 `.m`（非 category，但同边界）

| 文件 | 约 LOC | 判定 |
|------|-------:|------|
| `mgl_draw_encode.cpp` | ~1187 | **已迁出 ObjC**（O5.4 DONE）：原 `.m` 整文件重命名为 `.cpp`，剥除 6 处 `__bridge`，经 Makefile `wildcard MGL/src/*.cpp` 自动纳 non-ARC C++；draw-encode 决策不再属 ObjC 边界 |
| `mgl_batch_flush_restore_encode.c`（原 `.m`） | ~380 | **已迁出 ObjC**（T4）：flush/restore/check/stream/schedule 六个 driver 变 C；`@try/@finally` 帧留在 shim（`mglRendererFlushDrawBufferLockedPort`） |
| `mgl_batch_dyn_bind_encode.c`（原 `.m`） | ~330 | **已迁出 ObjC**（T4）：六个 driver 变 C（`mglBatchDynBind*` / `mglBatchApplySamplerSnapshot` / `mglBatchApplyDynamicBindings` / `mglBatchTryReplaySimpleBatch`） |
| `mgl_batch_issue_encode.c`（原 `.m`） | ~213 | **已迁出 ObjC**（T4，`mgl_batch_issue_encode` 转 C）：MDI/direct loops → `mgl_batch_mtl_issue_mdi_batch` / `mgl_batch_issue_direct_batch`；driver 变成 C 函数 `mglBatchIssueMDIBatch` / `mglBatchIssueDirectBatch` |
| `mgl_batch_replay_trace.c`（原 `.m`） | ~258 | **已迁出 ObjC**（T4）：trace 两个入口变 C driver `mglBatchTraceReplayBatch` / `mglBatchTraceReplayCommand`（声明在 `mgl_batch_rt_mark.h`）；**禁止再扩** |
| `mgl_batch_icb_mdi_encode.c`（原 `.m`） | ~152 | **已迁出 ObjC**（T4）：ICB/stream-MDI driver 变 C（`mglBatchIssueStreamMergedMDIBatch` / `mglBatchIssueIndirectCommandBufferBatch`）；唯一留下的 ObjC 是 ICB 分配失败的 `@try/@catch`，收在 shim 端口里 |
| ~~`mgl_batch_rt_mark_port.m`~~ | 0 | **已删**（T4 首切片）：RT-mark host 段转 C（`mgl_batch_rt_mark_host.c`） |
| `hash_table.m` | ~854 | 平台资源表；可保留或 C++ owner |
| `MGLRenderPassManager.m` | ~523 | 并入 RenderPass 下沉 |
| `MGLPipelineCache.m` | ~445 | **已薄端口（O3.4 实质完成）**：LRU/archive 策略在 C++ owner（`mglRenderLookupPipeline`/`StorePipeline`）；ObjC 仅 `id`↔`void*` 桥 + archive URL 路径；可按 O3.4 收口标记 [x] |
| `MGLPlatformRendererShell.m` | ~229 | **Keep-thin → T5 唯一壳**（与 `+Lifecycle` 合并为一 TU） |
| `mgl_readback.m` 等 compat | 小 | 策略进 ReadbackPolicy；`.m` 变转发 |

> **实测（2026-09-12；联合报告后口径）**：`MGLRenderer+*.m` categories 合计 **34.5k** LOC（基线 ~59k；O1/O2/C1 已降 ~24.5k）。**终态是清零**（T5 唯一壳除外）——**禁止**再以「薄平台 ≤8–12k 即终态」当目标。`MGLRenderer+Texture.m`+`+RenderPass.m`+`+Blit.m` 三厚块 = **~19.0k**（联合报告 ≈19,021 / 约占 `MGLRenderer*.m` 的 55%），仍是 P0-1 Rewrite 主体（**禁止整文件 Delete**）。`MGLPipelineCache`/`+VertexLayout`/`mgl_batch_*_encode` 已呈薄端口/ops 形，不应再计入「待沉厚代码」。
>
> 本周期新增回落（同日多刀，见 §5 第 25–28 条）：`+Buffer.m` 1479→**826**（vertex-attrib buffer map 沉 `mgl_vertex_attrib_plan.*` 且删掉 544 行 reflection fallback）、`+RenderPass.m` 退役 6 处源码文本扫描、3 份 sampler 启发式实现合并为 1 个共享谓词。度量：`scripts/objc_renderer_loc.sh`（当前输出 `MGLRenderer*.m total: 34555`，`+Buffer.m` 822）。

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

度量：`scripts/objc_renderer_loc.sh` + `scripts/objc_zero.sh`（**清零**为终态，非 ≤8–12k；Batch 簇按 Track B 诚实口径，含 encode/trace；升权指标见 §0.06）。

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

- [ ] **O6.1** 合并剩余端口（联合报告钉死）：
  - **`MGLPlatformRendererShell` + `+Lifecycle` → 唯一平台壳 TU**（layer/drawable/swap/view + create/bind/lease/dealloc；不再「双文件压缩一下」）
  - `MGLRenderer+MetalPort.m`（可选：所有 `id` 物化一行口）
  - `MGLRenderer+GPURecovery.m`（薄）
- [ ] **O6.2** 删除空 category：`+DrawSupport` / `+BatchReplay` / `+Binding` / `+VertexLayout` / …
- [ ] **O6.3** `MGLRenderer.m` 降到入口表 + 文档化 C ABI
- [ ] **O6.4** ObjC 清零达标（T5 唯一壳除外）；ARCH 表格更新为「唯一平台壳」——**禁**再写「≤8–12k 即终态」

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
- [ ] **O7.4 剩余候选（联合报告升 P1）**：
  1. **P1 名字启发式**：`mglDefaultAttribLocationForName` / `mglContextualDefaultAttribLocationForName` /
     `mglRendererTextureLooksLikeSampledColor2D`——先探针 `gl_type==0` / 命中率，再 Delete 或收表（禁止无 oracle 盲删）；
  2. 链接期重复 parse 去重（`mglShaderInterfaceCheck` 复用 `frontend_tu`，属 O5 类）。
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

36. **当前下一刀（联合报告 §7 对齐后；暂停期 / BindingState stash 未解除前优先设计）**：
    1. **P0-0**：air `esrc`×24 → `builtin_mask`/IR（oracle-equal 后删）；`emitTessBlock*` 外提；**停止**为 tess 胀 `mgl_air_backend`。
    2. **P0-1**：三厚块 materialize/upload 设计+金样（禁止整文件 Delete；禁新增 ObjC 行）。
    3. **P0-2**：下一 Batch 必须 shim 净减（优先 multi/shared）；`flush_restore→C` alone = 拒收假进度。
    4. **P1**：BindingState 一口 plan（stash 仅形状参考）；名字启发式探针；metal_port 禁扩。
    5. chore：DontCare 死声明；Lifecycle→唯一壳；`objc_zero` 计数对齐。
  - **禁则（加严）**：无 shim 净减 = 拒收；扩 `mgl_draw_metal_port.m`、胀 `mgl_air_backend.cpp`、新开厚 category、堆进 `mgl_render.cpp`；不得以「CTS 没跑到」代替 oracle；**禁**「≤8–12k 即终态」叙事。

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

40. **T2′ 首批：`mgl_readback` 转 C + 删空占位 category（`b3e8ea0`）**：
    ① `mgl_readback.m` → `.c`：`BOOL/YES/NO`→`bool/true/false`、`NSUInteger/NSInteger`→`size_t/ptrdiff_t`、
    去掉 `<Foundation/Foundation.h>`；其中真正需要改语义的一处是 **scratch buffer**——
    `NSMutableData *bgra = [NSMutableData dataWithLength:…]`（`.mutableBytes`/`.bytes`）换成
    `calloc(bgraBytesPerRow * height, 1)` + `free`（保持 `dataWithLength:` 的零填充语义），溢出检查
    `NSUIntegerMax`→`SIZE_MAX`；头文件同步（`<stdbool.h>`/`<stddef.h>`）。
    ② 删除 `MGLRenderer+BatchReplay.m`：一个**空 `@implementation`** 占位 category（文件自注"待 O6 删除"），
    方法实现早已在 `mgl_batch_dyn_bind_encode.m`。
    ③ 本批**没有新增 harness**——机械转换的 oracle 就是既有套件（`make test-all` 的 smoke/es-smoke/legacy-compat/
    regression + CTS 整轮），文档在此如实标注。
    **结论性分析（写入待办）**：`mgl_trace_log.m`（438 行，4 处语法/9 处词汇）**不能**并入 T2′——
    它的 ObjC 面是 `mglTraceLogNSString*` 两个入口，而全仓有 **~80 处调用点**（`+RenderPass` 最多），
    且部分格式串含 `%@`；要清零必须先把调用点改成 C API（`mglTraceLog` + UTF8 字符串），属 **T4** 量级，
    故它保持 `.m`（其 ObjC 声明已在 `#ifdef __OBJC__` 内，C TU 可安全包含该头）。
    度量：文件 **30 → 28**、行数 **40,044 → 39,751**、语法 **2,239 → 2,231**、词汇 **4,266 → 4,209**。
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 整轮（hotspot 非通过集合 diff 为空）。
    下一批：**T4 首切片**（挑一个 `__bridge`/`id` 端口文件改 C++ + `void*`）：`mgl_batch_rt_mark_port`(140 行) /
    `mgl_batch_replay_trace`(257) / `mgl_batch_issue_encode`(218) 三选一；`mgl_trace_log` 的 80 处调用点改造列入其后。

41. **T4 首切片：C 端口面 + batch-RT-mark 端口转 C（`4dd24c2`）**：
    ① 新建 **C 端口面** `mgl_renderer_ports.{h,c}`：`mglRendererAttachmentTextureFor(ctx, att)`（`-[MGLRenderer
    framebufferAttachmentTexture:]` 的逐行 C 版：renderbuffer→`rbo->tex`、纹理→缓存 `buf.tex` 或 `findTexture`
    并回填、NULL/无纹理各记一条 stderr）与 `mglRendererRenderPassStateOwnerPort(void *renderer)`（实现在
    `mgl_draw_metal_port.m`，与既有 `mglRendererRenderPassManager` 并列；文档里的 dual-proxy 不变式保证
    `_activeState == ctx->active_state`，故 C 侧用后者）。`+RenderPass.m` 的该方法**变成一行转发**（−33 行）。
    ② `mgl_batch_rt_mark_port.m`（140 行）**整体转 C** → `mgl_batch_rt_mark_host.c`：三个 ObjC 方法改成
    `mglBatchRtMarkColorAttachmentWritten(renderer, ctx, index)` 与
    `mglBatchRtMarkCurrentFramebufferDrawAttachments(renderer, ctx)`（声明进 `mgl_batch_rt_mark.h` 的 host 段），
    原 `__bridge id` 全部成为 `void *`；三个 ObjC 调用点改为直调；`MGLRenderer (BatchRtMark)` 声明与端口文件删除。
    **拆分教训**：host 段落先写进了 plan 文件 `mgl_batch_rt_mark.c`，`make test-all` 立刻抓到
    `test_batch_issue` 链接失败——该 harness 只链 plan 文件。按仓库"plan 模块可独立链接"的原则拆成独立
    `mgl_batch_rt_mark_host.c`，plan 文件恢复自包含。
    度量：文件 **28 → 27**、行数 **39,751 → 39,598**、语法 **2,231 → 2,216**、词汇 **4,209 → 4,205**。
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 整轮（hotspot 非通过集合 diff 为空）。
    下一批候选：`mgl_batch_replay_trace`(257) / `mgl_batch_issue_encode`(218) / `mgl_batch_icb_mdi_encode`(178)
    同法转 C；`mgl_trace_log` 的 ~80 处 `mglTraceLogNSString` 调用点改 C API（其后可清 438 行 ObjC）。

42. **`mgl_trace_log` 清零：76 处调用点改 C API（`6cd8567`）**：先前的分析说它要等 T4——本轮做的正是这件事。
    ① **调用点改造**：全仓 76 处 `mglTraceLogNSString(@"…", …)` → `mglTraceLog("…", …)` 按**调用跨度**精确改写
    （不碰文件里其它 `@"` 字面量），分布：`MGLRenderer.m` 34 · `+RenderPass` 14 · `+SwapDiagnostics` 9 ·
    `+Texture` 8 · `+GPURecovery` 3 · `+Blit` 3 · `+Buffer` 2 · `+Binding` 1 · `mgl_batch_flush_restore_encode` 1 ·
    `mgl_trace_log.m` 自身 1。
    ② **10 处含 `%@` 的**逐条改：格式串 `%@`→`%s`，实参包 `[x UTF8String]`（`reason`/`stage` 为 `NSString *`、
    `sampleTag`/`sampleTagCopy`/`sampleError` 同上、`reason ?: @"(none)"` → `reason ? [reason UTF8String] : "(none)"`）。
    ③ 删除两个 NSString 入口（`mglTraceLogNSStringV`/`mglTraceLogNSString`）及其在 `mgl_trace_log.h` 的
    `#ifdef __OBJC__` 声明与 Foundation 保护块；`mgl_trace_log.m` → **`.c`**（补 `<dispatch/dispatch.h>`——
    `dispatch_once` 本就是 C API）；Makefile 里它的 `METALCPP_OBJC_SRC` 改为 C 列表，并删掉已空的 ObjC 对象规则。
    **度量口径提醒**：语法次数反而 **2,216 → 2,223**——因为 `[x UTF8String]` 本身算一次消息发送，而调用点所在的
    ObjC 文件尚未转 C；这是过渡态的正常现象，文件数与行数才是本刀的真实收益。
    度量：文件 **27 → 26**、行数 **39,598 → 39,160**、语法 2,216 → 2,223（+7，见上）、词汇 **4,205 → 4,196**。
    验证：两个库构建无错；**`make test-all` 返回 0**（smoke / es-smoke / legacy-compat 193/193 /
    test_regression 92/0/2 / 各 plan harness）；CTS 整轮（hotspot 非通过集合 diff 为空）。
    下一批：`mgl_batch_replay_trace`(257) / `mgl_batch_issue_encode`(218) / `mgl_batch_icb_mdi_encode`(178)
    三个端口文件转 C——它们的依赖是**其它端口文件里的 ObjC 方法**（`traceReplayCommand` / `issueDirectBatch` /
    `tryReplaySimpleBatch` / `mdiArgumentScratchBuffer` / `resolveElementBufferForCommand`），
    需先按 T4 建立"C 侧 batch 端口面"（一函数一包装 + `void *renderer`），再逐个转。

43. **`mgl_batch_issue_encode` 转 C：batch issue 端口面落地（本轮）**：第 42 条列出的三个候选里先做 issue/direct 这一个。
    ① **文件转 C**：`mgl_batch_issue_encode.m`(218 行) → `mgl_batch_issue_encode.c`(213 行)。两个 ObjC 方法
    `-[MGLRenderer issueMDIBatch:context:encodeContext:]` / `-[MGLRenderer issueDirectBatch:context:encodeContext:]`
    变成 C driver `mglBatchIssueMDIBatch(renderer, batch, ctx, enc)` / `mglBatchIssueDirectBatch(...)`
    （声明进 `mgl_batch_issue.h`，`MGLRenderer*` → `void *`），`MGLRenderer+Draw_Private.h` 里两条方法声明删除，
    `mgl_batch_flush_restore_encode.m` 三处调用点改直调（ObjC 侧加 `(__bridge void *)c->r`）。
    ② **每处 renderer 调用都换成既有 C 端口**（不新增 ObjC）：device → `mglDrawHostDevice(renderer)`；
    `encodeCullDistanceElementDraw:…` → `mglDrawHostEncodeCullDistanceElementBytes(...)`；
    `bindCullDistanceEmulationBuffers:…` → `mglRendererBindCullDistanceEmu(...)`（其声明从 ObjC 头
    `MGLRenderer+Draw_Private.h` 复制进 C 安全头 `mgl_draw_issue.h`）；scratch / trace / simple-replay /
    dyn-bind / sampler / cull-capture / `processGLState` 走 `mgl_renderer_ports.h` 的 shim 端口。
    ③ **C 安全头补口**：`MGLEncodeContext` 原定义在 ObjC 头 `MGLRenderer+Draw_Private.h` 里，C TU 无法包含 →
    搬进新头 `mgl_encode_context.h`（ObjC 头改为 `#include` 它）；`mglResolveProgramForStageFromState` 此前被
    **7 个 `.cpp` 各自手写 `extern "C"` 原型**，现声明进 `mgl_render.h` 的 `extern "C"` 段（`mgl_batch_issue.h`
    也新增 `glm_context.h`+`draw_command.h`+`mgl_encode_context.h` 依赖，driver 原型才能是 C 安全的）。
    ④ **端口去重**：删掉 shim 里的 `mglRendererMetalDevicePort`（与既有 `mglDrawHostDevice` 完全同义，−6 行）；
    shim 只保留**命令版** element-buffer 解析 `mglRendererResolveElementBufferPort`（`resolveElementBufferForCommand:`
    与 `mglDrawHostResolveElementBuffer` 的 `resolveElementBufferForDraw:` 语义不同：前者按命令自带的
    `element_buffer_name` 解析，不能混用）；C 侧 env 判定改用 inline `mgl_env_flag_enabled("MGL_DISABLE_MDI")`
    （与 `mglEnvFlagEnabled` 语义相同：unset/空 → 0），索引类型映射改用 `mglRenderMTLIndexTypeForGLType`。
    **教训（链接一致性）**：第一版把 `mglResolveProgramForStageFromState` 加到 `mgl_types_program.h`，但该头的
    `extern "C"` 块只覆盖 XFB 几个函数 → 声明落到 C++ 链接，7 个 `.cpp` 的局部 `extern "C"` 原型立刻报
    `declaration … has a different language linkage` + `call is ambiguous`；放进 `mgl_render.h` 的 `extern "C"` 段
    （所有 `.cpp` 都包含它）后一次通过，且**不必改那 7 个 `.cpp`**。
    度量：文件 26（`.m`→`.c` 换 1 个、shim 新 1 个，净持平）、行数 **39,160 → 39,113**、语法 **2,223 → 2,230**
    （`+19` shim、`+3` 桥接、`−14` 转 C、`−1` shim 收窄，见 §0.0 口径提醒）、词汇 **4,196 → 4,189**；
    `MGLRenderer*.m` total **34,610 → 34,557**。
    验证：两个库构建无错（exit 0）；**`make test-all` 返回 0**（smoke / es-smoke / test_regression 92/0/2 /
    各 plan harness）；CTS 整轮 **七簇非通过集合逐条 diff 全为空**：hotspot 1270/52/4/1+1cw · tess 139/1 ·
    GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns。
    （推送时远端多出 `da51b1c`——`mgl_byte_hash.c` 里 trace 转换遗留的 `%@`→`%s` 修复，由 GitHub 网页提交；
    本刀**变基到它之后重建、重跑 `make test-all` 与整轮 CTS**，七簇数字与非通过集合 diff 与上表完全一致。）
    下一批：`mgl_batch_replay_trace`(257) / `mgl_batch_icb_mdi_encode`(178) / `mgl_batch_dyn_bind_encode`(367) /
    `mgl_batch_flush_restore_encode`(383) 四个 batch 端口文件同法转 C（shim 端口已备齐，逐个转 C 时把对应包装
    搬进实现文件、shim 随之缩小）。

44. **`mgl_batch_replay_trace` 转 C：trace 两个入口变 C driver（本轮第二刀）**：
    ① **文件转 C**：`mgl_batch_replay_trace.m`(256 行) → `mgl_batch_replay_trace.c`(258 行)。两个 ObjC 方法
    `-[MGLRenderer traceReplayBatch:context:flushId:batchIndex:phase:]` /
    `-[MGLRenderer traceReplayCommand:command:context:flushId:batchIndex:commandIndex:phase:reason:]` 变成
    `mglBatchTraceReplayBatch(...)` / `mglBatchTraceReplayCommand(...)`（声明进 `mgl_batch_rt_mark.h`，
    `void *renderer` 首参），7 处调用点改直调（`mgl_batch_flush_restore_encode.m` 6 处、`mgl_batch_icb_mdi_encode.m` 1 处），
    `mgl_batch_issue_encode.c` 的 trace 回调也从 shim 端口改为直调（shim 少一个端口）。
    ② **`mglWriteProgramMSLDump` 去 NSString**：`NSString *reason` → `const char *reason`（`forceDump` 用
    `strcasestr(reason,"tex")` 复刻原 `[lowercaseString] containsString:` 的语义；日志行 `%@`→`%s`），
    三个调用点改 C 字符串：`+BindingState.m` 用 `snprintf` 组 reason、`+RenderPass.m` 直接用 `cppError`
    （原本就是由它 `stringWithUTF8String:` 构造的 `errDesc`，语义等价）、trace 文件里两处用 `snprintf`。
    ③ **C 安全头补口（本轮共 5 处声明搬家）**：`mglTraceResolveDrawProgram` / `mglTraceShouldLogReplay` /
    `mglTraceFramebufferAttachmentTexture` / `mglTraceReplayCommandVertexAttribSamples` / `mglWriteProgramMSLDump`
    → `mgl_trace_strategy.h`；`mglRendererGetValidatedVAO` → `mgl_vertex_attrib_query.h`；
    `mglCurrentRenderProgramKey` → `mgl_render.h`。它们此前只声明在 ObjC 头里，C TU 无法包含。
    ④ **新端口（shim +4）**：`mglRendererFragmentTraceBindingsPort`（`_resourceFallback.fragmentTextureTraceBindings`，
    返回具名类型 `MGLFragmentTextureTraceBinding *`）、`mglRendererPipelineStatePort`、
    `mglRendererPipelineProgramNamePort`、`mglRendererRenderPassFramebufferNamePort`；
    `mglRendererObjectPointerLikelyValid`（ObjC 头的 `static inline`）在 C 侧直接用 `mglObjectPointerLooksPlausible`。
    **新做法（本刀首次用）：trace 文本做 A/B oracle**。CTS 默认不开 trace，这条路径不在常规语料里——于是
    ① 用当前（C）库跑 `MGL_TRACE_LOG=1 MGL_TRACE_LOG_DRAW=1 MGL_TRACE_LOG_RESOURCES=1 build/test_regression all`；
    ② `git stash` 回到旧（ObjC）库重建、同一命令再跑一遍；③ 把两份日志按"时间戳/fid/tid/pid/路径/指针"归一化后
    比对 `REPLAY_*` 与 `IFACE DUMP` 行：**373 行逐行逐字段完全一致**（唯一差异是全局 `fid` 计数整体差 1，
    来自与本刀无关的慢路径诊断帧）。这比"CTS 没跑到所以不算"要强：改动过的代码路径有了逐字段等价证据。
    度量：文件 **26 → 25**、行数 **39,113 → 38,863**、语法 **2,230 → 2,218**、词汇 **4,189 → 4,184**、
    `MGLRenderer*.m` **34,557 → 34,555**。
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 七簇非通过集合 diff 全为空（hotspot 1270/52/4/1+1cw ·
    tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）；trace A/B 如上。
    下一批：`mgl_batch_icb_mdi_encode`(178) / `mgl_batch_dyn_bind_encode`(367) / `mgl_batch_flush_restore_encode`(383)
    三个 batch 文件转 C（端口已备齐；icb 含 `@try/@catch`，需按 C 侧错误码改写并单独说明）。

45. **`mgl_batch_icb_mdi_encode` 转 C：ICB / stream-MDI driver（本轮第三刀）**：
    ① **文件转 C**：`mgl_batch_icb_mdi_encode.m`(178) → `.c`(152)。两个 ObjC 方法
    `-[MGLRenderer issueStreamMergedMDIBatch:context:encodeContext:]`（返回 `BOOL`）与
    `-[MGLRenderer issueIndirectCommandBufferBatch:context:encodeContext:]` 变成 C driver
    `mglBatchIssueStreamMergedMDIBatch` / `mglBatchIssueIndirectCommandBufferBatch`（返回 `int`，声明进
    `mgl_batch_issue.h`），`mgl_batch_flush_restore_encode.m` 两处调用点改直调。
    `-[MGLRenderer mdiArgumentScratchBufferWithLength:offset:]` 的**整体**搬进 shim 的
    `mglRendererMdiScratchBufferPort`（它的唯一调用者就是那个端口），ObjC 头里的四条声明一并清掉。
    ② **`@try/@catch` 的处置**：`@try { mgl_batch_mtl_create_icb } @catch (NSException *)` 无法用 C 表达（Metal 在
    ICB 分配失败时抛 NSException），因此这一小段**留在 shim**：`mglRendererCreateIndirectCommandBufferPort(renderer,
    indexed, count, &failed)`，内含原样的限流 `NSLog`，C 侧只按 `failed` 决定 trace 的 fallback 原因
    （`icb_create_exception` / `icb_create_nil`）。所有权：端口返回 **+1**，C 侧 `CFRelease`——与原
    `__bridge_transfer` + `__bridge_retained` 的净效果一致。
    ③ **C 化替换**：`@available(macOS 10.14, *)` → `__builtin_available`（C 也可用）；
    `resolveElementBufferForCommand:` → `mglRendererResolveElementBufferPort`；`mglIndexTypeForGLType` →
    `mglRenderMTLIndexTypeForGLType`；`[r processBuffer:]` → 新端口 `mglRendererProcessBufferPort`；
    `_device` → `mglDrawHostDevice`；`mglEnvFlagEnabled` → `mgl_env_flag_enabled`；scratch 端口新增
    `mglRendererProcessBufferPort`（共 +2 端口）。
    ④ **踩坑（这次被聚合门禁抓住）**：第一版把 `mglRendererMdiScratchBufferPort` **直接**填进 ops 表的
    `.alloc_scratch`——但 ops 表传的第一参是 **ctx（`MGLIcbMdiCtx *`）而不是 renderer**，端口把它当
    `MGLRenderer *` 解引用 → `test-dirty-hash` **段错误**。`make test-all` 立刻报 `Error 2`（不是只挑 harness 跑），
    修法是保留 `mglIcbScratch(void *v, …)` 解包 `((MGLIcbMdiCtx *)v)->r` 再调端口。
    **教训**：端口签名"看起来一样"不等于调用约定一样——ops 表 vtable 的首参语义必须逐个核对。
    ⑤ **oracle**：两套 trace 语料 A/B（默认 + `MGL_ENABLE_ICB=1`）——`REPLAY_*`+`IFACE DUMP` 行归一化后
    **373 行 / 295 行逐字段一致**；ICB 轮的回归结果 82 PASS/10 FAIL/2 SKIP 与旧库**完全相同**（该 10 项是 ICB
    opt-in 路径的既有失败：`Fragment/Vertex shader cannot be used with indirect command buffer`，非本刀引入）。
    度量：文件 **25 → 24**、行数 **38,863 → 38,724**、语法 **2,218 → 2,207**、词汇 **4,184 → 4,170**。
    验证：两个库构建无错；**`make test-all` 返回 0**（本刀第一次跑就是它抓到的段错误）；CTS 七簇非通过集合 diff
    全为空（hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns ·
    pp 1/3/1ns）；trace A/B 如上。
    下一批：`mgl_batch_dyn_bind_encode`(367) / `mgl_batch_flush_restore_encode`(383)——后者是本簇最后一个文件，
    做完 Batch 簇的 `mgl_batch_*_encode` 就不再是 ObjC。

46. **`mgl_batch_dyn_bind_encode` 转 C：dyn-bind / sampler / simple-replay（本轮第四刀）**：
    ① **文件转 C**：`mgl_batch_dyn_bind_encode.m`(366) → `.c`(330)。六个入口变 C driver（声明进 `mgl_batch_issue.h`）：
    `mglBatchDynBindVertexDirect` / `mglBatchDynBindUniformDirect` / `mglBatchDynBindSampledDirect` /
    `mglBatchApplySamplerSnapshot` / `mglBatchApplyDynamicBindings` / `mglBatchTryReplaySimpleBatch`
    （原来返回 `bool`/`BOOL`）。**退掉三个 shim 端口**——`mglRendererApplyDynamicBindingsPort` /
    `mglRendererApplySamplerSnapshotPort` / `mglRendererTryReplaySimpleBatchPort` 的 ObjC 实现就是这三个方法，
    现在 C 侧直调；`mgl_batch_issue_encode.c` 的三处回调、`mgl_batch_flush_restore_encode.m` 的 `cApplyS` 跟着改直调。
    ② **新端口 12 个**（shim）：`mglRendererBindingStateOwnerPort`、`mglRendererUpdateDirtyBaseBufferListPort`、
    `mglRendererBindMTLBufferPort`、`mglRendererMapBuffersToMTLPort`、`mglRendererBind{Vertex,Fragment}BuffersToCurrentRenderEncoderPort`、
    `mglRendererBindTexturesToCurrentRenderEncoderPort`、`mglRendererRestoreRenderEncoderAfterTextureUploadPort`、
    `mglRendererTextureUnitForSampledResourcePort`、`mglRendererTextureForSampledResourcePort`、
    `mglRendererSamplerStateForSnapshotKeyPort`（把 `-samplerStateForSnapshotKey:` 的**方法体**搬进端口，返回非持有引用，
    与原来的 ARC 语义一致）、`mglRendererFallbackSamplerStatePort`。
    ③ **决策/常量搬到 C 安全头**：`kMGLMaxBufferSlots`(31) 与 `kMGLMinimumStageBindingSize`(256) → `mgl_buffer_slots.h`
    （`mgl_render.cpp` 里那个重复的 `constexpr kMGLMaxBufferSlots` 删掉、`kMinimumStageBindingSize` 改为引用该常量，
    消灭第二处真值来源）；`mglMipDiagEnabled` / `mglMipDiagStateChanged` / `mglMipDiagMixState` 从
    `MGLRenderer_Private.h` 的 ObjC inline 搬到 `mgl_state_log.h/.c`（仍用单源 `mgl_env_flag.h` 解析、
    仍用 `dispatch_once`），ObjC 调用点（`+Binding.m` / `+BindingState.m`）不变；
    `mglRendererResolveVertexAttributeBufferIndex` 声明进 `mgl_vertex_attrib_query.h`。
    ④ **一处可见行为变化（如实记录）**：`mglSampAfter` 的 `MGL MIP_DIAG` 输出从 `NSLog` 改为
    `fprintf(stderr, …)`——**sink 仍是 stderr、前缀仍是 `MGL MIP_DIAG`、字段与取值完全一致**，只是不再带 NSLog 的
    `时间戳 进程[pid:tid]` 前缀（`MGL_MIP_DIAG` 明示"与 MGL_TRACE_LOG 无关"，所以不能用 `mglTraceLog`）。
    ⑤ **oracle（三套，都做了 A/B）**：trace 语料 `REPLAY_*`+`IFACE DUMP` 归一化后 **373/295 行逐字段一致**；
    `MGL_MIP_DIAG=1` 的 stderr 语料剥掉 NSLog 前缀后 **529/212 条消息逐字段一致**；ICB 轮回归 82/10/2 与旧库相同。
    ⑥ **度量口径提醒（重要）**：本刀语法 **2,207 → 2,209（+2）**——转掉的 22 处 ObjC 大部分变成 shim 里的包装
    （12 个新端口，每个约"1 次消息发送 + 1 次 `__bridge`"），另有 3 个端口退役抵消一部分。文件 **24 → 23**、
    行数 **38,724 → 38,452**、词汇 **4,170 → 4,152**。**shim 现状：26 个包装 / 314 行 / 49 语法**——这是 Batch 簇
    收敛后剩下的 ObjC 面，也是下一阶段的主要债务：要么把包装对应的 ObjC 方法体继续下沉进 C（T3），要么整块按
    T4 转 C++/metal-cpp（届时对应用户端方法可删）。
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 七簇非通过集合 diff 全为空（hotspot 1270/52/4/1+1cw ·
    tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
    下一批：`mgl_batch_flush_restore_encode`(383)——Batch 簇最后一个 `mgl_batch_*_encode` 文件。

47. **整文件删除 `MGLRenderer+Batch.m`：成员按域落 C + ObjC 壳进 shim（本轮第五刀）**：
    ① **按域分配**（不是堆进一个"杂物 C 文件"）：`recordArrayDrawSubmittedMode:vertexCount:` /
    `recordElementDrawSubmittedMode:indexCount:` → `mglBatchRecordArrayDrawSubmitted` /
    `mglBatchRecordElementDrawSubmitted`（`mgl_batch_rt_mark_host.c`，声明进 `mgl_batch_rt_mark.h`），
    `mgl_draw_metal_port.m` 两个 host 包装改直调；`bindActiveTexturesToMTL` → `mglBatchBindActiveTexturesToMTL`
    （`mgl_batch_replay.cpp`，新增 1 个端口 `mglRendererBindMTLTexturePort`），ObjC 调用点
    `+Binding.m` / `+RenderPass.m` 改 `mglBatchBindActiveTexturesToMTL((__bridge void *)self, ctx)`；
    `restoreStateFromKey:context:` → `mglBatchRestoreStateFromKey`（**新 TU** `mgl_batch_restore_host.c`，见③），
    唯一调用点 `mgl_batch_flush_restore_encode.m` 改直调。
    ② **ObjC 壳进 shim**：dual-proxy 三件套（`mglActivateReplayStateForContext:` /
    `mglRestoreLiveActiveStateForContext:` / `mglAssertDualProxyInSyncForContext:`）、`flushDrawBuffer:`（METAL_LOCK +
    `@try/@finally`）与 C 入口 `mglRendererFlushDrawBuffer`（`@autoreleasepool` + `@try/@catch` 兜底）以
    `@implementation MGLRenderer (BatchZeroShell)` 落在 shim——它们是纯 ivar/异常/锁壳，原封不动搬过去，
    ObjC 调用点（`+Blit.m` / `+RenderPass.m` / flush_restore 的 5 处断言）**一行都不用改**。
    ③ **计划文件必须能独立链接（教训第二次）**：`mglBatchRestoreStateFromKey` 第一版写进 **plan 文件**
    `mgl_batch_restore.c`，`make test-all` 立刻报 `test_batch_restore` **链接失败**
    （缺 `mglRestoreProgramPipelinePair` / `mglRendererSyncFramebufferBindingNames` / `searchHashTable`）——
    与之前 `mgl_batch_rt_mark_host.c` 的教训同源。已拆出 host TU `mgl_batch_restore_host.c`，plan 文件恢复自包含。
    ④ 顺带把 `mglTraceNowSeconds` 从 ObjC 头搬进 C 安全头 `mgl_trace_log.h`（`mglNowSeconds` 仍留 ObjC，
    因为它用 CFAbsoluteTime）；`mglRestoreProgramPipelinePair` / `mglRendererSyncFramebufferBindingNames` 声明进
    `mgl_render.h`。
    ⑤ **oracle**：trace 语料 374/296 行、`MGL_MIP_DIAG` 语料 529/212 条，归一化后与旧 ObjC 库**逐字段一致**；
    回归 92/0/2 与 ICB 轮 82/10/2 与旧库相同；`flushDrawBuffer` 是每次绘制都走的路径，CTS 七簇也覆盖。
    度量：文件 **23 → 22**、行数 **38,452 → 38,352**、语法 **2,209 → 2,204**、词汇 **4,152 → 4,152**、
    `MGLRenderer*.m` **34,555 → 34,386**；shim 现 **26 端口 + 5 方法 / 382 行 / 61 语法**。
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 七簇非通过集合 diff 全为空（hotspot 1270/52/4/1+1cw ·
    tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
    下一批：`mgl_batch_flush_restore_encode`(383)——Batch 簇最后一个 `mgl_batch_*_encode`；它的
    `@try/@finally` 主体、`teardownBatchReplayForContext:`（已有 `MGLBatchTeardownOps` 计划可复用）与
    ops 回调是主体工作。

48. **联合报告吸收（docs-only，2026-09-13）**：三方已签
    [`OBJC_LLVM_JOINT_AUDIT_2026-09-14.md`](OBJC_LLVM_JOINT_AUDIT_2026-09-14.md)
    （审计 tip `4ed1d54`＝23 文件 / ~38.5k LOC；交付日用户把截止从周一改到 **周日 08:00**；当前 tip `88ac73d`＝**22**）。
    本文新增 §0.05–§0.07：Delete/Rewrite/Keep-thin/Keep-product-dual/Keep-A/B-temporary 总表
    （ICB 门禁 **92/0/2** vs 现状 **82/10/2**；禁 Fragment/Vertex+ICB reject + AGX RECOVERY 日志；
    TES=Keep-product-dual；Metal-cpp A/B 死透）；T4 **无 shim 净减=拒收**；P0-0 `esrc`；
    T5 唯一壳 TU；名字启发式升 P1；过誉笔记（plan@C 贴皮 / shim 会计 / O7 esrc 未阻断 / 缺格式串门禁）；
    禁「薄平台 ≤8–12k 即终态」。**BindingState stash@{0} 未动**。无 `.m/.c/.cpp` 代码刀。

49. **`mgl_batch_flush_restore_encode` 转 C：Batch 簇收尾（本轮第六刀，但按新规不计 T4 进度）**：
    ① **六个 driver 变 C**（声明分落 `mgl_batch_restore.h` / `mgl_batch_issue.h`）：`mglBatchFlushBegin` /
    `mglBatchFlushRunBatches` / `mglBatchTeardownReplay` / `mglBatchScheduleDrawBatch` /
    `mglBatchCheckShouldExecute` / `mglBatchRecordCommandStats` / `mglBatchTraceSkipCommands` /
    `mglBatchTraceStreamCmd0` / `mglBatchIssueStreamMergedBatch` / `mglBatchRestoreStateForBatch`，
    以及全部 `f*`/`c*`/`s*` ops 回调（原为方法内 static 回调）。
    ② **`@try/@finally` 处置（与 ICB 同套路）**：`flushDrawBufferLocked:` 拆成 Begin（绑定 ctx、快照 live state、
    切 replay workspace，返回"有无 batch"）/ RunBatches（原 `@try` 体：flush 循环 + 每轮汇总日志）/ TeardownReplay
    （原 `@finally` 体），由 shim 的 `mglRendererFlushDrawBufferLockedPort` 用 `@try/@finally` 夹住——
    "异常时 teardown 一定跑"原样保留；`MGLBatchFlushPass` 在两侧传 hit/skipped/saved/saved_error/replay_error。
    ③ **新端口 14 个**：dual-proxy 断言、replay/live 切换、`_activeState` 直写、`_currentCBHasWork`、
    absolute-offset 读写、三个 batching 开关、batch arena 复位、trace-replay id、binding-state 有效性、
    render-pass 匹配与 `prepareRenderPassIfFBOChanged:`。
    ④ **两次失败（`make test-all` 抓到）**：`mglBindingStateIsValid` 搬进 `mgl_binding_stage.{h,c}` →
    `test_binding_stage`/`test_metalcpp_smoke` 链接失败（该 TU 被 harness 单独链接，且 smoke 里按 C++ 编译）→
    **结论：它必须留在 ObjC inline，C 侧走端口** `mglRendererBindingStateIsValidPort`；
    `kMGLDiagnosticStateLogs`（`static const BOOL = NO`）→ C 安全头 `mgl_trace_log.h` 的 `#define`（单一定义）。
    ⑤ **发现（未改行为）**：`mgl_batch_restore.h` 的两个"A3 整段 driver"计划
    （`mgl_batch_restore_run_for_batch` / `mgl_batch_teardown_run`）**只有 `test_batch_restore` 在用**，
    生产路径当初没接上；本刀按逐行等价转写，故它们仍是 harness-only —— 要么把新 driver 改成填 ops 调计划
    （推荐，消灭第二实现），要么删计划。**T3 决策债**。
    ⑥ **oracle**：trace 语料 374/296 行、`MGL_MIP_DIAG` 语料 529/212 条与旧 ObjC 库逐字段一致；回归 92/0/2 与
    ICB 轮 82/10/2 相同；flush 是每次绘制必经路径，CTS 七簇覆盖。
    度量：文件 **22 → 21**、行数 **38,352 → 38,099**、语法 **2,204 → 2,190**、词汇 **4,152 → 4,141**；
    `MGLRenderer*.m` **34,386 → 34,387**。**shim：26 → 40 个端口（+14），511 行 / 61 语法 ⚠️ 违反 §0.04 的
    「无 shim 净减＝拒收」，本刀只算"Batch 簇 ObjC 面清零 + 端口面集中"，不算 T4 净减。**
    验证：两个库构建无错；**`make test-all` 返回 0**；CTS 七簇非通过集合 diff 全为空（hotspot 1270/52/4/1+1cw ·
    tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。

### 0.08 下一刀的验收条件：shim 净减（按 §0.04 硬规）

> **状态（2026-09-13，P0-1 第八刀后）**：下表是"40 端口时代"的候选清单；本周期已推进到
> **端口声明 20 / shim 实现 20 / 331 行 / 47 语法**。真正有效的净减路线是 **"一个状态 struct + 一个端口"** 的
> 反向用法：像 `_batching`、`_command`（`MGLCommandState`）这类**纯 C 字段状态**一旦进了 `MGLRendererStateAreas`，
> 对应端口就应**逐个退役**（第七刀退役 `mglRendererBatchingStatePort` + `mglRendererTraceReplaySetPort`；
> 第八刀把 `mglRendererCommandStatePort`（29 处调用点）与 `mglRendererMdiScratchBufferPort` 也变成
> `mgl_renderer_ports.c` 里的 **C 函数**——它们本来只是 `areas.command` 的一个转发）。仍在下表里的候选按"能真删端口"排序。

Batch 簇已清空，剩余 ObjC 面集中在 **shim（40 端口 + 5 方法 / 511 行 / 61 语法）** 与 **21 个 `.m`**。
下一刀**只接受净减**，候选（按"能真删掉端口"排序）：

| 候选 | 现在的端口 | 净减路径 | 风险 |
|---|---|---|---|
| **`_batching` 状态域** | `AbsoluteVertexBindingOffsets`(读写) / `SetCurrentCBHasWork` / `SetActiveState` / `SkipSameKeyRestoreEnabled` / `DirtyKeyDeltaEnabled` / `ArenaSnapshotEnabled` / `ResetBatchArena` 共 **7** | 把这组标志 + batchArena 指针搬进 C 侧状态（`GLMContext` 或 `MGLRendererCore` 的 C 结构），shim 端口整批消失，C driver 直接读写 | 中：涉及 ivar 所有权与 `+Lifecycle` 初始化/销毁；需 CTS + trace A/B |
| **flush 帧** | `mglRendererFlushDrawBufferLockedPort` | 若确认 Metal 不会在 flush 内抛异常（探针：`MGL_ENABLE_ICB=1` 的 reject 路径 + AGX RECOVERY 场景），可去掉 `@try/@finally` 整段 | 高：异常时 teardown 语义改变，需明确证据 |
| **trace / pass 端口** | `TraceReplaySet` / `CurrentRenderPassMatchesFramebuffer` / `PrepareRenderPassIfFBOChanged` | 后两者属 `+RenderPass` 厚块；随 P0-1 的 materialize 下沉一起转 C++ | 中（绑定 P0-1） |
| **sampler / dyn-bind 端口** | `SamplerStateForSnapshotKey` / `FallbackSamplerState` / 采样资源两个 | 方法体已在端口内，下一步把 `createMTLSamplerForTexParam:` 与 backend 快照缓存的物化改 C++（metal-cpp） | 中 |
| **`+Batch.m` 留下的 category 5 方法** | dual-proxy 三件套 + `flushDrawBuffer:` + C 入口 | `mglRendererFlushDrawBuffer` 的 lease/autoreleasepool + `@try/@catch` 可保留为**唯一壳 TU** 的一部分（T5），dual-proxy 断言是 debug 检查 | 低-中 |

**纪律**：下一 PR 必须给出 `shim wrappers: 40 → N (<40)` 与 `shim LOC` 的**净减数字**，否则按 §0.04 拒收；
「再转一个 `.m`、端口照搬」不再算进度。

50. **shim 净减第一刀：`_batching` 状态域搬进 C（本轮第七刀，**按 §0.04 达标**）**：
    ① **做法**：`MGLBatchingState`（4 个 `BOOL` 开关 + `MGLBatchArena`）原在 ObjC 头 `MGLRenderer_State.h`，
    每个标志都要一个 shim 端口。现在结构体搬到新的 C 安全头 **`MGL/include/mgl_batching_state.h`**（字段改 `uint8_t`，
    `MGLBatchArena` 本就来自 C 头 `draw_command.h`），renderer 仍以 ivar 持有，**只用一个端口**
    `mglRendererBatchingStatePort()` 把地址交给 C，C driver 直接读写字段。
    ② **净减账**：退掉 6 个端口（`AbsoluteVertexBindingOffsets` 读/写、`SkipSameKeyRestoreEnabled`、
    `DirtyKeyDeltaEnabled`、`ArenaSnapshotEnabled`、`ResetBatchArena`），新增 1 个 →
    **shim 43 → 38 个端口、511 → 477 行、81 → 76 语法**；全仓语法 **2,190 → 2,185**、行数 **38,099 → 38,065**。
    ObjC 侧 `_batching.*` 的读点（`+Lifecycle.m` 12 处、`+BindingState.m`、`+VertexLayout.m`）**一行未改**——
    `uint8_t` 与 `BOOL` 在这些位置等价（赋值只可能是 0/1）。
    ③ **oracle**：trace 语料 374/296 行、`MGL_MIP_DIAG` 语料 529/212 条与旧库逐字段一致（A/B 走的是**真旧库**：
    第一次旧库构建因 stash 后 `.d` 依赖残留指向新头文件而失败，已 `find build -name '*.d' -delete` 重做，
    两库字节不同已核对）；回归 92/0/2、ICB 轮 82/10/2 与旧库相同；CTS 七簇 diff 全空。
    ④ **教训（记一次）**：`git stash` 做 A/B 时，`build/**/*.d` 里记着新头文件的依赖，会让"旧树"构建立刻
    `No rule to make target`；**清 `.d` 再构建**，并核对两库字节不同再跑对比，否则会把新库当旧库跑出"空 diff"假证据。
    下一刀（同样要求净减）：§0.08 表里按序取，优先 **`_currentCBHasWork`（1 端口，但要改 9 处 ObjC 写点，风险中）**
    或 **`+Batch.m` 留下的 category 5 方法（dual-proxy 断言/锁壳，可并入 T5 唯一壳）**；
    `P0-1` 三厚块与 `P0-2` 的 shim 记账需并行推进（三厚块的 materialize 下沉一次能退役多个端口）。

51. **shim 净减第二刀：`MGLCommandState` 搬进 C（本轮第八刀，**按 §0.04 达标**）**：
    ① **做法**：render pass manager 的 `MGLCommandState`（16 个字段：`renderPassStateOwner`、`renderPassFramebuffer(Name)`、
    `traceReplayFlushId/BatchIndex`、`currentRenderEncoderOwner`、两个 FBO-match 缓存字段等）原定义在 ObjC 头
    `MGLRenderPassManager.h`，C 侧每个字段一个端口。现在结构体搬到新的 C 安全头 **`mgl/include/mgl_command_state.h`**
    （两个 `BOOL` 字段改 `uint8_t`；其余字段本就是 C 类型），manager 仍以 ivar `_state` 持有，
    **一个端口** `mglRendererCommandStatePort()` 通过既有 readonly property `state` 把 `const MGLCommandState *` 交给 C。
    ② **净减账**：退掉 5 个端口——`mglRendererRenderPassStateOwnerPort`（实现在 `mgl_draw_metal_port.m`）、
    `mglRendererRenderPassFramebufferNamePort`、`mglRendererBatchTraceFlushIdPort`、`mglRendererBatchTraceBatchIndexPort`、
    `mglRendererCurrentRenderEncoderOwnerPort`；新增 1 个 →
    **shim 38 → 35 个端口、477 → 459 行、76 → 73 语法**；全仓语法 **2,185 → 2,181**、行数 **38,065 → 38,039**。
    ③ **改动面**：6 个 C driver（flush_restore / icb / issue / replay_trace / dyn_bind / rt_mark_host）里的
    `mglRendererXxxPort(r)` 全部改为 `mglRendererCommandStatePort(r)->xxx`；ObjC 侧 `_state.xxx` 读写一行未改。
    `mglRendererTraceReplaySetPort`（写 trace id 的 mutator，方法体是两行赋值）**保留为端口**——避免为了省一个包装
    而 cast 掉 `const`，收益不值风险。
    ④ **oracle**：trace 语料 374/296 行、`MGL_MIP_DIAG` 语料 529/212 条与旧库逐字段一致（旧库分支这次出现
    `dyld: Library not loaded` → **A/B 目录必须同时放 `libmgl.dylib` 与 `libglfw.dylib`**，已补；两库字节不同已核对）；
    回归 92/0/2、ICB 轮 82/10/2 与旧库相同；CTS 七簇 diff 全空。
    ⑤ **方法论沉淀（A/B 三坑，本轮各踩一次）**：① `build/**/*.d` 残留依赖会让 stash 后的旧树构建失败并**静默用新库跑对比**
    （第 50 条）；② A/B 目录缺 `libglfw.dylib` → `dyld` abort；③ 多个历史 trace 日志共存时按 mtime 取错分支。
    **固定动作**：`find build -name '*.d' -delete` → 构建 → `cmp` 两库 → 只取本轮日志 → 比对。
    下一刀（仍要求净减）：`MGLRendererCoreState`（`_core`，含 `activeState`/capability/drawable 尺寸/原子重置标志）
    同样可以搬成 C 结构 + 一个端口，可再退 1–2 个端口；更值钱的是 **P0-1 三厚块** 与 **`+Batch.m` 遗留的 category 5 方法**
    （后者可整体并入 T5 唯一平台壳 TU，一次退役 4 个方法 + 若干断言）。

52. **shim 净减第三刀：pipeline cache 状态搬 C + `_currentCBHasWork` 并入 batching 状态（本轮第九刀，**按 §0.04 达标**）**：
    ① `MGLPipelineCacheState`（`pipelineState`/`pipelineColor0Format`/`pipelineProgramName`/…/`dsCacheEnabled`）从 ObjC 头
    `MGLPipelineCache.h` 搬进新的 C 安全头 **`mgl_pipeline_cache_state.h`**（两个 `BOOL`→`uint8_t`），cache 仍以 ivar `_state` 持有，
    新端口 `mglRendererPipelineCacheStatePort()` 交出 `const MGLPipelineCacheState *` → 退掉
    `mglRendererPipelineStatePort` / `mglRendererPipelineProgramNamePort`（净 −1）。
    ② `_currentCBHasWork`（renderer ivar）并入 `MGLBatchingState.currentCommandBufferHasWork` → 退掉
    `mglRendererSetCurrentCBHasWorkPort`（净 −1，无新端口）；ObjC 侧 15 处写点（`+RenderPass` 7、`+Tessellation` 4、
    `+Blit` 2、`+Compute` 1、`mgl_draw_metal_port` 1）改名为 `_batching.currentCommandBufferHasWork`，ivar 删除。
    ③ **净减账**：**shim 35 → 33 个端口、459 → 445 行、73 → 71 语法**；全仓语法 **2,181 → 2,179**、
    行数 **38,039 → 38,025**、词汇 **4,139 → 4,137**（`_currentCBHasWork`/`pipelineXxx` 的 NS 词汇减少）。
    ④ **oracle**：trace 语料 374/296 行、`MGL_MIP_DIAG` 语料 529/212 条与旧库逐字段一致（A/B 三坑固定动作已执行：
    清 `.d` → 构建 → `cmp` 两库 → 只取本轮日志）；回归 92/0/2、ICB 轮 82/10/2 与旧库相同；CTS 七簇 diff 全空。
    ⑤ **规律沉淀**：**"ObjC 头里只有 C 字段的状态结构 = 一个端口换 N 个字段端口"** 是本层最稳的净减方式。
    已按此搬走三块：batching（−5）、command state（−4）、pipeline cache + `_currentCBHasWork`（−2）。
    剩余候选：`MGLRendererCoreState`（`_core`：activeState/capability/drawable/原子标志；需先把 `MGLCapability`/`MGLDrawable`
    做成 C 类型，可退 `mglRendererSetActiveStatePort` 等）、`MGLResourceFallbackState`（trace bindings，可退
    `mglRendererFragmentTraceBindingsPort`）、`MGLTessellationState`（`+Tessellation` 厚块的 materialize 下沉时会用到）。
    下一刀建议：**T5 唯一平台壳合并**（把 shim 的 4 个 category 方法 + `MGLPlatformRendererShell.m` + `+Lifecycle.m`
    收敛为一个壳 TU，shim 只留端口；这一步同时满足审计对 T5 的"唯一壳"要求）。

53. **shim 净减第四刀：状态区"一个 struct 一个端口"（本轮第十刀，**按 §0.04 端口净减、行数持平**）**：
    ① 新增 **`MGLRendererStateAreas`**（`mgl_renderer_ports.h`，C 侧自带存储、无隐藏 static）：
    `{ batching, command(const), pipeline_cache, binding_state_owner(地址), fragment_trace_bindings }` +
    一个端口 `mglRendererStateAreasPort(renderer, &areas)`。**退掉 3 个字段端口**：
    `mglRendererFragmentTraceBindingsPort`（→ `areas.fragment_trace_bindings`）、
    `mglRendererBindingStateOwnerPort`（→ `*areas.binding_state_owner`，取地址是因为 owner 可能在 driver 运行中变化）、
    `mglRendererPipelineCacheStatePort`（→ `areas.pipeline_cache->…`）。
    ② **净减账（如实）**：**shim 33 → 31 个端口（−2）、71 → 69 语法（−2）**，但**行数 445 → 446（+1）**——
    新端口带 `memset` + 5 个字段赋值，比它替代的 3 个两三行包装略长。**按 §0.04 的"端口/wrapper 净减"达标**，
    行数这一项持平，故不宣称行数收益。全仓语法 **2,179 → 2,177**、行数 **38,025 → 38,026**。
    ③ **oracle**：trace 374/296 行、`MGL_MIP_DIAG` 529/212 条与旧库逐字段一致（A/B 固定动作全执行）；
    回归 92/0/2、ICB 轮 82/10/2 与旧库相同；CTS 七簇 diff 全空。
    ④ **state-area 四刀小结**：batching（−5）、command（−4）、pipeline cache + 当前 CB 标志（−3）、
    本次 areas 归并（−2 端口）——**shim 43 → 31**。后续可继续并入 areas 的候选：`MGLRendererCoreState`（需先把
    `MGLCapability`/`MGLDrawable` 做成 C 类型）、`MGLResourceFallbackState` 其余字段、`MGLTessellationState`。

### 0.09 goal 交接状态（2026-09-13 19:00 更新，**第 52 轮**，P0-1 第三十五刀后）

- **tip（提交 `9dcbaee`，第 60 轮＝本轮次预算上限）**：`objc_zero.sh`：**16** 个 `.m` / 空 TU **0** / **34,396** 行 /
  语法 **1,957** / 词汇 **3,836**；
  shim **13 端口 / 223 行 / 37 语法**（声明面＝实现面，无死声明）。
  基线对照：文件 53 → 16、行数 43,989 → 34,564（−21.4%）、语法 2,268 → 1,973（−13.0%）、
  词汇 4,353 → 3,836（−11.9%）、shim 43 → 13 端口（−70%）。
- **第 40 轮的一次失败尝试（已回退，必须记录）**：`resetMetalState` 的 C 化在补 areas 槽地址时编译失败——
  **`_commandQueue` 在壳 TU 里不可见**（`_core`/`_backend`/`_batcing`/`_pipelineCache`/`_gpuRecovery`/`_tessellation` 等都可见，
  唯独它不可见），clang 报 `expected identifier` 并连锁报 `_backend` 未声明。**已 `git checkout -- .` 全部回退，未进提交。**
  → **结论：`resetMetalState` 不能靠 areas 槽地址读 `_commandQueue`，应改由 `MGLRenderer.m` 提供一个 C 入口
  （例如 `void *mglRendererCurrentCommandQueue(void *renderer)` / `int mglRendererRecreateCommandQueue(void *renderer)`），
  或把该 ivar 的可见性补齐后再走槽地址路线。**
- **轮次上限已到（24/24）**：本轮没有再开新刀，改为把交接做扎实——下面按"下一步能直接开工"的粒度列清楚。
- **剩余 13 个端口**（都还需要 ObjC）：`StateAreas`（核心）/ `EnsureWritableCommandBuffer`（轮转命令缓冲）/
  `FlushDrawBufferLocked`（port 里是 `@try/@catch` + METAL_LOCK）/ `BindMTLTexture`（**锁已成纸面理由**，见第 65 条）/
  `ProcessGLState`（锁定体是几百行 ObjC）、`MapBuffersToMTL`、`Bind{,Vertex,Fragment,Texture}sToCurrentRenderEncoder`、
  `RestoreRenderEncoderAfterTextureUpload`、`CurrentRenderPassMatchesFramebuffer`、`PrepareRenderPassIfFBOChanged`、
  `CreateIndirectCommandBuffer`（真 `@try/@catch`）。
- **纸面理由清单（下几刀优先验证）**：`METAL_LOCK()`＝`MGL_ASSERT_GL_THREAD()`、`METAL_UNLOCK()`＝空操作，
  因此凡是"因为要持锁"而留在 ObjC 的端口，都要重新评估 C 化可行性。
- **Batch 簇已清零**：`mgl_batch_*` 全部为 C；`MGLRenderer+Batch.m`、`mgl_batch_flush_restore_encode.m` 整文件删除。
- **A/B 设施（每刀复用）**：`git worktree add --detach /Users/fterward/MGL-ab-old HEAD` 得真旧库（借 `config.mk` +
  `build/aux` + `external/glfw/build`，**不要 `make clean`**），`cmp` 两库不同后跑
  `/private/tmp/run_ab7b.sh <side>`（default 与 `MGL_BATCH_MAX_DRAWS=1` 两臂，均由 `test_regression` 产生）
  + `/private/tmp/ab_full.py`；**只有加载 `libmgl` 的二进制才会写 trace 日志**（`test_batch_*` 系列不链接 libmgl）。
- **下一步优先级（按审计 §0.05 与 §0.08）**：
  1. **清"一行转发 C"的壳**（第九/十刀手法，低风险高产出）：`+DrawSupport.m` 余下的
     `currentDrawRasterizationIsEmpty` / `applyPolygonOffsetForDrawMode` / `ensureRasterEncoderForDraw` /
     `resolveIndirectBufferForDraw` / `prepareEmulatedIndirectCPURead`（体 1–3 行），以及
     `+GPURecovery.m`(351) · `+SwapDiagnostics.m`(557) 里的同类壳；
  2. **P0-1 三厚块**（`+RenderPass` 7.1k 行 · `+Texture` 6.5k · `+Blit` 4.1k）：抽 C++ 域 + 金样，**禁新增 ObjC 行**，
     每下沉一个方法体可连带退役对应端口（净减主力）；`+Blit.m` 建议从 91 行的
     `blitFramebufferDirectColorCopyWithState` 之后的 scaled/format-conversion 簇继续；
  2. 与 GLSampled 同域的 `+Texture.m` 的 `freshGLSampledRenderTargetCopyForSampling:`（**有真实调用点**，
     需连同采样视图窗口逻辑一起下沉）；
  3. **T5 唯一平台壳合并**：shim 仅剩的 1 个 ObjC 方法（`flushDrawBuffer:` 锁壳）+ `MGLPlatformRendererShell.m`(230) +
     `+Lifecycle.m`(666) → **一个**壳 TU；
  4. **ICB 门禁 82/10/2 → 92/0/2**（Keep-A/B-temporary 的退休条件；10 项失败为
     `Fragment/Vertex shader cannot be used with indirect command buffer` + AGX RECOVERY 日志）；
  5. **P0-0** `mgl_air_backend.cpp` 的 24 处 `strstr(esrc,…)` 双轨（LLVM 路径最高优先，oracle-equal 后删文本侧）；
  6. 继续 state-area 净减（`_resourceFallback` 其余字段 / `_tessellation`）。
- **纪律提醒**：每刀必须给 **shim wrapper 净减数字**（§0.04），并跑满 §0.2 的三套语料 + 该刀自己的 oracle；
  A/B 固定动作见第 51 条（清 `.d` → 构建 → `cmp` 两库 → 只取本轮日志）。

54. **shim 净减第五刀：`MGLRendererCoreState` 与 dual-proxy 不变式转 C（**按 §0.04 达标**）**：
    ① `MGLRendererCoreState`（`activeState` / capability 快照 / `drawBuffers[]` / `defaultDrawableWrittenSinceLastSwap` /
    四个 `_Atomic` 交接通道）连同 `MGLDrawable` 与 `_FRONT.._MAX_DRAW_BUFFERS` 枚举从 ObjC 头
    `MGLRenderer_State.h` 搬进新的 C 安全头 **`mgl_renderer_core_state.h`**（`BOOL`→`uint8_t`）。
    ② **dual-proxy 不变式真正落到 C**：新 TU `mgl_renderer_core_state.c` 实现
    `mglCoreActivateReplayState` / `mglCoreRestoreLiveActiveState` / `mglCoreAssertDualProxy`
    （等价于原 `-mglActivateReplayStateForContext:` / `-mglRestoreLiveActiveStateForContext:` /
    `-mglAssertDualProxyInSyncForContext:`：两次指针写 + replay 工作区 memcpy；断言语义保持"desync 即 abort"）。
    ③ **净减账**：退掉 4 个端口（AssertDualProxy / ActivateReplayState / RestoreLiveActiveState / SetActiveState），
    `MGLRendererStateAreas` 增加 `core` 字段（**不新增端口**）；**shim 内 ObjC 方法从 5 个减到 1 个**
    （只剩 `flushDrawBuffer:` 锁壳）。度量：**shim 31 → 27 端口、446 → 394 行、69 → 62 语法**；
    全仓行数 **38,026 → 37,974**、语法 **2,177 → 2,170**、词汇 **4,137 → 4,136**。
    ④ **oracle**：trace 374/296 行、`MGL_MIP_DIAG` 529/212 条与旧库逐字段一致；回归 92/0/2、ICB 轮 82/10/2 相同；
    CTS 七簇 diff 全空。dual-proxy 断言在每次 flush 上都跑，覆盖充分。
    ⑤ **环境事故（已恢复，必须记录）**：为清 stale `.d` 我执行了 `make clean` —— 本机 **Metal toolchain 组件缺失**
    （`cannot execute tool 'metal' due to missing Metal Toolchain`），`build/aux/*.metallib` 被删后无法重建，整个库构建中断。
    **恢复办法**：用 `scripts/gen_aux_assets.py` 的**逆操作**从提交版 `MGL/src/mgl_aux_assets.c` 里把 7 个 metallib
    逐字节还原到 `build/aux/`，并放置时间戳正确的空 `.air` 占位（避免触发 `metal`）；随后重跑生成器，**产出的
    `mgl_aux_assets.c/h` 与提交版逐字节一致**（`git status` 干净）——这同时反证还原是精确的。
    `build/aux/README-restored.txt` 写明占位风险：**改 `*.metal`/MANIFEST 前必须先删 `build/aux`**，否则会用到旧 metallib。
    结论：**本机不要跑 `make clean`**（要清依赖用 `find build -name '*.d' -delete`，这也是 A/B 固定动作）。
    剩余 shim 端口路线：`_resourceFallback` 其余字段、`_tessellation`（P0-1 下沉时并入）、`_bindingStateOwner`
    已在 areas 内、`mglRendererCommandStatePort`/`BatchingStatePort` 可并入 areas（但调用点多，收益 −2）。

### 0.10 三厚块（`+RenderPass`/`+Texture`/`+Blit`）转换评估（2026-09-13 实测）

**问题**：它们能否整文件删除、改在 AIR 前后端实现？

**实测成分**（脚本口径：方法体按 `^- (`/`^+ (` 到行首 `}` 计；"含 ObjC 发送的代码行"= 去掉注释/空行后含 `[recv sel]` 的行）：

| 文件 | 行数 | ObjC 方法 | 方法体行数 | 文件内 C 风格函数 | 含 ObjC 发送的代码行 | 调 `mglRender*` C facade | 调 C 计划模块 | `__bridge` |
|---|---|---|---|---|---|---|---|---|
| `+RenderPass.m` | 7,100 | 51 | 6,181（87%） | 45 个 / 643 行 | 290 = **4.8%** | 665 次 / **193 个入口** | 24 次 / 10 个 | 86 |
| `+Texture.m` | 6,981 | 53 | 6,392（92%） | 33 个 / 406 行 | 168 = **3.0%** | 307 次 / **150 个入口** | 19 次 / 12 个 | 92 |
| `+Blit.m` | 4,941 | 30 | 4,302（87%） | 36 个 / 458 行 | 115 = **2.7%** | 175 次 / **80 个入口** | 21 次 / 11 个 | 94 |

**结论一：可以最终删除，但做法是"转换"而不是"删除后重写"。**
87–92% 的行在方法体里，而方法体里只有 3–5% 的代码行含 ObjC 消息发送——其余是 GL 状态读写、格式/区域数学、
以及对 **C facade（`mgl_render.cpp`，metal-cpp）** 和 **C 计划模块** 的调用。这三个文件本质是
"**C 代码套了 `.m` 外壳**"。P0-1 的"禁止整文件 Delete"正是指不能跳过转换直接删。

**结论二：不应该放进 AIR 前后端。**
AIR 层是 shader 路径（`mgl_air_backend.cpp` / `mgl_ir.c` / `mgl_glsl_{lexer,parser,sema}.c` /
`mgl_metallib_writer.cpp` / `mgl_air_loader.cpp`）；render-pass / texture / blit 是**运行时 GL→Metal** 语义，
其 Metal 物化已经在 `mgl_render.cpp` 里，并被这三文件以 193/150/80 个 C 入口调用。把它们塞进 AIR 会
(a) 与 `mgl_render.cpp` 重复实现，(b) 直接违反 P0-0「禁再胀 `mgl_air_backend.cpp`」。

**结论三：哪些"本来就已经实现"？**
- **已实现且已在用**：Metal 物化（`mgl_render.cpp`）；O3.1 的 render-pass load/store/clear/match 计划、
  O4.4 的 depth/stencil blit gate、pixel-format/region/readback 计划——但合计只 64 个调用点，
  占三文件方法体逻辑 **远不到 1%**（联合报告说的 "plan@C 贴皮" 在这里成立）。
- **没有在别处实现**：方法体里那 **~16.9k 行**域内编排（GL 校验、上传路径、格式转换决策、区域裁剪、
  blit 参数推导、render-pass 状态机）。删除它们必须把这些逻辑**原样搬走**。
- **纯外壳可去掉**：79 个外部 `[self …]` 依赖（115/68/69 次）+ 19 处 `NSString/NSError` + 272 处 `__bridge`；
  其中已有 C 端口的 8 个，以及本周期刚做成 C 可见的 `_renderPassManager.state` / `_pipelineCache.state`。

**关键点（决定本项与 P0-2 相容）**：**端口只在 C→ObjC 时才是必需的**。把方法转成 C 函数之后，
**ObjC 调用点可以直接调 C 函数**（传 `(__bridge void *)self`），因此这三文件的转换**不会让 shim 变大**，
反而随方法搬走而净减。

**分阶段路径（每刀都要给出 shim wrapper 净减）**
1. **A：零/极低依赖方法先行**——如 `+RenderPass.m` 的 `newRenderEncoder`(4 行)、`newRenderEncoderLocked`(4)、
   `endRenderEncoding`(6)、`framebufferAttachmentTexture`(6)、`bindMTLProgram`(7)、`ensureWritableCommandBuffer`(7)；
   `+Texture.m` 的 `textureIndexForExpectedMetalType`(4)、`swizzleTexDesc`(8)、`textureUnitForSampledBinding`(4)；
   `+Blit.m` 的 `releaseGLSampledRenderTargetCopyForTexture`(17)、`clearRectDepthState`(24)、
   `scaledBlitPipelineForPixelFormat`(29)。注意这些方法全仓被调用 25/35/23 次不等（跨 4–8 个文件），
   转 C 后要同步改这些 ObjC 调用点为直调。
2. **B：高频外部依赖做成 C 入口并删掉原 ObjC 方法**——`recordGPUError`(30 次/3 文件)、`mglDrawableTexture`(26/4)、
   `bindMTLTexture`(42/10)、`resetMetalState`(8/3)、`newCommandBuffer`(15/5)、`flushCommandBuffer`(11/5)、
   `mglNextDrawable`(8/4)、`getOptimalAlignmentForPixelFormat`(6/1)。
3. **C：`NSString/NSError` → C 字符串**（19 处）。
4. **D：文件清空 → 删除**（此时才允许整文件删）。

**工作量估计**：134 个方法 / ~16.9k 行 / 79 个外部依赖（去重后约 40 个）。按当前"每刀 300–500 行 + 端口面"的节奏，
约 **8–12 个切片**；前 3–4 刀应专挑"零依赖 + 调用点集中"的方法，保证每刀都能报出 shim 净减。

55. **P0-1 首刀：`+Blit.m` 的 pipeline/sampler/depth-state 缓存簇转 C（**shim 端口零增长**）**：
    ① **搬走的内容**：7 个方法（`scaledBlitPipelineForPixelFormat` / `scaledDepthBlitPipelineForPixelFormat` /
    `scaledBlitComputePipelineForPixelFormat` / `msaaIntegerResolvePipelineForSigned` / `clearRectPipelineForColorFormat:…` /
    `clearRectDepthState` / `scaledBlitSamplerForFilter`）+ 6 个文件内 asset helper
    （`mglLookupAux{Render,Compute}Pipeline` / `mglCreateAux{Render,Compute}PipelineFromAsset` /
    `mglBlitCreate{Sampler,DepthStencilState}`）→ 新 TU **`MGL/src/mgl_blit_pipelines.c`** + C 安全头
    **`MGL/include/mgl_blit_pipelines.h`**（7 个入口，均返回**借用**引用，所有权归 renderer 生命周期的
    C++ aux pipeline 缓存与 backend blit 缓存）。
    ② **顺带清掉的 ObjC 词汇**：`NSError **error` → `char errbuf[512]`、`NSString` 描述串 → C 字符串、
    `NSLog` → `fprintf(stderr, …)`、`id` → `void *`、`__bridge_transfer`/`__bridge id` 全部消失。
    ③ **`MGLRendererStateAreas` 扩两个字段**（`void *backend` + `GLMContext ctx`）即够用，**没有新增端口**——
    这正是 §0.10 的结论：**端口只在 C→ObjC 时需要**；方法转 C 后 ObjC 调用点直接调 C 函数（13 处改直调）。
    ④ **度量**：`+Blit.m` **4,942 → 4,584 行（−358）**、语法 244 → **236**、词汇 880 → **821（−59）**；
    全仓行数 **37,974 → 37,617**、语法 **2,170 → 2,167**、词汇 **4,136 → 4,077**；**shim 端口 27 不变**（行数 +2 = areas 两个字段赋值）。
    ⑤ **oracle**：trace 语料 374/374、296/296 **逐字段一致**；stderr 语料 824/824、622/622 行，仅 3 类差异且均已确认无害：
    (a) BINARY ARCHIVE created/loaded 取决于 archive 文件状态的运行序伪差；(b) 失败诊断文本由 NSError 描述
    （`Error Domain=MGLBlitPipeline Code=2 "…"`）改为 C++ 原始 message（更直接）；(c) `mglDispatchError` 的函数标签由
    ObjC selector `-[MGLRenderer(Blit) scaledBlitPipelineForPixelFormat:]` 改为 C 函数名。回归 92/0/2、ICB 轮 82/10/2 与旧库相同；
    CTS 七簇 diff 全空。
    ⑥ **环境/流程事故（两条，都影响证据可信度，必须改规则）**：
    - `make clean` → **本机 Metal toolchain 组件缺失**，`build/aux/*.metallib` 无法重建（见第 54 条），已按资产表逆推还原并逐字节验证；
    - **只删 `build/**/*.d` 会关闭头文件依赖重建**：本刀给 `MGLRendererStateAreas` 加字段后，部分 `.o` 未重编，
      出现"指针像被踩坏"的 SIGSEGV（`test_dirty_hash` 崩在 `+Blit.m:1666` 的 `glm_ctx->active_state->readbuffer`，ctx 是垃圾值），
      一度被误判为"旧库也崩"。**修正后的 A/B/重建固定动作**：
      **① 切树或改头文件后，`find build/core build/es -name '*.o' -o -name '*.d' | xargs rm -f` 全量重编；
      ② 每次重建后重新 `cp` 库到 A/B 目录并 `cmp` 确认两库不同；③ 只取本轮日志；④ 网络不可用时
      `verify-gl-api` 的 `git fetch` 会失败 → 改为直接跑 `python3 scripts/verify_gl_api.py`（离线校验，registry 已在 `external/`）。**
    下一刀（同法，仍要求 shim 零增长或净减）：`+Blit.m` 的 `scaledBlitPipelineForPixelFormat` 之外的
    compute/clear 路径调用方（`MGLRenderer.m` 的 scissored clear、`+SwapDiagnostics.m` 的 drawable 缩放）已在本刀改直调；
    接着按 §0.10 阶段 A 推进 `+Blit.m` 的 `releaseGLSampledRenderTargetCopyForTexture`、`+RenderPass.m` 的
    `newRenderEncoder`/`endRenderEncoding`/`ensureWritableCommandBuffer` 等零依赖方法。

56. **P0-1 第二刀：`+Texture.m` 的 sampler-unit 决策与三个叶子方法转 C（**shim 净减 1 个端口**）**：
    ① **决策下沉（T3）**：83 行的 `-textureUnitForSampledResource:program:metalBinding:stage:`（sampler 单元解析：
    explicit/reflected/binding 级与 stage 级优先级、`Sampler0/Sampler2` 那类 Minecraft 特例）**整段搬进 C**，
    `mgl_texture_compat.h/.c` 新增 `mglTextureUnitForSampledResource(res, program, metal_binding, stage)`；
    **顺手退掉 shim 端口** `mglRendererTextureUnitForSampledResourcePort`（C 侧 `mgl_batch_dyn_bind_encode.c`
    改为直调，program 由 `mglResolveProgramForStageFromState` 解析，与该端口原语义一致）→ **shim 27 → 26 端口**。
    ② **叶子方法删除**：`textureUnitForSampledResource:metalBinding:stage:`（3 行转发）、
    `textureUnitForSampledBinding:stage:`（**0 调用者的死代码**）、`swizzleTexDesc:forTex:`（1 调用点 → C `mglTextureSwizzleDescriptor`）、
    `textureIndexForExpectedMetalType:`（1 调用点 → 直接用 `mglRenderTextureIndexForMetalType`）、
    `releaseGLSampledRenderTargetCopyForTexture:`（4 调用点 → C `mglTextureReleaseGLSampledCopy`）。
    合计 **删 5 个方法 + 1 个 shim 端口**，14 处调用点改直调（`+Texture` / `+Binding` / `+BindingState` / `+Compute` / `+Blit` / `+RenderPass`）。
    ③ **度量**：`+Texture.m` **6,981 → 6,868 行**、`+Blit.m` 4,584 → **4,566**；全仓行数 **37,617 → 37,459**、
    语法 **2,167 → 2,150（−17）**、词汇不变（4,077）；shim 端口 **27 → 26**、行数 396 → **382**。
    ④ **oracle**：trace 语料 374/374、296/296 逐字段一致；stderr 语料 824/824、622/622 行，差异仅剩
    BINARY ARCHIVE created/loaded/saved 的运行序伪差（上一刀的诊断文本改动两臂已一致）；回归 92/0/2、ICB 轮 82/10/2 相同；
    CTS 七簇 diff 全空。
    下一刀：继续 §0.10 阶段 A/B——`+RenderPass.m` 的 `newRenderEncoder`/`newRenderEncoderLocked`（转发到
    `...WithReason:MGL_ENC_REASON_OTHER`，13+8 个调用点可内联）、`endRenderEncoding` 需**保留**（它带 METAL_LOCK，
    50 个调用点不能机械内联）、`+Blit.m` 的 `clearRectDepthState` 已随首刀搬走；
    更值钱的是把 `textureForSampledResource:metalBinding:stage:expectedType:`（+Texture 的采样取纹理环）与
    `+RenderPass.m` 的 `newRenderEncoder(Locked)WithReason:` 一并下沉。

57. **P0-1 第三刀：`textureForSampledResource` 采样取纹理环下沉（**shim 再净减 1 个端口**）**：
    ① **搬走 189 行纯 C 决策**：`-textureForSampledResource:metalBinding:stage:expectedType:textureUnit:`（174 行：1D/buffer/MS/typed-slot
    优先级、默认纹理拒绝规则、texel-buffer 缺失即拒绝、两条限流诊断）与它的 15 行 `:expectedType:` 包装
    → 新 TU **`MGL/src/mgl_texture_binding_resolve.c`** + C 安全头 **`mgl_texture_binding_resolve.h`**
    （`mglTextureForSampledResource(ctx, res, binding, stage, expected_type, unit)` /
    `mglTextureForSampledResourceForStage(...)`）。**单独开 TU 的原因**：`mgl_binding_texture.c` 被
    `test_binding_texture` 单独链接，不能拉进 `mgl_render.*` 依赖（老教训）。
    ② **退掉 shim 端口** `mglRendererTextureForSampledResourcePort`（C 侧 `mgl_batch_dyn_bind_encode.c` 直调
    `mglTextureForSampledResourceForStage`）；ObjC 调用点 5 处改直调（`+BindingState` 2、`+Compute` 2、`+Texture` 1）。
    ③ 两处 `NSLog` 诊断 → `fprintf(stderr, …)`（同 sink，字段不变）；`MGL_STATE(ctx)` → `ctx->active_state`（dual-proxy 不变式）。
    ④ **度量**：全仓行数 **37,459 → 37,241**、语法 **2,150 → 2,142**、词汇 4,077 → **4,075**；
    **shim 26 → 25 端口 / 382 → 367 行**；`+Texture.m` 6,868 → **6,679**。
    ⑤ **oracle**：trace 语料（本轮 default 臂额外开 `MGL_TRACE_LOG_RESOURCES=1`，覆盖纹理绑定路径）**374/374、296/296 逐字段一致**；
    stderr 仅 BINARY ARCHIVE created/loaded/saved 的运行序伪差；回归 92/0/2、ICB 轮 82/10/2 相同；CTS 七簇 diff 全空。
    ⑥ **推送方式变更（用户指示）**：**不再用 HTTPS**，改用已有的 SSH remote `origin`
    （`git@github.com:53453450/MGL-minecraft.git`，`ssh -T git@github.com` 已验证），推送命令固定为
    `git push origin main:main`。此前 HTTPS 因网络中断积压的 3 个提交已一次性推送成功。
    下一刀：`+RenderPass.m` 的 `newRenderEncoder(Locked)WithReason:` 环与 `endRenderEncoding` 系列
    （`endRenderEncoding` 带 METAL_LOCK，**保留**）；以及 `+Blit.m` 的 `clearRectPipelineForColorFormat:` 调用链。

58. **P0-1 第四刀：sampler 物化转 C（**shim 再净减 1 个端口**）**：
    ① **搬走**：`-createMTLSamplerForTexParam:target:`（14 行，4 处调用点）、`-fallbackSamplerState`（21 行）、
    `id` 版本的文件内 helper `mglTextureCreateSampler`、`-bytesPerPixelForFormat:`（10 行，4 处调用点）、
    以及**0 调用者的死方法** `textureForSampledBinding:stage:expectedType:` →
    新 TU **`MGL/src/mgl_texture_sampler.c`** + C 安全头（`mglTextureCreateSamplerForTexParam`（+1）、
    `mglTextureFallbackSamplerState`（借用，backend fallback 缓存持有））+ `mgl_pixel_format.{h,c}` 的
    `mglTextureBytesPerPixelForFormat`。
    ② **退掉 shim 端口** `mglRendererFallbackSamplerStatePort`（C 侧 `mgl_batch_dyn_bind_encode.c` 直调）；
    shim 的 snapshot-sampler 端口改为调用 C 的 sampler 创建（+1 交给 backend 快照缓存后释放），
    因此 shim 端口 **25 → 24**、行数 367 → 366。
    ③ **踩坑与教训（重要）**：我先删掉了 `fallbackSamplerState` 方法，但**漏了一处 ObjC 调用点**
    （`+BindingState.m` 的 `bindTexturesToCurrentRenderEncoder:` 路径用 `[self fallbackSamplerState]`），
    结果 `test-all` 的 `test-dirty-hash` 直接抛 `unrecognized selector` 崩溃（`Abort trap: 6`）。
    **新规则：删/转一个方法后，必须用"裸 selector 名"全树 grep（含 `self.x` 属性写法与 `[other x]` 接收者），
    不能只 grep `[self X` 或端口名。** 本次即由聚合门禁在同一轮内抓到，未进提交。
    ④ **度量**：`+Texture.m` 6,679 → **6,614**；全仓行数 **37,241 → 37,177**、语法 **2,142 → 2,129**、词汇 4,075 → **4,066**。
    ⑤ **oracle**：trace 语料 374/374、296/296 逐字段一致（default 臂含 `MGL_TRACE_LOG_RESOURCES=1`）；
    stderr 仅 BINARY ARCHIVE 运行序伪差；回归 92/0/2、ICB 轮 82/10/2 相同；CTS 七簇 diff 全空；
    P0-1 四刀累计 **shim 27 → 24 端口**、全仓 **−801 行**。
    下一刀：`+Blit.m` 的 `textureCanUseGLSampledRenderTargetCopy`(37) / `lazyRefreshGLSampledRenderTargetCopyForTexture`(52) /
    `blitFramebufferDirectColorCopyWithState`(91) 等零外部依赖方法，以及 `+Texture.m` 的
    `mglApplyPending{FBO,Default}{Color,Depth}Clear*` 系列（21–30 行 ×4，`+Blit.m` 已在调用）。

59. **P0-1 第五刀：readback/blit 前置 clear 应用族转 C（**shim 持平、行数 −130**）**：
    ① **搬走**：`mglApplyPendingFBODepthClearForReadback:attachment:textureObj:mtlTexture:`（26 行）、
    `mglApplyPendingFBOColorClearForReadback:…attachmentEnum:`（30）、`mglApplyPendingDefaultDepthClearToTexture:`（18）、
    `mglApplyPendingDefaultColorClearToTexture:`（21）→ 新 TU **`MGL/src/mgl_texture_readback_clear.c`** + C 安全头
    （4 个入口；`_renderPassManager.state->currentCommandBufferOwner` 改为
    `mglRendererCommandStatePort(renderer)->currentCommandBufferOwner` ✓ 本周期已把 command state 做成 C 可见，
    `STATE(...)` → `areas.ctx->active_state->…` ✓ dual-proxy 不变式）。12 处调用点（`+Texture` 6、`+Blit` 6）改直调。
    ② **如实记账（shim 持平，不到"净减"）**：本轮**先多开了一个端口** `mglRendererStateAreasCtxPort`，
    发现 `mglRendererStateAreasPort` 已带 `ctx` 字段后**立即回退**（多一个端口＝违反 §0.04），
    最终 **shim 24 端口持平**、行数 366（+0）；ObjC 行数 **37,177 → 37,047（−130）**、
    词汇 4,066 → **4,062**，但**语法 2,129 → 2,137（+8）**——12 处调用点各引入一个 `(__bridge void *)` 桥接，
    这是"方法转 C 而调用点仍是 ObjC"的固有代价（第 46 条同款现象），如实记录、不宣称语法收益。
    ③ **oracle**：trace 语料 374/374、296/296 逐字段一致（default 臂含 `MGL_TRACE_LOG_RESOURCES=1`）；
    stderr 仅 BINARY ARCHIVE 运行序伪差；回归 92/0/2、ICB 轮 82/10/2 相同；CTS 七簇 diff 全空。
    ④ **教训（第三次同源）**：`mglMarkTextureLevelRenderTargetWritten` 是 ObjC 头里的**宏**（4 参 Impl + `__func__`/`__LINE__`），
    C 侧必须声明并使用 `…Impl(tex, level, __func__, __LINE__)`；先按 2 参声明会与 `mgl_batch_rt_mark_host.c` 的
    `extern` 冲突。**规则：从 ObjC 头借符号给 C 用时，先确认它是宏还是函数。**
    下一刀（目标重新回到净减）：`+Blit.m` 的 GLSampled-copy 簇——`textureCanUseGLSampledRenderTargetCopy:source:`(37) +
    `lazyRefreshGLSampledRenderTargetCopyForTexture:…`(52) + **`updateGLSampledRenderTargetCopyForTexture:…`(375)**，
    后者只依赖 `ensureWritableCommandBuffer` 与前者，整簇约 **464 行**，是迄今最大单块；
    若把它转 C，可顺带退役/收窄若干 blit 端口。

60. **P0-1 第六刀：sampled-copy 谓词转 C + 退役两个冗余端口（**shim 净减 1**）**：
    ① **转 C**：`-textureCanUseGLSampledRenderTargetCopy:source:`（37 行，唯一的自制引用是
    `mglTextureCanUseGLSampledRenderTargetCopy`（`mgl_rt_sync.h` 的 C inline）与 `mglBlitTextureInfo`（`+Blit.m` 的
    file-static 3 行包装））→ 新 TU **`MGL/src/mgl_blit_sampled_copy.c`** + C 安全头；C 侧自带 3 行等价 helper
    （`mglBlitSampledCopyTextureInfo`，注明与 `+Blit.m` 的同源关系），调用点 2 处（`+Blit` 1、`+RenderPass` 1）改直调。
    ② **退役两个"已被 state areas 覆盖"的端口**（本轮真正的净减来源）：
    `mglRendererPipelineCacheStatePort`（areas 已带 `pipeline_cache` → `mgl_batch_replay_trace.c` 改用 `areas.pipeline_cache`）、
    `mglRendererBindingStateIsValidPort`（`mgl_batch_flush_restore_encode.c` 自带 `mglBatchBindingStateIsValid(owner)`
    本地检查 + 从 areas 取 owner 地址）。**shim 24 → 23 端口 / 366 → 360 行**；全仓行数 **37,047 → 37,004**、
    语法 **2,137 → 2,136**、词汇 4,062 → **4,056**。
    ③ **事故（本轮自己造的，门禁抓住）**：抽取三个方法时把 `lazyRefreshGLSampledRenderTargetCopyForTexture`(52) 与
    **`updateGLSampledRenderTargetCopyForTexture`(375)** 一并从 `+Blit.m` 删掉、却只补回了 37 行的谓词，
    于是 `test-dirty-hash` 立刻 `Abort trap: 6`（缺实现）。**已从抽取副本逐字回插**并把其中对已转 C 谓词的调用改为直调。
    **新规则：批量抽取=先"抽取即落盘副本"，转换与回插必须在同一脚本内完成；门禁必须在抽取后立刻跑一次。**
    ④ **oracle**：本刀**未跑 trace A/B**（纯代码搬移 + 端口退役，风险面小）——如实记录；跑的是
    **门禁全过**（`verify_gl_api` 离线 + 28 个 `test-all` 目标，`test_regression` 92/0/2、es-smoke ok）+
    **CTS 七簇非通过集合 diff 全空**（hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 ·
    piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
    下一刀：把 **`lazyRefresh…`(52) + `updateGLSampledRenderTargetCopyForTexture`(375)** 真正转 C——后者
    375 行里只有 **2 处消息发送**（`ensureWritableCommandBuffer:reason:` 与本次已转的谓词）与 **79 处 `mgl*` C 调用**，
    因此只需为 `ensureWritableCommandBuffer:` 开**一个**端口（再用"退役一个冗余端口"抵消），
    另有 1 处 `@autoreleasepool`（C 侧去掉，临时对象随调用方池释放）与 `mglBlitCreateRenderEncoder(_renderPassManager,…)`（需 C 化签名）。

61. **P0-1 第七刀：flush 驱动改走 state areas + 退役 batching/trace 两端口（**shim 净减 2，零新增包装**）**：
     ① **做法（"areas 反向用法"）**：`MGLRendererStateAreas` 已带 `batching` / `command` 字段，于是
     `mgl_batch_flush_restore_encode.c` 里 6 处 `mglRendererBatchingStatePort(c->r)` 读点改为先取
     `mglRendererStateAreasPort(...)` 再读 `areas.batching`；flush 帧写 trace-replay 身份的
     `mglRendererTraceReplaySetPort(hit, bi)` 改为 `areas.command->traceReplayFlushId / …BatchIndex = …`。
     **等价性先核对再改**：旧端口实现就是 `[manager setTraceReplayFlushId:batchIndex:]`，而该 setter 写的是
     `_state`（`mglRendererCommandStatePort` / `areas.command` 指向的同一记录）；`mglRendererBatchingStatePort`
     就是 `&r->_batching`。为此把 `MGLRendererStateAreas.command` 的 `const` 去掉——shim 内**一次**显式 cast，并在原地注明
     "记录本身可变、flush 驱动要写 trace 身份"，不新增任何包装函数。
     ② **死声明清理（同为记账面净减）**：`mgl_renderer_ports.h` 里 `mglRendererTryReplaySimpleBatchPort` /
     `mglRendererApplyDynamicBindingsPort` / `mglRendererApplySamplerSnapshotPort` 三个**既无实现、也无调用点**的
     声明（dyn_bind 一刀时三个方法已被共享端口取代）一并删除——留着是链接期地雷，也虚增端口面。
     ③ **净减账**：端口**声明 24 → 21**（21 个端口全部有实现、有调用点）、**shim 实现 23 → 21**、
     **shim 行数 360 → 350**、**shim 语法 62 → 49**；**本刀新增包装 0 个**（满足 §0.04「无 shim 净减＝拒收」）。
     全仓 ObjC 行数 **37,004 → 36,994**、语法 **2,136 → 2,134**、词汇 **4,056（持平）**；
     `MGLRenderer*.m` 合计 **33,451**（**勘误**：§0.0 表里此格此前的 34,387 是第五刀时的数，第 1、2、3 刀删掉
     `MGLRenderer+Batch.m` 等之后一直没刷新，本轮按 `scripts/objc_renderer_loc.sh` 实测校准；HEAD 与工作树同为 33,451）。
     ④ **事故与规则**：首编报三处 `error: redefinition of 'areas'`——替换时函数体内**已有**同名局部
     （`mglBatchFlushBegin` / `mglBatchRestoreStateForBatch` / `mglBatchTeardownReplay`）。**规则：把端口调用替换成
     `MGLRendererStateAreas areas;` 前，先看该函数是否已有这个局部，合并复用而不是新插。**
     ⑤ **oracle（本刀必须做，因为直改 flush/batching 与 trace 身份）**：
     - **旧库是真 HEAD**：`git worktree add --detach /Users/fterward/MGL-ab-old HEAD`（`b295c50`），从工作树借
       `config.mk`、`build/aux`（**不能 `make clean`**，本机缺 Metal toolchain）、`external/glfw/build`，独立构建；
       `cmp` 两库字节不同后才跑（`char 145` 起不同）。
     - **语料两臂**：`test_regression all`（default）与 **`MGL_BATCH_MAX_DRAWS=1` 强制逐 draw flush**（flushy，
       真正穿透本刀改动的 flush/replay 路径）——**每臂都由 `test_regression` 产生**，因为只有它加载 `libmgl`；
       `test_batch_icb` **不链接 libmgl**（`otool -L` 无），**不可能**产出 trace 日志，
       故旧 `/tmp/ab_compare.py` 的 "icb 臂"在只有一份日志时会**拿同一份日志自比**（历史条目里的 icb 数字不可复现，如实记录）。
     - **比对口径**（脚本 `/private/tmp/ab_full.py`，**全量 trace**，不再只筛 `REPLAY_`）：剥 `[ts seq tid fid`、
       mask `0x…`、mask `built=`/`elapsed=`；trace 头行与 `BINARY ARCHIVE …` 行（pipeline 二进制归档是
       检出目录里的**文件**，先跑的一侧 create、后跑的一侧 load，外加 machO pack 随机失败）单列为运行序伪差。
     - **结果**：确定性行 **default 4,980/4,980、flushy 5,513/5,513 逐行保序完全一致**，其中 `REPLAY_*`/`IFACE DUMP`
       （含 `flush=`/`batch=` 身份字段）7/7 一致、`FBIND_GL` 274/274 多重集一致；**非** `processGLState.slow` 的行
       **没有任何一条只出现在一侧**（`other-only A=0 B=0`）。唯一差异是 `MGL TRACE processGLState.slow call=N`，
       且**同一份库连跑两次**同样差（default 86 行、flushy 3/7 行）→ 该行计数本身非确定（PSO 缓存/归档时序），
       与本刀无关，如实记录而不是"diff 全空"。
     - **stderr** `MGL` 行 307/307 多重集一致（仅剔除 2 条 `BINARY ARCHIVE` 运行序行）；**回归 plain 92/0/2、
       ICB 门禁 82/10/2，两侧（新库/旧库）完全相同**。
     ⑥ **门禁与 CTS**：`verify_gl_api` 离线通过 + 28 个目标全过（`GATE_EXIT=0`，`test_regression` **92/0/2**、
      `test_batch_icb: ok`、es-smoke ok）；**CTS 七簇非通过集合 diff 全空**
     （hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns，
     逐簇与 `/tmp/base_*.txt` 比对 `DIFF EMPTY`）。
     下一刀（回到第 60 条留下的整簇）：**`lazyRefreshGLSampledRenderTargetCopyForTexture`(52) +
     `updateGLSampledRenderTargetCopyForTexture`(375)** 转 C：只需为 `ensureWritableCommandBuffer:reason:` 开**一个**端口，
     并用"退役一个冗余端口"抵消（保持净减）；另有 1 处 `@autoreleasepool` 与 `mglBlitCreateRenderEncoder(_renderPassManager,…)` 需 C 化签名。
     之后按 §0.08 表继续：`CurrentRenderPassMatchesFramebuffer` / `PrepareRenderPassIfFBOChanged`（随 `+RenderPass` 厚块下沉）、
     `SamplerStateForSnapshotKey`（随 sampler 物化改 C++）、ICB 门禁 82/10/2 → 92/0/2、T5 唯一壳合并。

62. **P0-1 第八刀：GLSampled-copy 整簇转 C，顺带退掉两个纯数据端口（**shim 净减 1，21 → 20**）**：
     ① **转 C（净 −449 行 `MGLRenderer*.m`）**：`-[MGLRenderer updateGLSampledRenderTargetCopyForTexture:source:reason:]`
     （**375 行**，11 处调用点）整块搬进 **`MGL/src/mgl_blit_sampled_copy.c`**，成为
     `mglBlitUpdateGLSampledRenderTargetCopy(void *renderer, Texture *tex, void *source, const char *reason)`
     （谓词 `mglBlitTextureCanUseGLSampledRenderTargetCopy` 的同域文件）。**随后核对再改**：
     - 该方法里只有 **2 处**不是 C：`[self ensureWritableCommandBuffer:]`（要 `-newCommandBufferLocked` 轮转，
       保留为**唯一新增端口**）与旧的 `mglBlitCreateRenderEncoder(_renderPassManager,…)`（改为
       `mglRenderCreateRenderEncoderFromCommandBufferOwnerState(cs->currentCommandBufferOwner,…)`）。
     - 其余全部直调 C：`mglRenderCreateTextureFromState` / `mglRenderCreateTextureViewRange` /
       `mglRenderReleaseMetalObject` / `mglRenderSet*` / `mglRenderEncodeDraw` / `mglRenderEnd*Encoder` /
       `mglRenderCreateComputeEncoderBorrowed` / `mglBlitScaled{Pipeline,ComputePipeline,Sampler}For*`。
     - **所有权逐条核对**：`CFBridgingRetain(copy)` 的净效果是"ivar 持有 +1"，C 侧改为直接把
       `mglRenderCreateTextureFromState` 返回的 +1 转给 `tex->mtl_gl_sampled_data`（等价）；
       `@autoreleasepool` 里的两个 level view 是唯一的临时对象，改为显式 `mglRenderReleaseMetalObject`
       并在 3 个出口（view 失败 / encoder 失败 / 正常结束）都释放（新增 `mglBlitSampledCopyReleaseViews`）。
     - 着色器参数结构体保持**逐字节布局**：C 侧用同一套 vector 类型镜像
       （`vector_float4 uvRect; float forceOpaqueAlpha; vector_float3 _padding;` → sizeof 48、`_padding` 偏移 32；
       实测与本机 ObjC 定义一致），compute 侧 `vector_uint2 + 2×uint32` = 16 字节、`srcLevel` 偏移 8。
     ② **顺带删死代码**：`-[MGLRenderer lazyRefreshGLSampledRenderTargetCopyForTexture:stage:program:binding:unit:]`
     （**52 行**）**全树无调用点、无声明**（第 60 条那次事故时被一并回插），本刀直接删除。
     ③ **退掉两个纯数据端口（这才是净减来源）**：
     - `mglRendererCommandStatePort`（29 处调用点）→ **C 函数** `mglRendererCommandStateFor()`（实现在
       `mgl_renderer_ports.c`，内部就是 `areas.command`），调用点机械改名，**端口与 shim 包装一起消失**；
     - `mglRendererMdiScratchBufferPort` → **C 函数** `mglRendererMdiScratchBuffer()`：其 ObjC 方法
       `-[MGLRenderPassManager mdiArgumentScratchBufferWithDevice:length:offset:]`（36 行）整体搬走
       （arena 本就是 C++ `mglRenderAllocateMDIScratch`，owner 就在 `areas.command->mdiArgsScratchOwner`），
       该 ObjC 方法及其头声明一并删除，其唯一调用者（原先就是端口）随之消失；
     - 另删掉因它而失去唯一调用者的 file-static `mglRenderPassManagerCommandBufferState`（死代码）。
     ④ **净减账**：**端口 21 → 20**（−2 退役、+1 `mglRendererEnsureWritableCommandBufferPort`），
     `mgl_renderer_ports.h` 声明面与 shim 实现面**都是 20**；shim **350 → 331 行**、**49 → 47 语法**。
     全仓：ObjC 行数 **36,994 → 36,482**、语法 **2,134 → 2,128**、词汇 **4,056 → 3,992**；
     `MGLRenderer*.m` **33,451 → 33,002**（`+Blit.m` 4,507 → **4,061**）。
     ⚠️ **如实记账**：语法只降 6，因为 11 处调用点各引入 `(__bridge void *)` 桥接（与第 46/59 条同款现象），
     收益体现在**行数与词汇**上，不宣称语法收益。
     ⑤ **oracle（本刀必须做：整块渲染逻辑改写）**：旧库 = 提交 `6cc5c50` 的独立构建（worktree 更新到该提交后重建，
     `cmp` 两库不同）；语料 default 与 `MGL_BATCH_MAX_DRAWS=1` 两臂，均由 `test_regression` 产生：
     **确定性 trace 行 4,980/4,980 与 5,513/5,513 逐行保序完全一致**，其中
     **`RT_SAMPLE_COPY_*`（本刀改写的路径）248/248 与 276/276 完全一致**（该行携带 tex/label/尺寸/levels/
     `writeVersion`/`dirtyBefore`/`copyMask`/`copiedMask`/`dirtyAfter`/`levelSizes`/`compute` 全部字段）；
     唯一差异仍是与本刀无关的非确定行 `processGLState.slow`（同库连跑两次同样差）；
     stderr `MGL` 行 **307/307 多重集一致**（仅剔除 `BINARY ARCHIVE` 运行序行）；
     回归 plain **92/0/2**、ICB 门禁 **82/10/2** 在新旧库上完全相同。
     ⑥ **门禁与 CTS**：`verify_gl_api` 离线通过 + 28 个目标全过（`GATE_EXIT=0`）；
     **CTS 七簇非通过集合 diff 全空**（hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 ·
     piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
     下一刀：`+Blit.m` 里仍在 4k 行档，建议按域取 **`blitFramebufferDirectColorCopyWithState`(91) 之后的
     scaled/format-conversion 簇**，或 `+Texture.m` 的 `freshGLSampledRenderTargetCopyForSampling:`（与
     `lazyRefresh` 同域，但**有真实调用点**，需连带把采样视图窗口逻辑一起下沉）；同时按 §0.08 推
     ICB 门禁 82/10/2 → 92/0/2 与 T5 唯一壳合并。

63. **P0-1 第九刀：删掉 18 个 `mtlDraw*` 一行转发 + 退役 capture-cull 两端口（**shim 净减 2，20 → 18**）**：
     ① **`MGLRenderer+Draw.m` 的两层转发合一**：该文件的 18 个 C 桥（`mglRendererDrawArrays` 等）原本是
     "取 backend lease → `@autoreleasepool` → `[renderer mtlDrawArrays:…]`"，而 `mtlDraw*` 方法体**只是一行**
     `mglIssueDraw*` / `mglDrawHostGuardIssue*` 调用。现在桥内**直接调那个 C 函数**（`self` → `(__bridge void *)renderer`，
     方法参数名按桥参数名对齐：`ctx→glm_ctx`、`instancecount→instance_count`、`basevertex→base_vertex`、
     `baseinstance→base_instance`、multi 版 `first/count/drawcount/basevertex→firsts/counts/draw_count/base_vertices`），
     19 个方法与 `MGLRenderer+Draw_Private.h` 里的声明一起删除（−99 行方法 + −66 行声明）。
     **backend lease 与 `@autoreleasepool` 保留**（逐 draw 排空临时对象是既有语义，不属本刀范围）。
     ② **capture-cull 端口退役**：`mglRendererCaptureCullArrayPort` / `mglRendererCaptureCullElementPort` 的 ObjC 实现
     只是转发到 `mglDrawHostCaptureCullDistanceArray/Element`（本就是 C，声明在 `mgl_draw_issue.h`），
     于是 `mgl_batch_issue_encode.c` 的两处调用点改**直调 C**，端口 + `+DrawStageHost.m` 里两个方法（28 行，
     全树无其它调用者）+ 私有头声明（21 行）一并删除。
     ③ **净减账**：**端口 20 → 18**（`mgl_renderer_ports.h` 声明面与 shim 实现面都是 18）；shim **331 → 298 行**、
     **47 → 45 语法**。全仓：ObjC 行数 **36,482 → 36,281**、语法 **2,128 → 2,106**、词汇 **3,992 → 3,986**；
     `MGLRenderer*.m` **33,002 → 32,834**（`+Draw.m` 511 → **372**）。
     ④ **oracle**：旧库 = 提交 `6df99ca` 的独立构建（`cmp` 两库不同）；两臂 trace 的**确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**（含全部 `RT_SAMPLE_COPY_*`、`REPLAY_*`、`IFACE DUMP` 行），
     stderr `MGL` 行 **307/307 多重集一致**；唯一差异仍是与本刀无关的非确定行 `processGLState.slow`
     （两侧各自数量随机，同库连跑两次同样差）。**plain 92/0/2、ICB 门禁 82/10/2 在新旧库上相同**。
     ⑤ **门禁与 CTS**：`verify_gl_api` 离线通过 + 28 个目标全过（`GATE_EXIT=0`）；
     **CTS 七簇非通过集合 diff 全空**（hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 ·
     piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
     ⑥ **教训（第四次同源，值得单列）**：第一次改写时我用**字符偏移**（`len('\n'.join(lines[:i]))`）在"先改写、后删除"
     的两步之间做切片，因为改写让文本变长，偏移全部错位 → **把文件截断成 400 行垃圾**（编译报 `missing '@end'`）。
     **规则：同一文件的多步编辑一律基于行列表（或每步重读文件、重算位置），绝不在文本长度会变的步骤之间复用字符偏移。**
     下一刀：`MGLRenderer+DrawSupport.m`(351) / `+GPURecovery.m`(351) / `MGLRenderer+SwapDiagnostics.m`(557) 里
     同类"一行转发 C"的壳（先按本刀手法清掉再谈方法体下沉）；以及 §0.09 列表里的 ICB 门禁与 T5 唯一壳合并。

64. **P0-1 第十刀：element-buffer / processBuffer / sampler-snapshot 三簇体转 C（**shim 净减 4，18 → 14**）**：
     ① **一次退四个端口**，做法都是"方法体本来只依赖 C，只有 `id` 挡着"：
     - `mglRendererResolveElementBufferPort` → **三个 C 函数** `mglRendererResolveElementBufferForDraw` /
       `…ForCommand` / `…`（`getElementBuffer`、`mglRendererGetValidatedBuffer`、`mglDrawCommandElementBuffer`、
       `mglDispatchError` 全是 C；`NSLog` → `fprintf(stderr, …)` 同 sink 同字段；`id *mtlBufferOut` → `void **`），
       5 处 C 调用点（dyn_bind / icb_mdi / issue_encode ×3）改直调；
     - `mglRendererProcessBufferPort` → `mglRendererProcessBuffer`：方法体是"缺 Metal 存储就 bind、脏了就走
       `mglRenderUpdateDirtyBuffer`"（`-processBuffer:` 22 行删除），**唯一保留的端口是 `BindMTLBufferPort`——
       因为它要 `METAL_LOCK`**；7 处 ObjC 调用点（+Compute / +Tessellation ×2 / +Texture / +DrawSupport / draw_metal_port）
       改为 `mglRendererProcessBuffer((__bridge void *)self, …)`；
     - `mglRendererUpdateDirtyBaseBufferListPort` → C 调用者直接调 `mglRenderUpdateDirtyBaseBufferList(ctx, list, where)`
       （`ctx` 由 `areas.ctx`/`c->ctx` 提供），声明放进 **C 头** `mgl_renderer_ports.h`（`MGLRenderer+Draw_Private.h`
       是 ObjC 头，C 不能 include）；
     - `mglRendererSamplerStateForSnapshotKeyPort` → `mglRendererSamplerStateForSnapshotKey`：整段体（缓存查询 + 物化 +
       backend Put）**逐行搬到 C**，只把 `r->_backend` 换成 `areas.backend`；所有权注释与"未持有引用"契约原样保留。
     ② **净减账**：**端口 18 → 14**（声明面与 shim 实现面都是 14）；shim **298 → 230 行**、**45 → 38 语法**。
     全仓：ObjC 行数 **36,281 → 36,121**、语法 **2,106 → 2,093**、词汇 **3,986 → 3,972**；
     `MGLRenderer*.m` **32,834 → 32,742**；`+DrawSupport.m` 350 → **275**、`MGLRenderer.m` −22 行。
     ③ **oracle（本刀改了逐 draw 的 buffer 处理与采样器快照路径，必须 A/B）**：旧库 = 提交 `a521302` 的独立构建
     （`cmp` 两库不同）；两臂 trace 的**确定性行 4,980/4,980 与 5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307
     多重集一致**。**非确定项做了对照实验**：`processGLState.slow` 行数在"新 vs 旧"是 295/341（default）与 489/595（flushy），
     而**同一份库连跑两次**也是 295/340 与 489/595 —— 证实该行计数与改动无关，本刀仍如实报告而不是写成"diff 全空"。
     plain **92/0/2**、ICB 门禁 **82/10/2** 在新旧库上相同。
     ④ **门禁与 CTS**：`verify_gl_api` 离线通过 + 28 个目标全过（`GATE_EXIT=0`）；
     **CTS 七簇非通过集合 diff 全空**（hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 ·
     piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
     下一刀：剩余 14 个端口里 **`ProcessGLState` / `UpdateDirtyBuffer` 一类"只差一个 ObjC 字段访问"的**已不多，
     建议转向 §0.09 的 T5 唯一壳（`MGLPlatformRendererShell.m` + `+Lifecycle.m` + shim 的 1 个方法）与
     `+DrawSupport.m` 剩余方法（`resolveIndirectBufferForDraw` / `prepareEmulatedIndirectCPURead` /
     `currentDrawRasterizationIsEmpty` / `applyPolygonOffsetForDrawMode` / `ensureRasterEncoderForDraw`）——
     后四个已是"谓词/一行转发"形（1–3 行体），按本刀手法可整批下沉。

65. **P0-1 第十一刀：draw-support 谓词落 C + `METAL_LOCK` 真相（**shim 净减 1，14 → 13**）**：
     ① **新 TU `mgl_draw_support.{h,c}`**：`+DrawSupport.m` 的四个方法整块转 C——
     `resolveIndirectBufferForDraw:`(35) · `currentDrawRasterizationIsEmpty`(64) · `applyPolygonOffsetForDrawMode:`(46) ·
     `currentDrawModeIsFullyCulled:`(9) → `mglDrawResolveIndirectBuffer` / `mglDrawRasterizationIsEmpty` /
     `mglDrawApplyPolygonOffset` / `mglDrawModeIsFullyCulled`。13 处调用点（`mgl_draw_metal_port.m` 7、`+Tessellation.m` 4、
     其余在 C 侧）改直调；`+DrawSupport.m` **350 → 124 行**（`+DrawSupport_Private` 头里 4 条声明删除）。
     实现细节：`_renderPassManager.state` → `areas.command`、`ctx` → `areas.ctx`、`MGL_STATE(ctx)->x` →
     `ctx->active_state->x`、`_bindingStateOwner` → `*areas.binding_state_owner`；`mglDrawSupportTextureInfo(id)` →
     直接 `mglRenderGetTextureInfo(void *)`；`NSLog` → 同 sink 同字段 `fprintf`。
     ② **接口坑（记一次）**：`mgl_draw_mode.h` 虽是"纯 inline 谓词"头，但里面用了 **`NSUInteger`**（第一个 C 使用者会编译失败），
     C 侧改用其底层 C 函数 `mglRenderDrawModeProducesPolygons((uint64_t)mode)`；另一个坑是我在新头注释里写了
     `mglRender*/…`，`*/` 提前结束注释直接把头文件语法搞坏——**头注释里不许出现 `*/` 组合**。
     ③ **退役 `mglRendererBindMTLBufferPort`（本刀净减来源）**：查 `MGLRenderer_Private.h` 发现
     **`METAL_LOCK()` 只是 `MGL_ASSERT_GL_THREAD()`、`METAL_UNLOCK()` 是空操作**（互斥量早已不存在），
     于是 `bindMTLBuffer:` 的锁壳不再需要 ObjC：整个 bind（`mglRenderBindBufferStorage` + 同字段诊断）搬成 C 的
     `mglRendererBindMTLBuffer`，`mglRendererProcessBuffer` 与 `mgl_batch_dyn_bind_encode.c` 直调；**shim 14 → 13 端口 / 230 → 223 行**。
     ④ **度量**：全仓 ObjC 行数 **36,121 → 35,959**、语法 **2,093 → 2,077**、词汇 **3,972 → 3,958**；
     `MGLRenderer*.m` **32,742 → 32,589**。
     ⑤ **oracle**：旧库 = 提交 `0100520` 的独立构建（`cmp` 两库不同）；两臂 trace 的**确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；`processGLState.slow` 计数本轮 default 恰好
     290/290、flushy 320/291（已知非确定，见第 64 条的同库双跑对照）。plain **92/0/2**、ICB 门禁 **82/10/2** 新旧库相同。
     ⑥ **门禁与 CTS**：`verify_gl_api` 通过 + 28 目标全过（`GATE_EXIT=0`）；**CTS 七簇非通过集合 diff 全空**
     （hotspot 1270/52/4/1+1cw · tess 139/1 · GS 136/0 · refq 164/54/5 · piq 17/12/1 · compute 113/38/1ns · pp 1/3/1ns）。
     下一刀：**剩余 13 个端口里 `MapBuffersToMTL` / `BindMTLTexture` / `ProcessGLState` 的"锁"已成纸面理由**（同 ③），
     可逐个把方法体的 C 部分搬走；`+DrawSupport.m` 只剩 `prepareEmulatedIndirectCPURead`(30) 与
     `ensureRasterEncoderForDraw`(68)（前者要 `flushCommandBuffer:`、后者要 `newRenderEncoderLockedWithReason:`，
     两者仍属 ObjC 大方法）——下一步建议评估这两个方法的 C 化，或转 §0.09 的 T5 唯一壳。

66. **P0-1 第十二刀：`MGLRenderer+Draw.m` 整文件转 C（**文件 21 → 20**，本轮第一个 .m 消失）**：
     ① **做法**：该文件在第九刀之后只剩 18 个 C 桥 + 3 个 C helper，唯一的 ObjC 成分是
     18 处 `@autoreleasepool { … }`、`MGLRenderer *` 局部与 `(__bridge void *)`。逐项等价替换：
     - `@autoreleasepool { X }` → **`objc_autoreleasePoolPush()` / `objc_autoreleasePoolPop(pool)`**
       （`@autoreleasepool` 的底层就是这两个 libobjc C 函数，链接已有 `-lobjc`；**逐 draw 池语义不变**）；
     - `static MGLRenderer *mglRendererDrawTarget(GLMContext)`（原来是 ObjC 私头里的
       `static inline mglRendererForContext`）→ `static void *… { return ctx ? ctx->platform_renderer_shell : NULL; }`
       （`platform_renderer_shell` 本就是 `void *`）；
     - `mglRendererEnterBackendLease` 同样来自 ObjC 私头（`static inline` 包 `mglRendererBackendBeginContext`），
       C 侧自带同义 inline；`(__bridge void *)renderer` → `renderer`；空的 `@implementation MGLRenderer (Draw)` 删除。
     - include 换成 C 头（`glm_context.h` / `mgl_renderer_backend.h` / `mgl_sampler_compat.h` / `mgl_draw_issue.h` …），
       文件 `git mv` 为 **`MGL/src/mgl_draw_entry.c`**（Makefile 用 `wildcard MGL/src/*.c`，无需改构建脚本）。
     ② **度量**：`objc_zero.sh` **文件 21 → 20**（本周期 P0-1 阶段第一次真正删掉一个 `.m`）、行数 **35,959 → 35,587**、
     语法 **2,077 → 2,036**、词汇 3,958（持平）；`MGLRenderer*.m` **32,589 → 32,218**。
     ③ **如实记账**：**本刀没有退役端口**（shim 仍 13/223）。§0.04 的"净减"针对的是"把消息发送搬进 shim 充数"，
     本刀是**整个 ObjC TU 消失**这一更高一层的收益；不把它计入 shim 账。
     ④ **oracle**：旧库 = 提交 `0716d8a` 的独立构建（`cmp` 两库不同）；两臂 trace 的**确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**（`processGLState.slow` 288/292、304/293，
     已知非确定）；plain **92/0/2**、ICB 门禁 **82/10/2** 新旧库相同；**CTS 七簇非通过集合 diff 全空**；
     `verify_gl_api` 通过 + 28 目标全过（`GATE_EXIT=0`）。
     ⑤ **并行工作区提示（必须记录）**：本刀提交时工作区里还有**不属于本刀的并行改动**
     （`MGL/include/mgl_env_flag.h`、`MGL/include/mgl_types_state.h`、`MGL/src/draw_command.c`、`MGL/src/mgl_batch_path.c`、
     `Makefile`（新增 `test-state-dataflow`）、`scripts/state_dataflow_coverage.py`）。它们**未随本刀提交、未被修改**，
     但本刀 A/B 的 new 臂是在**含这些改动的树**上构建的，因此结论应读作"合成树 vs `0716d8a` 在这些语料上无可观测差异"。
     **规则：同一 checkout 可能有并行改动时，提交前必须 `git status` 逐条确认归属，只 `git add` 自己的路径（禁用 `git add -A`）。**
     下一刀：继续 T1/T2 式的**整文件转 C**——候选 `+VertexLayout.m`(203) / `MGLRenderer+DrawStageHost.m`(334) /
     `MGLPipelineCache.m`(446) / `+Binding.m`(490)，先扫每个文件剩余的 ObjC 成分是否只剩锁壳/断言；
     同时按 §0.09 的"纸面理由清单"复评 `BindMTLTexture` / `ProcessGLState` / `MapBuffersToMTL` 三端口。

67. **P0-1 第十三刀：`+DrawSupport.m` 并入 `+RenderPass.m`（**文件 20 → 19**）**：
     ① 该文件在第十、十一刀后只剩两个方法——`prepareEmulatedIndirectCPURead:`(30) 与 `ensureRasterEncoderForDraw`(68)——
     而两者调用的正是 `+RenderPass.m` 里的东西（`-flushCommandBuffer:` / `-processGlState:` / `-newRenderEncoderLockedWithReason:`）。
     按"依赖在哪就并到哪"的原则，把 98 行方法体整块移入 `MGLRenderer+RenderPass.m` 末尾（含同源注释块），
     补齐三个 include（`MGLRenderer+DrawSupportUtil.h`、`mgl_draw_mode.h`、`mgl_draw_encode.h`），随后 `git rm` 该文件。
     方法是纯位移，**没有改一行逻辑**，头声明留在 `MGLRenderer+Draw_Private.h` 不动（调用点零改动）。
     ② **度量**：`objc_zero.sh` **文件 20 → 19**、行数 **35,587 → 35,571**、语法 **2,036 → 2,032**、词汇 3,958（持平）；
     `MGLRenderer+RenderPass.m` 7,099 → 7,197（+98）。
     ③ **oracle**：旧库 = 提交 `1565fdf` 的独立构建（`cmp` 两库不同）；两臂 trace 的**确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**（含 `built=` 归一后 multiset 相同），stderr `MGL` 行 **307/307 多重集一致**；
     plain **92/0/2**、ICB 门禁 **82/10/2** 新旧库相同；**CTS 七簇非通过集合 diff 全空**；
     `verify_gl_api` 通过 + 28 目标全过（`GATE_EXIT=0`）。
     ④ **纪律**：本刀同样**不动 shim**（13 端口 / 223 行），按第 66 条的口径只计入"ObjC TU 数量"这一层收益；
     提交时工作区曾有第 66 条记录的并行改动，**只 `git add` 本刀路径**；那些改动在提交前已从工作区消失（非本刀所为），
     因此**提交后在纯净树上重建（0 error）并重跑 `test-regression` 92/0/2、`test-batch-icb: ok`、`test-dirty-hash`、`test-es-smoke`**
     以确认"提交内容本身"可用；本刀的 CTS/A-B 结论仍标注为"当时含并行改动的合成树"下测得。
     下一刀（轮次将尽）：按 §0.09 继续"整文件转 C / 类别合并"——`+VertexLayout.m`(203, 仅 7 语法) 与 `MGLPipelineCache.m`(446)
     的 ObjC 成分已很薄，先补 `_tessellation` 进 areas、再评估 `bindFramebufferTexture:isDrawBuffer:` 与
     `mapGLBuffersToMTLBufferMap:stage:` 两条 ObjC 链；同时保留"每刀给 net 数字 + 自跑 oracle"的纪律。

### 0.11 goal 轮次用尽后的开工清单（2026-09-13 13:05，第 24 轮末）

**已确认的"零成本"下一刀（按投入产出排序，均已实测过成分）**：

| 目标 | 现状 | 下一步动作 | 预计收益 |
|---|---|---|---|
| **`+VertexLayout.m`** | 203 行 / 仅 **7** 语法；3 个方法 | `updateBlendStateCache`(90 行) 只用 `ctx->active_state` + `mglRenderBlend*` → 直接搬 C；`generateVertexDescriptorState:`(11 行) 只差 `_tessellation`（把 4 个字段并进 `MGLRendererStateAreas` 即可）；`bindFramebufferAttachmentTextures`(79 行) 需先 C 化 `bindFramebufferTexture:isDrawBuffer:` | 整个 `.m` 消失（19 → 18），−200 行 |
| **`MGLPipelineCache.m`** | 446 行 / 62 语法 | 方法体多是 `mglRender*PipelineCacheOwner*` 转发（第 65 条同款"只差 ObjC 句柄"） | 可能整文件转 C |
| **`processGLStateLocked:`** | `+RenderPass.m` 内数百行 | 体里 ObjC 只有 `@try/@catch`（Metal 恢复）、`_device`/`_commandQueue`（backend 句柄，areas 已有 `backend`）与少量方法发送；按第 65 条先逐一列依赖 | 退役 `ProcessGLState` 端口 + 数千行降 ObjC |
| **`bindMTLTextureLocked:`** | `+Binding.m` ~190 行 | 依赖 `uploadDirtyCPUTextureData*` / `uploadFullCPUTextureDataIntoTexture` / `endRenderEncodingLocked` / `createMTLTextureFromGLTexture` 四条 ObjC 链；先 C 化其中最薄的一条（`endRenderEncodingLocked`） | 退役 `BindMTLTexture` 端口 |
| **`mapGLBuffersToMTLBufferMap:stage:`** | `+Buffer.m` ~90 行 | 重活在 `mapShaderBufferResourcesToBufferMap:stage:`；逐段搬到 `mgl_binding_policy.c`/`mgl_vertex_attrib_plan.c`（plan 已是 C） | 退役 `MapBuffersToMTL` 端口 |
| **T5 唯一壳** | shim(223) + `MGLPlatformRendererShell.m`(229) | 审计硬要求：合并成**一个**壳 TU 并写明行数上限与移除路径；shim 内已只剩 1 个 ObjC 方法（`flushDrawBuffer:`） | 满足 T5 终态形态 |

**每刀固定动作**（第 51/61/66 条已固化，照抄即可）：
1. 改前先"抽取即落盘副本"；多步编辑**基于行列表**、不要在文本长度会变的步骤间复用字符偏移（第 63 条事故）。
2. 删/转方法后用**裸 selector 名**全树 grep（含 `self.x` 与 `[other x]` 接收者）。
3. 旧库 = `git worktree` 到上一提交构建（借 `config.mk` + `build/aux` + `external/glfw/build`，**永不 `make clean`**），`cmp` 两库不同后再跑。
4. oracle：`/private/tmp/run_ab<N>.sh {new,old}`（default 与 `MGL_BATCH_MAX_DRAWS=1` 两臂，均由 `test_regression` 产生）+ `/private/tmp/ab_full.py`；
   `processGLState.slow` 行数**已知非确定**（同库双跑对照见第 64 条），报告时如实标注。
5. 门禁：`python3 scripts/verify_gl_api.py` + 28 个 `test-*` 目标；CTS：`/private/tmp/run_t4b_battery*.sh` 后与 `/tmp/base_<cluster>.txt` 逐簇 diff。
6. **提交纪律**：`git status` 逐条确认归属，只 `git add` 自己的路径（**禁用 `git add -A`**，见第 66 条）；推送 `git push origin main:main`
   （22 端口不通时用 `git -c url."ssh://git@ssh.github.com:443/53453450/MGL-minecraft.git".insteadOf="git@github.com:53453450/MGL-minecraft.git" push origin main:main`）。
7. 文档：§0.0 进度行 + §5 新条目 + §0.09 交接状态，三处同步刷新。

68. **P0-1 第十四刀：删掉 6 个编译器证明死掉的 static 函数（**−80 行 / 语法 −3 / 词汇 −14**）**：
     ① 由 `-Wunused-function` 精确定位，逐个删除（删前用花括号配对切块，连同紧邻注释一起删）：
     `+Tessellation.m` 的 `mglTessCreateTextureLevelView`(16) · `mglTessPlanBytesOrBind`(12) ·
     `mglTessPlanDispatchOrBind`(31) · `mglTESXFBFieldByteSize`(4) · `mglCheckedNSUIntegerProduct`(11)；
     `+Texture.m` 的 `mglTextureBufferLength`(6)。删后重编**再无 unused-function 告警**（这两文件此前每次全量构建都在刷）。
     ② **度量**：行数 **35,571 → 35,491**、语法 **2,032 → 2,029**、词汇 **3,958 → 3,944**；文件仍 19、shim 仍 13 端口 / 223 行。
     ③ **oracle**：旧库 = 提交 `5473a84` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**、ICB 门禁 82/10/2 不变；
     **CTS 七簇非通过集合 diff 全空**；`GATE_EXIT=0`。
     ④ 说明：本刀是"清死代码"，**不动 shim / 不转文件**，只如实计行数、语法与词汇的下降。
     下一刀（按 §0.11 第一行）：`+VertexLayout.m`——`updateBlendStateCache` 的体只差
     `[_pipelineCache setBlendFactorsForAttachment:…]`（其体是 `[self ensureOwner]` + `mglRenderSetPipelineBlendState(_owner,…)`，
     需要一个 C 可见的 setter 或把 `MGLPipelineCacheState` 的 blend 部分做成 state area 的可变指针）；
     `generateVertexDescriptorState:` 需要把 `MGLTessellationState` 变成 C 安全头并挂进 `MGLRendererStateAreas`；
     `bindFramebufferAttachmentTextures` 需先 C 化 `bindFramebufferTexture:isDrawBuffer:`（`+RenderPass.m:1931`）。

69. **T5 第一步：端口 shim 并入平台壳，ObjC 壳 TU 唯一化（**文件 19 → 18**）**：
     ① 做法：把 `mgl_renderer_port_shim.m`（223 行、13 个端口 + 1 个 category 方法）整块并入
     **`MGLPlatformRendererShell.m`**，合并 include（去重 11 条）并加分段注释；随后 `git rm` 该文件。
     两个文件本就没有同名 static / 全局符号（合并前逐名比对过 `static` 与 `mgl*` 定义，交集为空）。
     **端口数不变（仍 13）**——这是审计要求的"ObjC 壳只有一个 TU"的形态收口，**不冒充净减**。
     ② **踩坑（重要，值得单列）**：`test_metalcpp_smoke` 这条 C++ harness **单独编译 `MGLPlatformRendererShell.m`**
     （不带库的其余部分），合并后端口包装把 `mglBatchTeardownReplay` / `mgl_batch_mtl_create_icb` 等符号拖进来 → 链接失败。
     另外同一 TU 里同时看到 `MGLRenderer+Draw_Private.h`（无 `extern "C"`）与 `mgl_renderer_ports.h`（`extern "C"`）里的
     `mglRenderUpdateDirtyBaseBufferList` 声明 → ObjC++ 下报 **different language linkage**。
     修法：① 给那条声明补 `extern "C"`（它本来就是 C 函数）；② smoke 规则加 `-DMGL_PLATFORM_SHELL_SMOKE`，
     端口段用 `#ifndef MGL_PLATFORM_SHELL_SMOKE … #endif` 包住（smoke 只验证壳本身仍能以 ObjC++ 编译）。
     **规则：把"平台壳/端口"文件合并进任何被独立 harness 编译的 TU 前，先查 Makefile 里哪些 target 单独编译它。**
     ③ **度量**：`objc_zero.sh` **文件 19 → 18**、行数 **35,491 → 35,478**、语法 2,029、词汇 3,944；
     `MGLPlatformRendererShell.m` 229 → 437 行（壳 229 + 端口 208）。
     ④ **oracle**：旧库 = 提交 `429d4c8` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；
     **CTS 七簇非通过集合 diff 全空**；全门禁（28 目标）`GATE=0`（含修好后的 `test-air` / `test-metalcpp`）。
     下一刀：T5 第二步——把 `+Lifecycle.m`(665) 里真正属于平台壳的部分（NSView/drawable/swap 提交）并入同一个 TU，
     其余生命周期逻辑按域下沉 C；同时按 §0.11 继续 `+VertexLayout.m` 的三条前置依赖（pipeline-cache blend setter、
     `MGLTessellationState` 进 areas、`bindFramebufferTexture:isDrawBuffer:`）。

70. **P0-1 第十六刀：删掉 `+DrawStageHost.m` 的 5 个"已被 C 取代"的死方法（**−70 行**）**：
     ① 判别方法（本轮固化为"死方法三条件"，比 `-Wunused-function` 更严）：
     **`grep -c "\[self <sel>"` 文件内自调用 = 0** ∧ **全树（含 .mm/.c/.h，排除声明）引用 = 0** ∧ **不是 host-ops 表里的函数指针**。
     按此筛出并删除：`prepareAndEncodeDirectCullDistanceElementDraw:`(16) · `encodeCullDistanceArrayDraw:`(13) ·
     `encodeCullDistanceElementDraw:`(17) · `captureAIRVertexPositionsForTessellation:`(14) · `validateDrawArraysVertexInputs:`(10)
     （它们的 C 对应物 `mglDrawHost*` 早已被 host-ops 直接调用，ObjC 方法成了空壳残留）；私有头里 5 条声明同步删除。
     ⚠️ **反例（差点误删）**：`fragmentNeedsPerSampleMSValuesForContext:` 全树外部引用为 0，但**文件内 `[self …]` 有 2 处**
     （MS 采样循环），属"只在本文件用"的活方法——**所以第一步必须查文件内自调用**。
     ② **度量**：行数 **35,478 → 35,413**、语法 **2,029 → 2,023**、词汇 **3,944 → 3,931**；文件仍 18、shim 端口仍 13。
     ③ **oracle**：旧库 = 提交 `1a58d80` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；
     28 目标门禁 `GATE=0`。
     下一刀：同一"死方法三条件"扫全仓（`+Binding.m` / `+GPURecovery.m` / `+SwapDiagnostics.m` / `MGLRenderPassManager.m`
     都是候选），把"已被 C 取代但方法壳还在"的残留一次清完；之后再回到 §0.11 的结构性刀（`+VertexLayout.m` 三条前置依赖、
     T5 第二步 `+Lifecycle.m` 并入唯一壳 TU）。

71. **P0-1 第十七刀：把"死方法扫描"做成脚本并修掉三个假阳性陷阱（**−15 行**，工具入仓）**：
     ① **事故（本轮自己造的，编译器抓住）**：按第 70 条的"三条件"手工扫出 7 个候选，我把其中 5 个（含
     `mtlBlitFramebuffer:` 412 行、`mtlClearBuffer:` 447 行）直接删掉，**编译立刻报错**——
     `MGLRenderer.m:3260 [renderer mtlSwapBuffers:glm_ctx]`、`+Blit.m:1851`、`MGLRenderer.m:3733` 都是**活调用**，
     它们就在**定义所在的同一个文件里**，而我上一轮的"全树 grep"用 `grep -v "^MGL/src/MGLRenderer\.m"` 把整个文件排除了，
     于是把"文件内其它接收者的调用"全漏掉。**已 `git checkout` 全部回退**，未进任何提交。
     ② **修法**：把判别固化成脚本 **`scripts/objc_dead_methods.py`**（入仓，可复跑），逐条处理 4 类引用：
     - 选择器**带冒号**（上一版把 `mglWindowGeometryChanged:` 截成 `…Changed` 导致误判，而它正是 NSNotification 的 selector）；
     - **任意接收者**的发送 `[<recv> sel…]`（不只 `[self …]`，也不再整文件排除）；
     - **属性语法** `obj.device = x`（等价于 `setDevice:`，上一版漏掉；`setDevice:` 因此是活的）；
     - **框架回调**（KVO `observeValueForKeyPath:…`、通知 selector）单列"需人工确认"。
     **结论/规则：静态扫描只负责给候选，`clang` 才是 liveness 的唯一裁判——删除后必须立刻编译验证**；
     这条与第 63 条（多步编辑别复用字符偏移）、第 66 条（并行改动时只 add 自己路径）并列为"本周期三次自伤"。
     ③ 脚本在修正后只剩 **3 个候选**，其中 2 个经编译验证确为死代码并删除：
     `+Compute.m` 的 `mtlDispatchCompute:`(9) 与 `mtlDispatchComputeIndirect:`(6)（调用方早已改直调 `…Locked` 版本）；
     第 3 个 `observeValueForKeyPath:` 是 KVO 回调，**保留**。脚本现在的稳态输出是**零候选**（唯一残留就是这条待人工确认项）。
     ④ **度量**：行数 **35,413 → 35,398**、语法 **2,023 → 2,021**、词汇 3,931；文件仍 18、shim 端口仍 13。
     ⑤ **oracle**：旧库 = 提交 `651eefb` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非对称集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：直接跑 `python3 scripts/objc_dead_methods.py`（应为 0 候选）后回到结构性刀：§0.11 的 `+VertexLayout.m` 三条前置依赖
     （pipeline-cache blend setter / `MGLTessellationState` 进 areas / `bindFramebufferTexture:isDrawBuffer:`）
     与 T5 第二步（`+Lifecycle.m` 平台部分并入唯一壳 TU）。

72. **P0-1 第十八刀：FBO attachment 绑定整簇转 C（**−93 行**，零新增端口）**：
     ① 新 TU **`mgl_attachment_binding.{h,c}`**，两个入口替换两个 ObjC 方法：
     - `-[MGLRenderer bindFramebufferTexture:isDrawBuffer:]`（`+RenderPass.m` 18 行）→ `mglRendererBindFramebufferTexture`；
     - `-[MGLRenderer bindFramebufferAttachmentTextures]`（`+VertexLayout.m` **77 行**）→ `mglRendererBindFramebufferAttachmentTextures`。
     两者**不需要任何新端口**：附件纹理走已有的 C 端口 `mglRendererAttachmentTextureFor(ctx, att)`，Metal 侧走已有的
     `mglRendererBindMTLTexturePort`，`ctx`/`fbo` 分别来自 `areas.ctx` 与 `areas.ctx->active_state->framebuffer`；
     `NSLog` → 同 sink 同字段 `fprintf`，`DEBUG_PRINT`（`glm_context.h` 的 C 宏）原样保留。
     ② 调用点与声明：`MGLRenderer.m` 1 处、`+RenderPass.m` 2 处改直调；`+RenderPass_Private.h` 两条声明注释化。
     `+VertexLayout.m` **203 → 126 行**（该文件只剩 `generateVertexDescriptorState:` 与 `updateBlendStateCache`）。
     ③ **定位教训**：`bindFramebufferTexture:isDrawBuffer:` 的**定义不在 `+VertexLayout.m` 而在 `+RenderPass.m`**
     （`+VertexLayout.m` 里那 3 处只是调用点）——我先按"文件里有这个名字"就以为定义在同文件，脚本三次
     `StopIteration` 才发现。**规则：动手前用 `grep -n '^- *(' 定位"定义行"，不要用裸名字计数当定义证据。**
     ④ **度量**：行数 **35,398 → 35,305**、语法 **2,021 → 2,016**、词汇 **3,931 → 3,927**；文件仍 18、shim 端口仍 13。
     ⑤ **oracle（本刀改的是每次 FBO 绑定都走的路径，必须 A/B）**：旧库 = 提交 `f11ecba` 的独立构建（`cmp` 两库不同）；
     两臂 trace **确定性行 4,980/4,980 与 5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；
     plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 + `verify_gl_api` 均通过。
     下一刀：`+VertexLayout.m` 余下两方法（`generateVertexDescriptorState:` 11 行 + `updateBlendStateCache` 90 行）——
     前者差"`_tessellation` 三个字段进 areas"，后者差"pipeline-cache blend setter"。两条都能用**既有模式**零端口解决：
     给 `MGLRendererStateAreas` 加 `void *pipeline_cache_object` 与一个 tessellation 快照字段（areas 加字段不新增端口），
     再由壳 TU 里填（`_pipelineCache` / `_tessellation`），C 侧通过回调函数指针调用 cache 的既有方法。

73. **P0-1 第十九刀：`+VertexLayout.m` 整个文件转 C 并删除（**文件 18 → 17**，零新增端口）**：
     ① 新 TU **`mgl_vertex_layout.{h,c}`** 替换最后两个方法：
     `-generateVertexDescriptorState:`(11) → `mglRendererGenerateVertexDescriptorState`；
     `-updateBlendStateCache`(89) → `mglRendererUpdateBlendStateCache`。文件只剩空 `@implementation`，**整文件删除**。
     ② **零新增端口的两个抓手（本刀的通用手法，值得复用）**：
     - **areas 加字段不算新端口**：`MGLRendererStateAreas` 增加 `void *pipeline_cache_object`、三个 tessellation 快照字段
       （`tess_native_tes_active` / `tess_native_tes_program` / `tess_tcs_output_stride`），由**壳 TU 填**
       （`r->_pipelineCache`、`r->_tessellation.*`）。blend 结构用**前置声明** `struct MGLRenderPipelineBlendState_t`
       避免让 C 头拖进 `mgl_render.h`。
     - **函数指针放进 areas**：`int (*pipeline_cache_set_blend)(void *, uint32_t, const MGLRenderPipelineBlendState *)`，
       壳 TU 里实现为 `mglPlatformShellPipelineCacheSetBlend`（壳是 ObjC，可以直接发 `-setBlendFactorsForAttachment:…`），
       于是 C 能上传 blend 而**不需要新端口**、也不需要碰 cache 的私有 ivar。
     ③ **坑**：`(void *)` 与 `(__bridge void *)` 混用会分别报"incompatible types casting 'Program *'"与
     implicit-function-declaration 两类错——`Program *` 是 C 结构体指针，**只能直接转 `void *`**（`__bridge` 只服务 ObjC 对象）；
     `mglRenderGenerateVertexDescriptorState` 的声明在 ObjC 头里，C 侧要自带 `extern` 声明（沿用既有做法）。
     ④ **度量**：`objc_zero.sh` **文件 18 → 17**、行数 **35,305 → 35,206**、语法 **2,016 → 2,015**、词汇 **3,927 → 3,924**。
     ⚠️ **如实记账**：语法只降 1——删掉的 `.m` 本身仅 7 处语法，而壳 TU 为填 areas 新增了 `(__bridge …)`/消息发送，
     两者基本抵消；本刀的收益在**文件数、行数与词汇**，不宣称语法收益。
     ⑤ **oracle（本刀改的是每次 blend 变更与顶点描述符构建的路径）**：旧库 = 提交 `e2bf9bc` 的独立构建（`cmp` 两库不同）；
     两臂 trace **确定性行 4,980/4,980 与 5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；
     plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 + `verify_gl_api` 均通过。
     下一刀：同类"整文件转 C"继续挑薄文件——剩余 `MGLRenderer+Binding.m`(490/42 语法) · `+GPURecovery.m`(350/32) ·
     `+DrawStageHost.m`(余 ~250) · `MGLRenderPassManager.m`(524) · `MGLPipelineCache.m`(446)；
     手法照本刀：先建 C 头 + C 入口（必要时用 areas 字段 + 函数指针），再删 ObjC 方法，最后删空文件。

74. **P0-1 第二十刀：`+DrawStageHost.m` 再清两个方法（**−24 行**）**：
     ① `fragmentNeedsPerSampleMSValuesForContext:`(6) → C 的 `mglDrawFragmentNeedsPerSampleMSValues`（并入
     `mgl_draw_support.{h,c}`），文件内 2 处 MS 采样循环的调用点改直调；`captureAIRVertexPositionsForGeometryIndexed:`(19)
     经"死方法三条件"（文件内自调用 0、全树引用 0、非函数指针）判定为**死方法，直接删除**。
     ② **本刀的一个自我纠错**：我先把 `captureAIRVertexPositionsForGeometryIndexed:` 的体也搬成了 C 函数
     （`mglDrawCaptureVertexPositionsForGeometryIndexed`），随后才发现该 ObjC 方法**已经没有任何调用者**——
     于是**把刚写的 C 函数也一并删掉**，没有把死代码从 ObjC 搬到 C（"搬运死代码"不算进度）。
     **规则：迁移前先跑一遍死方法判定；死的东西直接删，不要换个语言继续养着。**
     ③ **度量**：行数 **35,206 → 35,182**、语法 **2,015 → 2,010**、词汇 **3,924 → 3,919**；文件 17、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `921b57b` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与
     5,513/5,513 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀（轮次预算内继续）：`+DrawStageHost.m` 只剩 5 个方法（MS 采样循环族 + `runVertexCaptureSession:` +
     `bindCullDistanceEmulationBuffers:` + `emulatedMSColor0TextureForContext:`）；`bindCullDistance…` 需要
     areas 再补两个 `_tessellation.cullDistanceCapture*` 字段 + 一个 `recordLastBoundVertexBuffer:` 的 C 入口
     （照第 73 条的"areas 字段 + 函数指针"手法，零端口）；MS 循环族因**含 block 参数**（`void (^)(void)`）暂留 ObjC。

75. **P0-1 第二十一刀：MS color0 取样转 C（**−22 行**）**：
     `-[MGLRenderer emulatedMSColor0TextureForContext:]`(22) → C 的 **`mglDrawEmulatedMSColor0Texture`**（并入
     `mgl_draw_support.{h,c}`）：`MGL_STATE(ctx)->framebuffer` → `ctx->active_state->framebuffer`，
     `[self framebufferAttachmentTexture:att]` → 已有的 C 端口 `mglRendererAttachmentTextureFor(ctx, att)`；
     文件内 2 处 MS 采样循环调用点（`runEmulatedMSSampleDrawLoopIfNeeded` / `broadcastEmulatedMS…`）改直调。
     **零新增端口、零 areas 变更**。
     度量：行数 **35,182 → 35,160**、语法 **2,010 → 2,007**、词汇 3,919（持平）；文件 17、shim 端口 13 不变。
     oracle：旧库 = 提交 `e2b71d0` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。

### 0.12 当前 17 个 `.m` 的构成与"下一刀优先级"（2026-09-13，第二十一刀后）

| 文件 | 行数 | 备注 / 下一步 |
|---|---|---|
| `MGLRenderer+RenderPass.m` | ~7.1k | 三厚块之一；`processGLStateLocked:` 是最大单块（ObC 只剩 `@try/@catch` + backend 句柄 + 少量发送） |
| `MGLRenderer+Texture.m` | ~6.5k | 三厚块之一 |
| `MGLRenderer.m` | ~4.7k | 仍持大量 C 桥（`mglRenderer*`）与少量 ObjC 内部方法 |
| `MGLRenderer+Blit.m` | ~4.0k | 三厚块之一 |
| `MGLRenderer+BindingState.m` | ~2.9k | `bindTexturesToCurrentRenderEncoder:` 等编码器绑定族（对应 shim 4 个端口） |
| `MGLRenderer+Tessellation.m` | ~2.2k | 残留 ObjC 方法多与 tess 捕获/描述符有关 |
| `mgl_draw_metal_port.m` | ~1.95k | host-ops 表；方法体多为一行转发（本周期已清若干） |
| `MGLRenderer+Compute.m` | ~1.25k | 计算路径；已有 2 个死方法被清 |
| `MGLRenderer+Buffer.m` | ~0.8k | `mapGLBuffersToMTLBufferMap:stage:` 链（对应 `MapBuffersToMTL` 端口） |
| `MGLRenderer+Lifecycle.m` | ~0.66k | T5 第二步候选：平台部分并入唯一壳 TU（KVO/通知回调保留） |
| `MGLRenderer+SwapDiagnostics.m` | ~0.56k | 2 个方法，均在 swap 路径 |
| `MGLRenderPassManager.m` | ~0.52k | manager 类；`mdiArgsScratch` 等已 C 化 |
| `MGLRenderer+Binding.m` | ~0.49k | 多为一行转发到 `mglRenderBinding*`，整文件转 C 候选 |
| `MGLPipelineCache.m` | ~0.45k | cache 类；方法多为 `mglRender*PipelineCacheOwner*` 转发 |
| `MGLRenderer+GPURecovery.m` | ~0.35k | 10 个方法、每个 0–3 处发送；整文件转 C 候选（需 areas 暴露 `_device`/`_commandQueue` 或 C 入口） |
| `MGLRenderer+DrawStageHost.m` | ~0.23k | 剩 5 个方法（MS 循环族含 block、`runVertexCaptureSession:` 需写 `self->ctx`、`bindCullDistance…` 需 2 个 areas 字段 + 1 个 C 入口） |
| `MGLPlatformRendererShell.m` | ~0.44k | **唯一壳 TU**：平台壳 229 行 + 13 个端口 208 行（T5 目标形态） |

> 计数口径：行数取 `scripts/objc_zero.sh` 的逐文件输出（上一轮实测四舍五入）；
> **下一刀仍按"投入产出"排序：`+Binding.m` → `+GPURecovery.m` → `MGLPipelineCache.m` 的整文件转 C**
> （手法见第 73 条：C 头 + C 入口，必要时 areas 加字段/函数指针，零端口），
> 之后是 `+DrawStageHost.m` 余量与 `+Lifecycle.m` 的 T5 合并。

76. **P0-1 第二十二刀：`+Binding.m` 的 8 个 binding-state 转发全部转 C（**−43 行**）**：
     ① 新 TU **`mgl_binding_state_ops.{h,c}`**：`invalidateLastBoundState` · `recordLastBoundVertexBuffer:` ·
     `recordLastBoundFragmentBuffer:` · `invalidateLastBoundVertexBufferAtIndex:` · `invalidateLastBoundFragmentBufferAtIndex:` ·
     `setViewportIfNeeded:` · `setScissorRectIfNeeded:` · `setTriangleFillModeIfNeeded:` → 8 个 `mglBinding*` C 入口。
     每条体都只是 `mglRenderBinding*` 调用：binding owner 取 `*areas.binding_state_owner`，
     owner-aware 形式（viewport/scissor/fill）再取 `areas.command->currentRenderEncoderOwner`。**零新增端口**。
     ② **签名坑（记一次）**：`MGLViewportValue` / `MGLScissorRectValue` 只定义在 **ObjC 头** `MGLRenderer+Draw_Private.h` 里，
     C 头不能用它们；于是 C 入口改成**基本类型签名**（viewport 6 个 double、scissor 4 个整数），调用点传结构体字段。
     **规则：给 C 的接口只用 C 头里有的类型，否则拆成基本类型。**
     ③ 调用点：约 17 处（`+RenderPass.m` 7、`+DrawStageHost.m` 4、`+BindingState.m` 3（含 `MGL_SMB_INVALIDATE` 宏体内）、
     `+Lifecycle.m` 2）全部改直调；`MGLRenderer+Draw_Private.h` 里 8 条声明注释化。`+Binding.m` 490 → **448 行**
     （余下 `bindMTLTextureLocked:` 339 行与 `syncResourceBindingsForContext:` 27 行仍是 ObjC）。
     ④ **度量**：行数 **35,160 → 35,117**、语法 2,007（持平）、词汇 **3,919 → 3,913**；文件 17、shim 端口 13 不变。
     ⑤ **oracle**：旧库 = 提交 `2f3108c` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+Binding.m` 只剩两块 ObjC——`bindMTLTextureLocked:`(339) 与 `syncResourceBindingsForContext:`(27)；
     前者依赖 `createMTLTextureFromGLTexture` / `createFallbackMTLTexture` / `endRenderEncodingLocked` / `NSDate`（计时），
     建议按依赖顺序逐个转 C（每次转一个依赖后立刻编译 + 跑 A/B）；后者是 10 处发送的编排方法，可最后处理。

77. **P0-1 第二十三刀：删掉第七刀留下的死 setter（**−5 行**）**：
     `-[MGLRenderPassManager setTraceReplayFlushId:batchIndex:]` 是第七刀退役
     `mglRendererTraceReplaySetPort` 时**遗留下来的最后一个调用者侧残留**——C 端现在直接写
     `areas.command->traceReplayFlushId / …BatchIndex`，全树（含 `.mm/.c/.h`，排除 build）只在该 setter 的定义与
     `MGLRenderPassManager.h` 的声明里出现。删除方法 + 声明。
     度量：行数 **35,117 → 35,112**、语法/词汇持平（2,007 / 3,913）；文件 17、shim 端口 13 不变。
     oracle：旧库 = 提交 `1a237a8` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     **附带发现（工具口径）**：本轮先用"小方法 + 外部调用点数"粗筛了 54 个候选，结果**不可用**——大量 getter 是
     通过**属性语法** `obj.state` 访问的，粗筛只看 `[recv sel` 就会把它们误判成"零引用"（与第 71 条同一个坑）。
     **结论：只有 `scripts/objc_dead_methods.py` 的四类检查（带冒号选择器 / 任意接收者 / 属性 setter / 框架回调）
     可以作为判据；临时写的启发式扫描一律不要直接用来删代码。**
     下一刀：回到第 76 条的路线——`+Binding.m` 的 `bindMTLTextureLocked:`(339) 按其依赖顺序逐个转 C
     （先 `endRenderEncodingLocked`，再 `createMTLTextureFromGLTexture`/`createFallbackMTLTexture`，`NSDate` 计时改 `mglTraceNowSeconds()`）。

78. **P0-1 第二十四刀：`bindMTLBuffer:` 双方法删除，全部调用点走已有 C 函数（**−17 行**）**：
     ① 第十一刀已经产生 C 函数 `mglRendererBindMTLBuffer`（体与 `bindMTLBufferLocked:` 完全相同：`mglRenderBindBufferStorage`
     + 同字段诊断），但 ObjC 的 `bindMTLBuffer:`（6 行锁壳）与 `bindMTLBufferLocked:`（11 行）一直留着。
     本刀把 5 个文件里约 10 处 `[self bindMTLBuffer:…]` 全部改成 `mglRendererBindMTLBuffer((__bridge void *)self, …)`，
     然后**连方法带声明一起删除**（`MGLRenderer+Binding_Private.h` 两条声明注释化）。
     ② **教训**：`METAL_LOCK()` 是空断言早已查明（第 65 条），所以这类"锁壳 + Locked 双份"是**结构冗余**——
     一旦对应的 C 函数存在，就应该把两个方法一起删；本刀即"同一功能曾同时存在 C 与 ObjC 两份实现"的收尾。
     ③ **度量**：行数 **35,112 → 35,096**、语法 **2,007 → 2,006**、词汇 **3,913 → 3,912**；文件 17、shim 端口 13 不变；
     `+Binding.m` 448 → **431 行**（余 `bindMTLTexture*`/`syncResourceBindings*`）。
     ④ **oracle**：旧库 = 提交 `874c929` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+Binding.m` 只剩 `bindMTLTextureLocked:`(339) 与 `syncResourceBindingsForContext:`(27)。
     建议先转 `endRenderEncodingLocked`（`+RenderPass.m:5060`，约 60 行，OBjC 成分是 `_batching`/`_renderPassManager` 两个
     areas 已覆盖的状态 + 日志），转换后 `bindMTLTextureLocked:` 的依赖表就少一项；每转一项立刻编译 + A/B。

### 0.13 余下 `.m` 的"依赖深度"复核（2026-09-13；**第 43 轮更新：现为 16 个**）

第 78 条建议的下一刀是 `endRenderEncodingLocked`（`+RenderPass.m:5060`，约 91 行）。本轮**先把依赖逐条列出**，结论是
**它不能单独转 C**，需要先补 4 个前置：

| 依赖 | 现状 | 说明 |
|---|---|---|
| `[_renderPassManager endCurrentRenderEncoder]` / `clearCurrentRenderEncoder` / `clearRenderPassIdentity` | 3 个 manager 方法 | 各需一个 C 入口或先 C 化（后两者体很短，`clearFboMatchCache` 一类） |
| `@try { … } @catch (NSException *)` | ObjC 异常语义 | 只能留在 ObjC（或改由壳 TU 提供一个 `mglPlatformShellEndEncoderGuarded(...)` C 入口包住 @try） |
| `[self updateGLSampledCopiesForEndedRenderPassFramebuffer:…]` | 另一个 ObjC 方法 | 需一并处理 |
| `NSLog` / `_batching` / `_renderPassManager.state->…` | 混合 | 后者两项已在 areas 覆盖（`areas.batching` / `areas.command`），`NSLog` → `fprintf` |

**因此"下一刀"应按"依赖深度 ≤1"来挑**，而不是按行数挑。按此复核，当前**依赖深度 ≤1 的整文件候选**是：

| 文件 | 方法/行数 | 依赖深度复核 |
|---|---|---|
| `MGLRenderer+GPURecovery.m` | 10 / 350 | 每个方法 0–3 处发送，除 `_device`/`_commandQueue`（backend 句柄，areas 已有 `backend`）与 `NSLog` 外基本是 C；**最可能整文件转 C** |
| `MGLRenderer+DrawStageHost.m` | 4 / 217 | 剩 `runVertexCaptureSession:`（要写 `self->ctx`）、`bindCullDistanceEmulationBuffers:`（要 2 个 areas 字段 + 1 个 C 入口）、MS 循环族（含 block）；**逐条可做但零散** |
| `MGLRenderer+SwapDiagnostics.m` | 2 / 556 | 两个方法都在 swap 路径、体较大，依赖 `copyRenderPassColorToDrawableIfNeeded:` 内的 drawable 逻辑 |
| `mgl_draw_metal_port.m` | **0 / 1,973** | **没有任何 ObjC 方法**，全是 host-ops 适配函数 + `[host …]` 转发；它是 T5 之后最大的单块 ObjC 面。方向：把 host-ops 表改成纯 C 函数指针（表本身已是 `void *`），把其中调用的类别方法逐个转 C |
| `MGLRenderer+Binding.m` | 3 / 430 | 余 `bindMTLTexture*`(339) 与 `syncResourceBindingsForContext:`(27)，依赖深度 ≥3（见上表同型） |

> **结论（给下一轮的可执行指令）**：先做 **`+GPURecovery.m` 的整文件转 C**（C 头 + C 入口 + areas 已覆盖的 backend 句柄，
> 预计 −350 行、文件 17 → 16）；若中途发现依赖深度 ≥2，则退回"逐个方法转 C"并在日志里记录依赖表。
> `mgl_draw_metal_port.m` 与三厚块（`+RenderPass`/`+Texture`/`+Blit`）属于最后阶段：它们的方法彼此调用密集，
> 应先做"叶子方法"（只调 C 与 areas 的那些）再向上收口。

79. **P0-1 第二十五刀：`+GPURecovery.m` 的两个零依赖方法转 C（**−19 行**，§0.13 路线的第一步）**：
     ① 按 §0.13"按依赖深度挑刀"的方法，先从 `+GPURecovery.m` 里挑出**发送数与 ivar 数都是 0** 的两个方法：
     `-clearTextureCache`(14) → `mglRendererClearTextureCache`；
     `-getOptimalAlignmentForPixelFormat:`(7) → `mglRendererOptimalAlignmentForPixelFormat`，
     新 TU **`mgl_gpu_recovery.{h,c}`**；`+Texture.m` 里 **6 处** `[self getOptimalAlignmentForPixelFormat:…]`
     与 `+GPURecovery.m` 内 1 处 `[self clearTextureCache]` 改直调；私有头两条声明注释化。
     ② **该文件的依赖表（本轮实测，供下一刀直接用）**：
     - 零依赖：`clearTextureCache` ✅已转、`getOptimalAlignmentForPixelFormat:` ✅已转；
     - 仅依赖 `_gpuRecovery` + `[NSDate date]`：`recordGPUError`(10) / `recordGPUSuccess`(12) / `shouldSkipGPUOperations`(19)
       → 需要 1 个 `_gpuRecovery` 的 areas 指针 + 1 个 C 时间函数（`mglTraceNowSeconds()` 已存在）；
     - 依赖 `_renderPassManager` 的方法（`cleanupCommandBuffer` / `clearProblematicGPUState` / `commitCommandBufferWithAGXRecovery:`）
       → 需要 manager 的 `discardCurrentCommandBuffer` / `clearCurrentRenderEncoder` / `endCurrentRenderEncoder` /
       `commitCommandBufferTransaction` / `releaseDetachedCommandBufferIfOwned` 五个 C 入口；
     - `resetMetalState`(35) 还需 `_pipelineCache.resetCaches`（cut 19 已把 cache 对象放进 areas，可加 1 个 C 入口）与
       `_depthStencilState`、`_commandQueue`、`_isVirtualized` 三个 ivar。
     ③ **度量**：行数 **35,096 → 35,077**、语法 **2,006 → 1,999**（−7）、词汇 **3,912 → 3,910**；文件 17、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `be79747` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：按上面第 ② 条继续拆 `+GPURecovery.m`——先加 `_gpuRecovery` 的 areas 指针 + 用 `mglTraceNowSeconds()` 替 `[NSDate date]`，
     把 `recordGPUError` / `recordGPUSuccess` / `shouldSkipGPUOperations` 三个转 C（预计 −41 行）；再补 5 个 manager C 入口收口整文件。

80. **P0-1 第二十六刀：`recordGPUError` / `recordGPUSuccess` 转 C（**−22 行**，areas 再加一个"槽地址"字段）**：
     ① 按第 79 条的依赖表推进：两个方法的唯一 ObjC 成分是 `_gpuRecovery.commandRecoveryOwner` 与
     `[[NSDate date] timeIntervalSince1970]`。做法：
     - `MGLRendererStateAreas` 增加 **`void **gpu_recovery_command_owner`**——存的是**槽的地址**（不是值），
       与既有 `binding_state_owner` 同型：owner 在恢复过程中可能被换掉，必须在用点解引用；壳 TU 填
       `&r->_gpuRecovery.commandRecoveryOwner`（**areas 加字段不算新端口**）。
     - 时钟：Objective-C 用的是**墙钟** `[[NSDate date] timeIntervalSince1970]`，而 `mglTraceNowSeconds()` 是**单调钟**，
       两者 epoch 不同 → **不能用它替换**；C 侧改用 `clock_gettime(CLOCK_REALTIME)` 自算等价的秒数，
       行为与原来一致（这是一处容易"看起来等价其实不等价"的陷阱，记下来）。
    ② 新 C 入口 `mglRendererRecordGPUError` / `mglRendererRecordGPUSuccess` 落在 `mgl_gpu_recovery.c`；
     3 个文件的调用点改直调（`+Texture.m`（含 `weakSelf` 形式）、`+RenderPass.m`、`MGLRenderer.m`），
     私有头两条声明注释化。
     ③ **度量**：行数 **35,077 → 35,058**、语法 **1,999 → 1,997**、词汇 **3,910 → 3,906**；文件 17、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `9d4007c` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+GPURecovery.m` 还剩 6 个方法——`shouldSkipGPUOperations`(19) 只差 `[self clearProblematicGPUState]`
     （后者要 manager 的 `discardCurrentCommandBuffer` C 入口）；`cleanupCommandBuffer`(27) 要 3 个 manager 入口；
     `commitCommandBufferWithAGXRecovery:`(99) 要 2 个入口 + `@try` 语义（可由壳提供 guarded C 入口）；
     `resetMetalState`(35) 要 cache 的 `resetCaches` C 入口 + 3 个 ivar。**建议下一刀先补 manager 的 5 个 C 入口**
     （它们是同一个文件里的小方法，一次补完，之后 `+GPURecovery.m` 可整文件转 C → 文件 17 → 16）。

81. **P0-1 第二十七刀：manager 三个 C 入口 + `shouldSkipGPUOperations`/`clearProblematicGPUState` 转 C（**−34 行**）**：
     ① 新 TU **`mgl_render_pass_manager_ops.{h,c}`**：`mglRenderPassManagerEndCurrentRenderEncoder` /
     `…ClearCurrentRenderEncoder` / `…DiscardCurrentCommandBuffer`——三者原来只是 `_state`（= `areas.command`）上的
     几个 C 调用，C 侧用同一份逻辑（含本地 twin `mglRenderPassManagerSyncRuntimeOwners`、`mglRenderClearFboMatchCache`、
     `mglRenderDestroyMDIScratchOwner`）。**manager 自身仍用自己的方法**，C 调用者用这些入口——零端口。
     ② `+GPURecovery.m` 的 `-clearProblematicGPUState`(15) 与 `-shouldSkipGPUOperations`(19) 转 C
     （`mglRendererClearProblematicGPUState` / `mglRendererShouldSkipGPUOperations`）：后者复用第 80 刀加进 areas 的
     `gpu_recovery_command_owner` 槽地址 + 墙钟 helper。调用点（`+Texture.m`、`+RenderPass.m`）改直调，声明注释化。
     ③ **度量**：行数 **35,058 → 35,024**、语法 **1,997 → 1,994**、词汇 **3,906 → 3,899**；文件 17、shim 端口 13 不变；
     `+GPURecovery.m` 350 → **~295 行**（余 `validateMetalObjects` 75、`cleanupCommandBuffer` 27、
     `commitCommandBufferWithAGXRecovery:` 99、`resetMetalState` 35）。
     ④ **oracle**：旧库 = 提交 `6374e39` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+GPURecovery.m` 收口——`cleanupCommandBuffer`(27) 只差一个**由壳提供的 guarded C 入口**
     （把 `@try/@catch` 留在 ObjC، 形如 `mglPlatformShellGuardedCleanup(renderer, fn)`），`resetMetalState`(35) 需要
     cache 的 `resetCaches` C 入口 + `_depthStencilState`/`_commandQueue`/`_isVirtualized` 三个值，
     `commitCommandBufferWithAGXRecovery:`(99) 需要 `commitCommandBufferTransaction`/`releaseDetachedCommandBufferIfOwned`
     两个入口 + guarded 语义；补完即可**整文件转 C → 文件 17 → 16**。

82. **P0-1 第二十八刀：`@try/@catch` 交给壳的 guarded C 入口，`cleanupCommandBuffer` 转 C（**−27 行**）**：
     ① **新增模式（可复用）**：Objective-C 异常语义没有 C 形式，于是把"守护"留在**唯一壳 TU**里，C 只提供裸体：
     ```c
     int mglPlatformShellGuardedCall(void *renderer, const char *what, int (*body)(void *));
     ```
     壳里是 `@try { return body(renderer); } @catch (NSException *e) { fprintf(stderr, "MGL ERROR: Exception during %s: %s\n", …); return 0; }`。
     这样 `@try/@catch` 只出现在壳里，业务体是纯 C，且**不新增端口**（它是 C 入口，不是 renderer port）。
     ② `-cleanupCommandBuffer`(27) → `mglRendererCleanupCommandBufferBody`（纯 C，用第 81 刀的 manager 入口
     `Discard/End/Clear`），调用点（`+RenderPass.m` 3 处、`MGLRenderer.m` 1 处）统一写成
     `mglPlatformShellGuardedCall((__bridge void *)self, "command buffer cleanup", mglRendererCleanupCommandBufferBody)`。
     ③ **两处口径陷阱**：`MGL_COMMAND_BUFFER_STATUS_COMMITTED` 在 **`mgl_sync.h`**（不是 `mgl_render.h`）；
     `kMGLVerboseFrameLoopLogs` 是 ObjC 私头里的 `static const BOOL = NO`，C 侧写成 `if (0)` 并注明来源
     （**不要为了一个永远是 NO 的日志开关去把 ObjC 头拖进 C**）。
     ④ **度量**：行数 **35,024 → 35,015**、语法 **1,994 → 1,991**、词汇 **3,899 → 3,897**；文件 17、shim 端口 13 不变；
     `+GPURecovery.m` 剩余 **`validateMetalObjects`(75) · `resetMetalState`(35) · `commitCommandBufferWithAGXRecovery:`(99)**。
     ⑤ **oracle**：旧库 = 提交 `48e652d` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀（轮次预算内最后几刀）：① `resetMetalState`(35)——需 cache 的 `resetCaches` C 入口（cache 对象已在 areas）
     + `_depthStencilState`/`_commandQueue`/`_isVirtualized` 三个值（可照 `gpu_recovery_command_owner` 的"槽地址"模式）；
     ② `commitCommandBufferWithAGXRecovery:`(99)——需 `commitCommandBufferTransaction` /
     `releaseDetachedCommandBufferIfOwned` 两个 manager 入口 + `mglPlatformShellGuardedCall`；
     ③ `validateMetalObjects`(75)——需 `_device`/`_commandQueue`/`_isVirtualized` 与 `[NSDate date]`（墙钟 helper 已有）、
     `[self resetMetalState]`（C 化后即可）。**三项补完即可整文件转 C（文件 17 → 16）。**

83. **P0-1 第二十九刀：`resetMetalState` 转 C（**行数 −1 / 词汇 −4，语法 +2，如实记账**）**：
     ① **上一轮那次失败的正确解法**：`resetMetalState` 需要"重建命令队列后回读 `_commandQueue`"。第 40 轮试图用 areas 槽地址读它，
     失败原因是 **`_commandQueue` / `_backend` 这些 ivar 只对"类体内的代码"可见**（方法里的 `_backend` 是 `self->_backend`），
     **不是 `@public`**，所以 C 侧 `r->_backend` 根本无法编译。本刀改用**方法 + 壳转发**：
     - `MGLRenderer.m` 加 `- (int)mglRecreateCommandQueue`（体里用 `_backend` / `_commandQueue`，方法体内合法），
       声明放进 `MGLRenderer+GPURecovery_Private.h`；
     - 壳 TU 加 C 入口 `mglPlatformShellRecreateCommandQueue(void *)`（壳是 ObjC，可以发消息）——
       **C 入口不是端口，端口计数不变**；
     - `mglRendererResetMetalState`（`mgl_gpu_recovery.c`）用 `mglPlatformShellGuardedCall` 跑 cleanup 体、
       用上面的壳入口重建队列、用 `mglPipelineCacheResetCaches`（壳里新加的 C 入口，cache 对象来自 areas）复位 cache、
       再调已有的 `mglRendererClearTextureCache`；`NSLog` → `fprintf`。
     ② 三个调用点（`+RenderPass.m` 3、`+GPURecovery.m` 1）改直调；`+GPURecovery.m` 私有头声明注释化。
     **`+GPURecovery.m` 只剩 2 个方法**：`validateMetalObjects`(75) 与 `commitCommandBufferWithAGXRecovery:`(99)。
     ③ **度量（如实）**：行数 **35,015 → 35,014**、语法 **1,991 → 1,993（+2）**、词汇 **3,897 → 3,893**。
     语法上升是因为"方法 + 壳转发"两条 ObjC 声明/发送抵掉了被删的 35 行方法体里的大部分语法——**不宣称语法收益**，
     本刀价值在于把业务体搬进 C、并把 `_commandQueue` 的可见性陷阱固化下来。
     ④ **oracle**：旧库 = 提交 `6a2dcec` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+GPURecovery.m` 收口最后两个方法——`commitCommandBufferWithAGXRecovery:`(99) 需 manager 的
     `commitCommandBufferTransaction` / `releaseDetachedCommandBufferIfOwned` 两个 C 入口 + `mglPlatformShellGuardedCall`；
     `validateMetalObjects`(75) 需 `_device`/`_isVirtualized` 的**方法+壳转发**式入口（照本刀模式，不要再用 areas 槽地址）与
     C 化的 `mglRendererResetMetalState`（本刀已具备）。补完即可**整文件转 C → 文件 17 → 16**。

84. **P0-1 第三十刀：`validateMetalObjects` 转 C（**−51 行 / 词汇 −22**）**：
     ① 按第 83 条的"方法 + 壳转发"模式（不再用 areas 槽地址）：`MGLRenderer.m` 加两个方法
     `-mglMetalDevicePointer` / `-mglMetalObjectsPresent`（方法体内用 `_device` / `_commandQueue` 合法），
     壳 TU 加 C 入口 `mglPlatformShellMetalDevice` / `mglPlatformShellMetalObjectsPresent`；
     `mglRendererValidateMetalObjects` 在 `mgl_gpu_recovery.c` 里用 `mglPlatformShellGuardedCall` 包住裸体
     （`@try/@catch` 留在壳），`@available(macOS 11.0, *)` → C 的 `__builtin_available`，`[NSDate date]` → 墙钟 helper，
     `NSLog` → `fprintf`；`consecutiveGpuErrors` / `lastErrorTime` / `throttleWindow` / `maxErrorsPerWindow` 改为 C 静态量。
     ② 调用点（`+RenderPass.m`）改直调，私有头声明注释化。**`+GPURecovery.m` 只剩 1 个方法**
     （`commitCommandBufferWithAGXRecovery:` 99 行，含嵌套 `@try/@catch/@finally`）。
     ③ **度量**：行数 **35,014 → 34,963**、语法 1,993（持平）、词汇 **3,893 → 3,871**；文件 17、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `618e701` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀（收口 `+GPURecovery.m` → 文件 17 → 16）：`commitCommandBufferWithAGXRecovery:` 需要
     ① manager 的 `commitCommandBufferTransaction` / `releaseDetachedCommandBufferIfOwned` 两个 C 入口
     （都只动 `_state`，照 `mgl_render_pass_manager_ops.c` 的写法）；② 一个**带 `@finally` 语义的 guarded 入口**
     （壳里 `mglPlatformShellGuardedCallWithFinally(renderer, what, body, finally_fn)`），因为原方法把
     `releaseDetachedCommandBufferIfOwned:` 放在 `@finally` 里，必须保证异常路径也执行；
     ③ `_deviceResetRequested` 的置位——先确认它是否在 `MGLRendererCoreState`（cut 54 搬进去的四个 `_Atomic` 通道之一），
     若是则 `areas.core->deviceResetRequested` 直接写，若否则再加一个方法+壳转发。

85. **P0-1 第三十一刀：`MGLRenderer+GPURecovery.m` 整文件删除（**文件 17 → 16**）**：
     ① 收口最后一刀式的三件事全部落地：
     - manager 两个 C 入口：`mglRenderPassManagerCommitCommandBufferTransaction` /
       `mglRenderPassManagerReleaseDetachedCommandBufferIfOwned`（都只动 `_state`，写进 `mgl_render_pass_manager_ops.c`）；
     - 壳里新增 **带 ctx + finally 的守卫** `mglPlatformShellGuardedCallCtx(renderer, what, body, ctx, finally_fn)`
       （`@try/@catch/@finally` 的 C 对手；`ctx` 传 command buffer，于是 finally 里能释放**正确**的那个 submission）；
     - `_deviceResetRequested` **确认在 `MGLRendererCoreState`**（第 54 刀搬进去的 `_Atomic bool`）→ C 侧直接
       `atomic_store_explicit(&areas.core->deviceResetRequested, …)`，无需新入口。
     `commitCommandBufferWithAGXRecovery:`(99) → `mglRendererCommitCommandBufferWithAGXRecovery` + 裸体
     `…Body` + finally 半片；文件里只剩注释与空 category → **`git rm` 整个 `.m`**。
     ② **三个 C/ObjC 边界坑（记一次）**：`mglShouldTraceCall` 是 ObjC 私头里的 `static inline`（C 侧要么自带同义实现，
     要么用 C 常量）→ 本刀把 `kMGLDiagnosticStateLogs` 以 C 常量镜像并复刻 80/500 调度；`id` 实参必须逐个
     `(__bridge void *)`（我的批量脚本曾把已是 `void *` 的实参加成双重桥接，编译器报
     `incompatible types casting 'void *' to 'void *'`）；`__builtin_available(macOS 11.0, *)` 是 `@available` 的 C 形式。
     ③ **度量**：**文件 17 → 16**、行数 **34,963 → 34,845**、语法 **1,993 → 1,986**、词汇 **3,871 → 3,863**；
     shim 端口仍 13（居唯一壳 TU）。
     ④ **oracle**：旧库 = 提交 `a915c93` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：按 §0.12/§0.13 继续挑薄文件——`+DrawStageHost.m`(≈217，余 4 方法，含 block 与 `self->ctx` 写)、
     `+SwapDiagnostics.m`(556，2 方法)、`+Binding.m`(431，`bindMTLTextureLocked:` 339 + `syncResourceBindings…` 27)、
     `MGLPipelineCache.m`(446)、`MGLRenderPassManager.m`(≈500)；最后是 `mgl_draw_metal_port.m`(1,973, 0 方法) 与三厚块。

### 0.14 私有 ivar 的三种可达方式（第 83–85 刀固化；**下一阶段最重要的规则**）

把方法体搬进 C 时，最先撞上的不是算法而是"renderer 的 ivar 拿不到"。实测有三种，按代价从低到高：

| 情形 | 做法 | 例 |
|---|---|---|
| ivar **在头文件里可见**（`_core` / `_backend` / `_ctx` / `_batching` / `_pipelineCache` / `_gpuRecovery` / `_tessellation` / `_bindingStateOwner` / `_resourceFallback`） | **加进 `MGLRendererStateAreas`**（值或"槽地址"）——**加字段不算新端口** | 第 73、80 刀 |
| ivar **只在类体内可见**（`_device` / `_commandQueue` / `_isVirtualized` / `_mglInMSSampleDrawLoop` 等） | **方法 + 壳转发**：在 `MGLRenderer.m` 加一个方法（方法体里 `_device` 合法），壳 TU 加一个 **C 入口**转发消息（C 入口不是端口） | 第 83–85 刀 |
| 需要 `@try/@catch/@finally` | **壳提供的守卫**：`mglPlatformShellGuardedCall(renderer, what, body)` 与带 ctx/finally 的 `mglPlatformShellGuardedCallCtx(...)`；业务体保持纯 C | 第 82、85 刀 |

**每次搬完必须做的事**：① 立刻 `make -j4 lib`（clang 是唯一裁判）；② 跑 `/private/tmp/run_ab<N>.sh {new,old}` + `ab_full.py`（真旧库、`cmp` 校验）；
③ CTS 七簇 diff；④ `bash scripts/objc_zero.sh` 记数；⑤ 只 `git add` 自己的路径。**反例警告**：不要用"裸名字出现次数"当 liveness 证据
（第 71/77 条各踩一次）；死方法一律用 `scripts/objc_dead_methods.py` 判定。

### 0.15 当前 16 个 `.m`（第 43 轮实测）与建议顺序

| 文件 | 行数 | 备注 |
|---|---|---|
| `MGLRenderer+RenderPass.m` | ~7.1k | 三厚块之一；`processGLStateLocked:` 是最大单块 |
| `MGLRenderer+Texture.m` | ~6.4k | 三厚块之一 |
| `MGLRenderer.m` | ~4.7k | 仍持 `mglRenderer*` C 桥与少量内部方法；**现在是"方法+壳转发"的落点** |
| `MGLRenderer+Blit.m` | ~4.0k | 三厚块之一 |
| `MGLRenderer+BindingState.m` | ~2.9k | 编码器绑定族（对应 shim 4 个端口） |
| `MGLRenderer+Tessellation.m` | ~2.2k | tess 捕获/描述符残留 |
| `mgl_draw_metal_port.m` | ~1.97k | **0 方法**，纯 host-ops 适配；最后阶段 |
| `MGLRenderer+Compute.m` | ~1.2k | 计算路径 |
| `MGLRenderer+Buffer.m` | ~0.8k | `mapGLBuffersToMTLBufferMap:stage:` 链 |
| `MGLRenderer+Lifecycle.m` | ~0.66k | T5 第二步候选（KVO/通知回调必须保留） |
| `MGLRenderer+SwapDiagnostics.m` | ~0.56k | 2 方法，均在 swap 路径 |
| `MGLRenderPassManager.m` | ~0.50k | manager 类（`_state` 已 C 化） |
| `MGLRenderer+Binding.m` | ~0.43k | 余 `bindMTLTextureLocked:` 339 + `syncResourceBindings…` 27 |
| `MGLPipelineCache.m` | ~0.45k | cache 类，方法多为 `mglRender*PipelineCacheOwner*` 转发 |
| `MGLRenderer+DrawStageHost.m` | ~0.22k | 余 4 方法：MS 循环族（block）、`runVertexCaptureSession:`（写 `self->ctx`）、`bindCullDistanceEmulationBuffers:` |
| `MGLPlatformRendererShell.m` | ~0.51k | **唯一壳 TU**：平台壳 + 13 端口 + 各 guarded/转发 C 入口 |

> **建议顺序**：`+DrawStageHost.m`（最小、4 方法）→ `+Binding.m` 的 `syncResourceBindingsForContext:` → `MGLPipelineCache.m`
> → `MGLRenderPassManager.m` → `+SwapDiagnostics.m` → `mgl_draw_metal_port.m` → 三厚块（`+Compute`/`+Buffer`/`+BindingState`/`+Tessellation`
> 在过程中顺带收）。每一步都按 §0.14 的三条路线取最短路径。

### 0.16 `+DrawStageHost.m` 四个方法的依赖表（第 44 轮实测，下一刀可直接照做）

| 方法 | 行数 | 消息发送 | ivar / 依赖 | 转换路径（按 §0.14） |
|---|---|---|---|---|
| `bindCullDistanceEmulationBuffers:` | **85** | **0** | `_backend`（areas ✓）、`_VERTEX_SHADER`（GL 常量 ✓）、`_tessellation.cullDistanceCaptureFirstInstance` / `…InstanceStride`（**头里可见**） | **最短**：给 areas 加两个 `uint32_t` 字段（`tess_cull_capture_first_instance` / `…_instance_stride`），壳里从 `r->_tessellation` 填；体里 `recordLastBoundVertexBuffer:` / `invalidateLastBoundVertexBufferAtIndex:` 已在第 22 刀变成 C（`mglBindingRecordLastBoundVertexBuffer` / `…AtIndex`），可直接调 |
| `runVertexCaptureSession:` | 19 | **0** | 只有 `self->ctx = drawCtx;`（**写 renderer 的 ctx**，`areas.ctx` 是副本、写了不生效） | 方法 + 壳转发：`MGLRenderer.m` 加 `- (void)mglSetDrawContext:(GLMContext)ctx`（体内 `ctx = drawCtx;`），壳加 C 入口 `mglPlatformShellSetDrawContext(void *, GLMContext)` |
| `runEmulatedMSSampleDrawLoopIfNeeded:` | 26 | `[self endRenderEncodingLocked]` | `_mglInMSSampleDrawLoop` / `_mglForcedMSSampleId` / `_mglMSSamplePlaneOffset`（私有）、block 参数 `void (^)(void)` | 需先 C 化 `endRenderEncodingLocked`（本身要 3 个 manager 入口 + guarded 入口，见第 81 条），再把 block 换成 `fn + ctx` 并在调用点改成函数指针 |
| `broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:` | 39 | `[self endRenderEncodingLocked]`、`[self newCommandBufferLocked]` | `_mglInMSSampleDrawLoop`（私有）、`_renderPassManager`（areas.command ✓） | 同上，另需 `newCommandBufferLocked`（大方法） |

> **⚠️ 第 45 轮实测修正（重要）**：`bindCullDistanceEmulationBuffers:` **"零发送"不等于"机械可搬"**。用脚本抽取+替换后编译报出四类问题，
> 全部需要手工处理，脚本化搬运在本例**失败并已整刀回退（未提交）**：
> ① 体里有 **`id captureBuffer`** 与 **3 处 `__bridge`**（不是"零 ObjC 成分"，只是"零消息发送"）；
> ② `mglRendererResolveVertexAttribBinding` 的 C 声明与 ObjC 头里的**签名不一致**（`conflicting types`），C 侧得按真实现写；
> ③ 依赖的 `mglDrawSupportEncodeContextIsActive` 声明在 **ObjC 头** `MGLRenderer+DrawSupportUtil.h`（含 Foundation，C 不能 include）→ 需本地 extern 并注意返回类型；
> ④ `MGLEncodeContext` / `MGLResolvedVertexAttribBinding` 分别在 `mgl_encode_context.h` / `mgl_vertex_attrib_binding.h`（容易漏）。
> **规则：判断一个方法能否搬，看的是"有没有 ObjC-only 语法（`id`/`__bridge`/block/消息）"，不是"有没有消息发送"**；
> 手工逐个搬（每次编译验证）而不是正则批处理。
>
> **建议下一刀**：改从 `runVertexCaptureSession:`(19 行、**真正零响应**) 开始——它只有一个 `self->ctx` 写入需要"方法 + 壳转发"；
> 随后转 `runVertexCaptureSession:`(19 行) 用"方法 + 壳转发"补一个 ctx 写入；**MS 循环两方法放最后**，
> 因为它们依赖 `endRenderEncodingLocked`（第 81 条评估为依赖深度 ≥3）。

### 0.17 `runVertexCaptureSession:` 的精确转换配方（第 46 轮实测，照着做即可）

```objc
- (BOOL)runVertexCaptureSession:(GLMContext)drawCtx capture:(id)capture params:(const uint32_t *)params
{
    if (!drawCtx || !capture || !params) return NO;
    self->ctx = drawCtx;                                  // ← 唯一的 ObjC-only 副作用
    MGLTessCaptureSessionHostOps ops = {
        .ctx = drawCtx, .renderer = (__bridge void *)self, // ← id/__bridge 各一处
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll, /* …5 个函数指针，都是 C */
    };
    return mglTessRunCaptureSession((__bridge void *)capture, params, &ops) ? YES : NO;
}
```

**步骤（每步后 `make -j4 lib`）**：
1. `MGLRenderer.m` 加 `- (void)mglSetDrawContext:(GLMContext)drawCtx { ctx = drawCtx; }`，声明进 `MGLRenderer+Draw_Private.h`
   （**必须在方法体里写，`ctx` 是私有 ivar**）。
2. 壳 TU 加 C 入口 `void mglPlatformShellSetDrawContext(void *renderer, GLMContext ctx)` → `[r mglSetDrawContext:ctx]`。
3. 新 C TU（或并入 `mgl_draw_support.c`）实现
   `int mglDrawRunVertexCaptureSession(void *renderer, GLMContext drawCtx, void *capture, const uint32_t *params)`：
   判空 → 调壳入口写 ctx → 组装 `MGLTessCaptureSessionHostOps`（`renderer = (void *)renderer`）→ 调
   `mglTessRunCaptureSession(capture, params, &ops)`。
4. 调用点（`mgl_draw_metal_port.m` 1 处，C）改直调；删除 ObjC 方法与 `MGLRenderer+Draw_Private.h` 声明。
**收益预估**：−19 行方法体、+1 方法（3 行）、+1 壳 C 入口（5 行）→ **行数小幅净减，语法基本持平**；价值主要是把"写 ctx"这一副作用
集中到壳入口，为后续两刀（`bindCullDistance…`、MS 循环族）铺路。

> **注意**：本文件剩下 3 个方法（`bindCullDistanceEmulationBuffers:` 85 行含 `id`+3×`__bridge`、MS 循环族含 block）
> 都**不能**用正则批处理（第 45 条的失败），必须手工逐段搬并逐步编译。

86. **P0-1 第三十二刀：`runVertexCaptureSession:` 其实是死方法（**−19 行**）**：
     ① 按第 46 条（§0.17）准备转换配方时，先做了一遍全树核查：`runVertexCaptureSession` 在**整个仓库**（含 `.m/.mm/.c/.cpp/.h`，
     排除 build）只出现在**它自己的定义**与 `MGLRenderer+Draw_Private.h` 的**声明**里——**没有任何调用者**。
     于是不走"转换"而直接**删除方法与声明**（死代码不该换个语言继续养，第 74 条的规则）。
     ② **工具盲点（记一次，重要）**：`scripts/objc_dead_methods.py` **没有**把它报为候选，因为该脚本的 `bare` 检查会把
     "私有头里的声明"也算作一次引用（保守设计）。**结论：脚本的"零候选"只保证"没有'连声明都没有'的死方法"；
     遇到"只有声明、没有调用"的情形，仍须手工做一次"定义 vs 调用点"的核查**（本轮即如此发现）。
     ③ **度量**：行数 **34,845 → 34,826**、语法 **1,986 → 1,984**、词汇 **3,863 → 3,859**；文件 16、shim 端口 13 不变；
     `+DrawStageHost.m` 203 → **185 行**（余 3 方法：`bindCullDistanceEmulationBuffers:` 85、MS 循环族 26+39）。
     ④ **oracle**：旧库 = 提交 `88d35fd` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+DrawStageHost.m` 余 3 方法都含 ObjC-only 语法（`id`/`__bridge`/block 或 `endRenderEncodingLocked` 依赖），
     必须手工搬并逐步编译（第 45 条失败教训）；建议顺序：`bindCullDistanceEmulationBuffers:`（先补 areas 两个
     `_tessellation.cullDistanceCapture*` 字段，再逐段处理 `id captureBuffer` 与 3 处 `__bridge`）→ MS 循环族（先 C 化
     `endRenderEncodingLocked`）。

### 0.18 "只有声明、无调用"候选清单（第 48 轮扫描；**仅为候选，禁止直接删**）

第 86 条发现 `scripts/objc_dead_methods.py` 的 `bare` 检查会把"私有头里的声明"当引用，于是本轮用一版**忽略声明**的启发式扫了全仓，
得到 **34 个"无调用形态引用"** 的方法。**其中大部分是假阳性**，典型三类：

| 假阳性类型 | 例子 | 实际情况 |
|---|---|---|
| 属性语法读取 getter | `MGLPipelineCache.state` / `.device`、`MGLRenderPassManager.state` | `obj.state` / `.device = x` 是调用，启发式看不见 |
| 框架回调 / 生命周期 | `initWithView:`、`initWithPSODedupEnabled:`、`dealloc`、`observeValueForKeyPath:` | 由运行时/KVO 调用，源码里没有调用点 |
| 跨行或复杂接收者的发送 | `mglBackendWillDestroy:`、`mglTextureForDrawable:`、`performOperation:` 等 | `[renderer …]` 的调用点在别处，或发送跨行 |

**因此本轮没有删除任何东西**（第 45 条那次教训：启发式只配当候选）。真正值得下一步逐个核查的**次级候选**（都是 4–9 行的小方法，
且名字像"GL→Metal 旧入口"）：`MGLRenderer.m` 的 `mtlFlush:`(4) · `mtlReadBackBuffer:`(6) · `mtlDeleteMTLObj:`(6) ·
`mtlBufferSubData:`(7) · `mtlMapUnmapBuffer:`(7) · `mtlFlushMappedBufferRange:`(6) · `mtlStencilOpForGLOp:`(8) ·
`blendFactorFromGL:`(13) · `blendOperationFromGL:`(13)；`MGLRenderPassManager.m` 的 `beginCommandBufferCommit`(5) ·
`clearPendingEvent`(7) · `commitDetachedCommandBufferIfOwned:`(15) · `appendSyncToCurrentCommandBuffer:`(15) ·
`preparePendingEventWithDevice:`(17)；`MGLRenderer+RenderPass.m` 的 `newCommandBufferAndRenderEncoder`(37) ·
`mtlInvalidateRenderPass:`(45)；`MGLRenderer+Compute.m` 的 `mtlDispatchComputeLocked:`(26)；`MGLPlatformRendererShell.m` 的
`mglTextureForDrawable:`(4) · `performOperation:`(21)。

**核查方法（每个候选 3 步，缺一不可）**：① `grep -rn "<sel>" --include='*.m' --include='*.mm' --include='*.c' --include='*.cpp' --include='*.h'`，
逐条看清命中是"调用"还是"声明/注释"；② 查属性语法（getter 看 `.name`，setter 看 `.name =`）；③ 查 `@selector(<sel>)` 与函数指针表
（`mgl_draw_metal_port.m` 的 host-ops 表）、以及 `respondsToSelector`。**三步都确认无调用，才可删除**。

87. **P0-1 第三十三刀：按 §0.18 三步法核实并删除 9 个死方法（**−67 行**）**：
     ① 对 §0.18 的次级候选逐个跑"三步核查"：**9 个候选的全部 grep 命中都只是"自己的定义 + 私有头声明 + 无关注释"**
     （例：`mtlFlush:` 在 `mgl_gl_extensions.c:5782` 的命中是**注释**；`mtlFlushMappedBufferRangeLocked:` 是**另一个方法名**），
     既无属性语法调用、也无 `@selector`/host-ops 表引用 → 判定为死代码并删除：
     `MGLRenderer.m` 的 `mtlReadBackBuffer:`(4) · `mtlFlush:`(4) · `mtlDeleteMTLObj:`(6) · `mtlBufferSubData:`(6) ·
     `mtlMapUnmapBuffer:`(7) · `mtlFlushMappedBufferRange:`(6) · `mtlStencilOpForGLOp:`(8) · `blendFactorFromGL:`(13) ·
     `blendOperationFromGL:`(13)；两个私有头里的对应声明一并清理。
     ② **刻意不动**：`mtlDispatchComputeLocked:`(26) 虽然也被启发式扫成候选，但 `+Compute.m:134` 有
     `[renderer mtlDispatchComputeLocked:…]` 的**真实调用**——这正是"启发式只配当候选"的例子，已写进文档。
     ③ **度量**：行数 **34,826 → 34,759**、语法 **1,984 → 1,979**、词汇 **3,859 → 3,856**；文件 16、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `64d5f19` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：§0.18 里还剩 `MGLRenderPassManager.m` 的 5 个候选（`beginCommandBufferCommit`(5) · `clearPendingEvent`(7) ·
     `commitDetachedCommandBufferIfOwned:`(15) · `appendSyncToCurrentCommandBuffer:`(15) · `preparePendingEventWithDevice:`(17)）与
     `+RenderPass.m` 的 `newCommandBufferAndRenderEncoder`(37) · `mtlInvalidateRenderPass:`(45)、`MGLPlatformRendererShell.m` 的
     `mglTextureForDrawable:`(4) · `performOperation:`(21)——**同样先跑三步核查再删**。

88. **P0-1 第三十四刀：§0.18 余下候选再删 8 个（**−145 行**，单刀最大死代码清理）**：
     ① 三步核查结果：
     - **死**（全部命中 = 定义 + 头声明 [+ 无关注释]）：`MGLRenderPassManager.m` 的 `beginCommandBufferCommit`(5) ·
       `clearPendingEvent`(7) · `commitDetachedCommandBufferIfOwned:`(15) · `appendSyncToCurrentCommandBuffer:`(15) ·
       `preparePendingEventWithDevice:`(17)；`+RenderPass.m` 的 `newCommandBufferAndRenderEncoder`(37，
       命中只剩自身 NSLog 文案与 `mgl_frame_activity.h` 的一处注释) · `mtlInvalidateRenderPass:`(45，
       命中只剩 `+Blit.m` 的注释与 `MGLRenderer.m` 的过时注释)；壳的 `mglTextureForDrawable:`(4)。
     - **活**：`MGLPlatformRendererShell.performOperation:` —— **`test_legacy_compat/test_metalcpp_smoke.mm` 里有 2 处真实调用**
       （`[shell performOperation:…]`），**保留**。这条再次印证"候选必须逐个核查"。
     ② 方法与三个头里的声明一并删除。
     ③ **度量**：行数 **34,759 → 34,614**、语法 **1,979 → 1,973**、词汇 **3,856 → 3,839**；文件 16、shim 端口 13 不变；
     `MGLRenderPassManager.m` 524 → **465 行**、`+RenderPass.m` 7,099 → **7,017 行**。
     ④ **oracle**：旧库 = 提交 `7ab0dec` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：§0.18 的候选已全部处理完（9 + 8 删、2 保留）。后续回到**结构性**路线（§0.15）：`+DrawStageHost.m` 余 3 方法
     （需手工搬 + areas 两个字段）→ `+Binding.m` 的 `bindMTLTextureLocked:` → `MGLPipelineCache.m` → `+SwapDiagnostics.m`
     → `mgl_draw_metal_port.m`（0 方法、1,973 行）→ 三厚块。

89. **P0-1 第三十五刀：删掉上一批"解锁壳"的 4 个 Locked 变体（**−50 行**）**：
     ① 第 33 刀删掉了 `mtlDeleteMTLObj:` / `mtlBufferSubData:` / `mtlMapUnmapBuffer:` / `mtlFlushMappedBufferRange:`
     四个**解锁壳**后，它们的 `…Locked` 本体就成了孤儿。用同一套三步核查确认：四个名字的全部命中只有
     **自身定义 + `MGLRenderer+RenderPass_Private.h` 声明**（无发送、无 `@selector`、无 host-ops 引用）→ 全部删除：
     `mtlDeleteMTLObjLocked:`(10) · `mtlBufferSubDataLocked:`(12) · `mtlMapUnmapBufferLocked:`(16) ·
     `mtlFlushMappedBufferRangeLocked:`(12)，声明同步清理。
     ② **本刀来源**：重跑"忽略声明"的候选扫描，专门看**上一批删除后新产生的孤儿**——这是一条可复用的收尾动作
     （"删一层壳后，下一轮先扫一次它解锁出来的本体"）。
     ③ **度量**：行数 **34,614 → 34,564**、语法 1,973（持平）、词汇 **3,839 → 3,836**；文件 16、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `f8989a2` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：候选扫描里剩下的都是**已知假阳性**（`.state` / `.device` / `.pipelineState` 属性语法、
     `performOperation:`（测试目录有调用）、`mtlDispatchComputeLocked:`（`+Compute.m:134` 有真实调用）、
     `mglBackendWillDestroy:`（同文件内 `[renderer …]` 调用））——**除非逐个手查，否则不要再动**。
     后续回到结构性路线（§0.15）：`+DrawStageHost.m` 余 3 方法 → `+Binding.m` 的 `bindMTLTextureLocked:`
     → `MGLPipelineCache.m` → `+SwapDiagnostics.m` → `mgl_draw_metal_port.m` → 三厚块。

### 0.19 目标现状总览与"下一刀"清单（第 52 轮实测）

**已达成**：T0（空 TU 清零）、T1（纯 C 文件改名）、T4 主体（**端口 43 → 13**，且 **ObjC 壳 TU 唯一化**：端口与平台壳同处
`MGLPlatformRendererShell.m`）、`+Batch` 整簇与 `+Draw.m`/`+DrawSupport.m`/`+VertexLayout.m`/`+GPURecovery.m` 四个文件消失、
死代码两批共清 17 个方法（−262 行）。

**未达成**（终态要求 `MGL/` 内零 `.m`）：**16 个 `.m` / 34,553 行 / 1,973 语法 / 3,836 词汇**。剩余 16 个文件的清单与规模见 §0.15。

**下一刀候选（按 §0.14 的三种路线取最短）**：

| 目标 | 行数 | 路线 | 预估 |
|---|---|---|---|
| `+DrawStageHost.m` 的 `bindCullDistanceEmulationBuffers:` | 85 | areas 加两个 `_tessellation.cullDistanceCapture*` 字段；体里 `id`/3×`__bridge` **手工**改 `void *` | −85 |
| `+DrawStageHost.m` 的 MS 循环族（2 个方法） | 26+39 | 先 C 化 `endRenderEncodingLocked`（依赖 ≥3）→ 再把 block 换 `fn+ctx` | −65（分两步） |
| `+Binding.m` 的 `syncResourceBindingsForContext:` | 27 | 依赖 10 处发送（多为已 C 化的绑定步骤）→ 逐个替换 | −27 |
| `MGLPipelineCache.m` | 446 | 方法多为 `mglRender*PipelineCacheOwner*` 转发；`state`/`device` 是属性 getter（会假阳性） | 可能整文件 |
| `MGLRenderer+Binding.m` 的 `bindMTLTextureLocked:` | 339 | 依赖 `createMTLTextureFromGLTexture` / `createFallbackMTLTexture` / `endRenderEncodingLocked` / `NSDate` | 分多刀 |
| `+SwapDiagnostics.m` | 556 | 2 方法都在 swap 路径，依赖 `_drawable`（私有 ivar → 方法+壳转发） | 分两步 |
| `mgl_draw_metal_port.m` | 1,973 | **0 方法**，纯 host-ops 适配；把表改成纯 C 函数指针 + 被调类别方法逐个 C 化 | 最后阶段 |
| 三厚块（`+RenderPass` 7,017 / `+Texture` 6.4k / `+Blit` 4.0k） | ~17.5k | 先"叶子方法"（只调 C 与 areas 的），再向上收口 | 最后阶段 |

**两条工具盲区（必须记住）**：① `scripts/objc_dead_methods.py` 的 `bare` 检查把"头声明"当引用 → **"只有声明、无调用"的方法它不报**，
需手工"定义 vs 调用点"核查（第 86 条）；② 反过来的启发式（忽略声明）会产生 3 类假阳性——属性语法、框架回调、跨行/复杂接收者
（第 48 条），**每删一个都必须跑"三步核查"**（第 88 条）；已知仍会假阳性的名字：`.state` / `.device` / `.pipelineState` /
`performOperation:` / `mtlDispatchComputeLocked:` / `mglBackendWillDestroy:`。

90. **P0-1 第三十六刀：删掉 `mglSetSwapInterval:`（**−11 行**），并把"不能删"的判据固化**：
     ① 三步核查：`mglSetSwapInterval` 的全部命中只有**定义 + `MGLPlatformRendererShell.h` 声明**，无属性语法、无 `@selector`、
     无 host-ops 引用 → 判死并删除（同时删声明）。
     ② **同批核查里"看似也死、实际不能删"的五类**（**写进文档，避免下轮重复踩**）：
     - **公开 API**：`initMGLRendererFromContext:` / `createMGLRendererFromContext:` 声明在 **public 的 `MGLRenderer.h`**，
       仓库内没有调用不代表外部使用者没有 → **保留**；
     - **属性 getter/setter**：`isBinaryArchiveEnabled`（`getter=isBinaryArchiveEnabled` 的只读属性）、`setPipelineState:`、
       `setDevice:` —— 可通过 `.binaryArchiveEnabled` / `.pipelineState = ` / `.device = ` 到达，扫描看不见 → **保留**；
     - **同文件内不同接收者的调用**：`mglBackendWillDestroy:` 在 `MGLRenderer+Lifecycle.m:38` 有 `[renderer mglBackendWillDestroy:backend]` → **保留**；
     - **测试目录调用**：`performOperation:` 在 `test_legacy_compat/test_metalcpp_smoke.mm` 有 2 处 → **保留**；
     - **被其它方法真实调用**：`mtlDispatchComputeLocked:`（`+Compute.m:134`）→ **保留**。
     ③ **度量**：行数 **34,564 → 34,553**、语法/词汇持平（1,973 / 3,836）；文件 16、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `02e0110` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ **本轮结论（重要）**：**死代码这条线已接近枯竭**——剩下的候选全部落在上述五类里。后续必须转回**结构性**路线（§0.19 表），
     即"手工搬方法体 + areas 字段/方法+壳转发/守卫"三件套；详见 §0.14 与 §0.17 的配方。

### 0.20 "廉价结构刀"也已枯竭（第 54 轮扫描结论）

对剩余 16 个 `.m` 做了一次"≤15 行 + 无消息发送 + 无 `id`/`__bridge`/block"的扫描，命中 34 个方法，但**全部属于"不值得单独动刀"的两类**：

1. **访问器**（4–9 行）：`isBinaryArchiveEnabled` / `mglSwapInterval` / `mglDrawableTexture` / `mglHasMetalLayer` /
   `mglMetalLayerDrawableSize` / `mglMetalLayerFrame` / `setRuntimeContext:` / `hasLastSubmittedCommandBuffer` /
   `waitForLastSubmittedCommandBuffer:` / `resetMDIScratch` / `clearFboMatchCache` / `endCommandBufferCommit` / `dealloc` 等
   —— 它们多是属性访问或由框架调用，**删不得**（§0.20 第 90 条五类），转换也无收益（4 行换成 4 行 + 调用点改动）。
2. **一行的 C 转发**（4–15 行）：`checkForDirtyBufferData:` / `updateDirtyBaseBufferList:` / `updateDirtyBuffer:` /
   `commitCommandBufferTransaction:` / `releaseDetachedCommandBufferIfOwned:` / `clearStageBindingCopyBack(s)` 等
   —— 体本身已是"一次 C 调用"，但**调用点都是 ObjC 发送**，逐个改直调是纯 churn（行数不降、语法不降），
   收益要等它们所在的整块逻辑一起下沉时才能兑现。

**结论**：至此**两种"低垂果实"（死代码 / 小方法）都已摘完**。后续必须做**整块搬迁**——即 §0.19 表里的
`+DrawStageHost.m` 剩余 3 方法（85/26/39 行）、`bindMTLTextureLocked:`(339)、`MGLPipelineCache.m`(446)、
`+SwapDiagnostics.m`(556)、`mgl_draw_metal_port.m`(1,973) 与三厚块；这些都需要"手工逐段搬 + 每段编译 + 整轮 A/B/CTS"，
**建议在上下文充裕的新会话里成批推进**，并严格按 §0.14 的三种路线与 §0.17 的配方执行。

### 0.21 第 55 轮复验与"下一步必须整块搬"的最终确认

- **复验（提交 `24bc954`，未改代码）**：`make -j4 lib` **0 error**；`test_regression` **92/0/2**、`test_batch_icb: ok`；
  `objc_zero.sh`：**16** 个 `.m` / 空 TU **0** / **34,553** 行 / 语法 **1,973** / 词汇 **3,836**；工作区干净。
- **本轮再次核实的"看似可删、实则活"名单**（补进第 90 条的五类之外）：`updateDirtyBuffer:`（`+Tessellation.m` 2 处、
  `+BindingState.m` 1 处、`+Compute.m` 1 处等 **5 处真实调用**）、`checkForDirtyBufferData:` / `updateDirtyBaseBufferList:` /
  `clearStageBindingCopyBack(s)` / `getVertexBufferIndexWithAttributeSet`（均有头声明 + 调用）。**结论同 §0.20：没有可零风险删的余量了。**
- **下一轮起必须按整块搬推进**（每块 = 手工逐段 + 每段编译 + 整轮 A/B/CTS），顺序建议（§0.19 表的第一、二行）：
  1. `bindCullDistanceEmulationBuffers:`(85)：areas 加 `tess_cull_capture_first_instance` / `…_instance_stride` 两个 `uint32_t`
     （壳里从 `r->_tessellation` 填）→ 新建 `mgl_draw_stage_host.{h,c}` → 手工把 `id captureBuffer` 改 `void *`、
     3 处 `__bridge` 去掉 → 段内 `recordLastBoundVertexBuffer:` / `invalidateLastBoundVertexBufferAtIndex:` 改直调第 22 刀的 C 函数
     → 删方法 + 声明 → 调用点（`mgl_draw_metal_port.m` 2 处，C）改直调。
  2. MS 循环族（26+39）：先 C 化 `endRenderEncodingLocked`（需 3 个 manager 入口 + `mglPlatformShellGuardedCall`），
     再把 block 参数换成 `fn + ctx`。
  3. 之后按 §0.19 表逐行往下（`bindMTLTextureLocked:` → `MGLPipelineCache.m` → `+SwapDiagnostics.m` → `mgl_draw_metal_port.m` → 三厚块）。

91. **P0-1 第三十七刀：`bindCullDistanceEmulationBuffers:` 整块转 C（**−87 行**，第一个"整块搬"样本）**：
     ① 按 §0.21 的配方执行（**手工逐段 + 每段编译**，不用正则批处理）：
     - areas 加两个 `uint32_t`：`tess_cull_capture_first_instance` / `tess_cull_capture_instance_stride`，壳里从
       `r->_tessellation.cullDistanceCaptureFirstInstance/…InstanceStride` 填（**加字段不算新端口**）；
     - 体搬进 `mgl_draw_support.c` 的 `mglDrawBindCullDistanceEmulationBuffers`：
       `ctx` → `areas.ctx`、`_backend` → `areas.backend`、`_tessellation.*` → 两个新字段、
       **`id captureBuffer = (__bridge id)X` → `void *captureBuffer = X`**、去掉全部 `__bridge`、
       `[self recordLastBoundVertexBuffer:…]` / `[self invalidateLastBoundVertexBufferAtIndex:…]` → 第 22 刀的 C 函数、
       `MIN` → 本地 `mglDrawSupportMinU32`；
     - 依赖声明三处**必须本地 extern**（都在 ObjC 头里）：`mglDrawSupportEncodeContextIsActive`（返回 `int`）、
       `mglRendererGetValidatedVAO`、`mglResolveProgramForStageFromState`；类型头要补 `mgl_encode_context.h`、
       `mgl_vertex_attrib_binding.h`、`mgl_renderer_backend.h`；
     - 方法与私有头声明删除；`mgl_draw_metal_port.m` 两处调用点（一行 `MGLRenderer *host` 形式 + 一处 `mglStageHostSelf` 守卫）改直调。
     ② **编译期踩到的四类错（本轮已全部解决，供后续整块搬直接复用）**：`MGLEncodeContext` 未引入（缺 `mgl_encode_context.h`）；
     `mglMinU32` 定义在使用点之后；转换脚本残留两处 `self`（原来是 `[self …]` 的接收者）；两处 ObjC 调用点仍发旧 selector。
     ③ **度量**：行数 **34,553 → 34,466**、语法 **1,973 → 1,963**、词汇 3,836（持平）；文件 16、shim 端口 13 不变；
     `+DrawStageHost.m` 185 → **102 行**（余 MS 循环族 26+39）。
     ④ **oracle**：旧库 = 提交 `e1a0a1f` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+DrawStageHost.m` 余 MS 循环族（26+39）——先 C 化 `endRenderEncodingLocked`（3 个 manager 入口 + `mglPlatformShellGuardedCall`，
     注意其中 `_renderPassManager` 的三处调用已由第 81 刀的 `mglRenderPassManager*` C 入口覆盖），再把 block 参数换成 `fn + ctx`。

91. **P0-1 第三十七刀：`bindCullDistanceEmulationBuffers:` 整块转 C（**−87 行**，第一个"整块搬"样本）**：
     ① 按 §0.21 配方**手工逐段 + 每段编译**（不用正则批处理）：
     - areas 加两个 `uint32_t`：`tess_cull_capture_first_instance` / `tess_cull_capture_instance_stride`，壳里从
       `r->_tessellation.cullDistanceCaptureFirstInstance/…InstanceStride` 填（**加字段不算新端口**）；
     - 体搬进 `mgl_draw_support.c` 的 `mglDrawBindCullDistanceEmulationBuffers`：`ctx` → `areas.ctx`、
       `_backend` → `areas.backend`、`_tessellation.*` → 两个新字段、
       **`id captureBuffer = (__bridge id)X` → `void *captureBuffer = X`**、去掉全部 `__bridge`、
       两处 `[self …]` → 第 22 刀的 C 函数、`MIN` → 本地 `mglDrawSupportMinU32`；
     - 三个依赖**必须本地 extern**（声明在 ObjC 头里）：`mglDrawSupportEncodeContextIsActive`（返回 `int`）、
       `mglRendererGetValidatedVAO`、`mglResolveProgramForStageFromState`；类型头补 `mgl_encode_context.h` /
       `mgl_vertex_attrib_binding.h` / `mgl_renderer_backend.h`；
     - 方法与私有头声明删除；`mgl_draw_metal_port.m` 两处调用点（`MGLRenderer *host` 形式与 `mglStageHostSelf` 守卫形式）改直调。
     ② **编译期四类错（已全部解决，供后续整块搬复用）**：缺 `mgl_encode_context.h`；helper 定义在使用点之后；
     转换残留两处 `self`（原 `[self …]` 的接收者）；两处 ObjC 调用点仍发旧 selector——**后两类正说明"手工逐段搬 + 每段编译"不可省**。
     ③ **度量**：行数 **34,553 → 34,466**、语法 **1,973 → 1,963**、词汇 3,836（持平）；文件 16、shim 端口 13 不变；
     `+DrawStageHost.m` 185 → **102 行**（余 MS 循环族 26+39）。
     ④ **oracle**：旧库 = 提交 `e1a0a1f` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：`+DrawStageHost.m` 余 MS 循环族（26+39）——先 C 化 `endRenderEncodingLocked`（3 个 manager 入口 +
     `mglPlatformShellGuardedCall`；其中 `_renderPassManager` 的三处调用已被第 81 刀的 `mglRenderPassManager*` C 入口覆盖），
     再把 block 参数换成 `fn + ctx`。

### 0.22 收掉 `+DrawStageHost.m` 的四步直线计划（第 57 轮把依赖链全部走通，可直接照做）

目标：**删掉 `MGLRenderer+DrawStageHost.m`（现 102 行 / 2 方法）→ 文件 16 → 15**。经核实，路上每一环的依赖**现在都已具备**，
四步均为"手工搬 + 每步编译 + 整轮 A/B/CTS"：

| 步 | 目标 | 依赖现状 | 预估 |
|---|---|---|---|
| ① | `updateGLSampledCopiesForEndedRenderPassFramebuffer:drawCount:drawBuffers:reason:`（`+RenderPass.m:4903`，约 40 行）→ C | 只用 `ctx`（areas ✓）与两处 `[self framebufferAttachmentTexture:]`（**已有 C 端口 `mglRendererAttachmentTextureFor`**）；`drawCount`/`drawBuffers` 原体已 `(void)` 弃用 | −40 |
| ② | `endRenderEncodingLocked`（`+RenderPass.m:5060` 附近，约 91 行）→ C | 它的三个 manager 依赖（`End`/`Clear`/`Discard`）**已在第 27 刀做成 C 入口**；`@try/@catch` 由 **`mglPlatformShellGuardedCall`** 承担；`_batching`/`_renderPassManager` 由 areas 覆盖；唯一剩余依赖就是第 ① 步 | −91 |
| ③ | MS 循环族两方法（`+DrawStageHost.m` 26+39）→ C | 依赖 `[self endRenderEncodingLocked]`（第 ② 步后可用）与 `[self newCommandBufferLocked]`（大方法，需在其内部改用 C 入口或一并搬）；**block 参数 `void (^)(void)` 换成 `fn + ctx`**，调用点（`mgl_draw_metal_port.m` 各 2 处）随之改 | −65 |
| ④ | 删除 `MGLRenderer+DrawStageHost.m` + 私有头声明 | 文件内已无方法 | 文件 **16 → 15** |

**注意事项（前几轮换来的）**：`_mglInMSSampleDrawLoop` / `_mglForcedMSSampleId` / `_mglMSSamplePlaneOffset` 是**私有 ivar**
→ 用"方法 + 壳转发"（§0.14 第二条）而不是 areas 槽地址（第 83/85 刀的结论）；每步完成后立刻 `make -j4 lib`
（clang 是唯一裁判），再做 A/B 与 CTS；不要试图用正则批处理（第 45/91 条）。

92. **P0-1 第三十八刀：§0.22 第①步——`updateGLSampledCopiesForEndedRenderPassFramebuffer:` 转 C（**−70 行**）**：
     ① 体搬进 `mgl_blit_sampled_copy.c` 的 `mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer(renderer, fbo, reason)`：
     `ctx` → `areas.ctx`；两处 `[self framebufferAttachmentTexture:attachment]` → **已有 C 端口**
     `mglRendererAttachmentTextureFor(areas.ctx, attachment)`；`id source = (__bridge id)(tex->mtl_data)` → `void *source = tex->mtl_data`；
     末尾的 `mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, …)` → 传 `renderer`；
     **`drawCount` / `drawBuffers` 两个参数按原体语义（本来就是 `(void)` 弃用）直接取消**；`NSLog`/trace 调用不变。
     ② 方法与 `MGLRenderer+RenderPass_Private.h` 声明删除；唯一调用点（`+RenderPass.m` 的 `endRenderEncodingLocked` 内）改直调。
     ③ **编译期三类错（已解决）**：残留 `(void)drawCount/drawBuffers`；`mglRTWriteAuthorityIsCurrentAndUsesOriginal` 的头是
     **`mgl_coordinate.h`**（不是 `mgl_frame_activity.h`）；调用点仍发旧 selector。
     ④ **度量**：行数 **34,466 → 34,396**、语法 **1,963 → 1,957**、词汇 3,836（持平）；文件 16、shim 端口 13 不变。
     ⑤ **oracle**：旧库 = 提交 `6e4d2ca` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：§0.22 第②步 `endRenderEncodingLocked`（≈91 行）——它的三个 manager 依赖、guarded 入口与本步的方法都已就绪，
     可直接手工搬（注意 `@try/@catch` 用 `mglPlatformShellGuardedCall`、`_batching` 用 `areas.batching`）。

### 0.23 `syncResourceBindingsForContext:` 的阻塞点（第 59 轮实测）

`-[MGLRenderer syncResourceBindingsForContext:alreadyDone:]`（`+Binding.m`，27 行）看似"全走已有 C 入口"，实测**有一处阻塞**：

| 依赖 | 现状 |
|---|---|
| `[self mapBuffersToMTL]` | ✅ 端口 `mglRendererMapBuffersToMTLPort` |
| `[self updateDirtyBaseBufferList:&state->X]` ×2 | ✅ C `mglRenderUpdateDirtyBaseBufferList(ctx, list, where)` |
| `_renderPassManager.state->currentRenderEncoderOwner` | ✅ `areas.command` |
| `[self bindVertexBuffersToCurrentRenderEncoder:&encCtx]` / `…FragmentBuffers…` / `…Textures…` | ✅ 三个端口（shim） |
| `mglBatchBindActiveTexturesToMTL(self, ctx)` | ✅ 已是 C |
| `[self restoreRenderEncoderAfterTextureUploadForDraw:]` | ✅ 端口 `mglRendererRestoreRenderEncoderAfterTextureUploadPort` |
| **`[self bindBufferSizeConstantsForRenderEncoder]`** | ❌ **阻塞**：该方法（`+RenderPass.m:6642`，约 40 行）含 `id vertexSizeBuffer` / `_device` / 多处 `__bridge`，**且没有对应端口或 C 入口** |

**结论**：这一刀的前置是先把 `bindBufferSizeConstantsForRenderEncoder` 转 C（需 `_device` → 走"方法 + 壳转发"，`id`/`__bridge` 手工改 `void *`，`_backend` 用 areas）。
**优先级判断**：它的收益（27 行）与前置成本（40+ 行的 `bindBufferSizeConstants…`）不成比例，**因此不推荐作为下一刀**；
仍建议按 §0.22 的直线计划推进——**第②步 `endRenderEncodingLocked`**（依赖已全部就绪：3 个 manager C 入口 + 壳守卫 + 第 38 刀刚转好的
`mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer`，`_batching`/`_renderPassManager` 由 areas 覆盖），
做完第③④步即可**删掉 `+DrawStageHost.m`（文件 16 → 15）**。

### 0.24 第 60 轮（本轮次预算末轮）交接快照

- **状态（提交 `9dcbaee`，工作区干净）**：`objc_zero.sh` = **16 个 `.m` / 空 TU 0 / 34,396 行 / 语法 1,957 / 词汇 3,836**；
  shim **13 端口 / 223 行 / 37 语法**（声明面＝实现面，端口与平台壳同处 `MGLPlatformRendererShell.m` = 唯一壳 TU）。
  复验：`make -j4 lib` **0 error**；`test-regression` **92/0/2**；`test_batch_icb: ok`；上一刀 CTS 七簇 diff 全空、A/B 确定性行逐行一致。
- **相对基线（2026-09-12 @ `8e64afb`）**：文件 **53 → 16**（−37）、行数 **43,989 → 34,396**（−21.8%）、
  语法 **2,268 → 1,957**（−13.7%）、词汇 **4,353 → 3,836**（−11.9%）、端口 **43 → 13**（−70%）。
- **终态仍未达成**：要求 `MGL/` 内零 `.m`（至多一个平台壳 TU），现存 16 个 `.m`；目标**保持 active**，未标记完成。
- **下一步（唯一推荐路线，§0.22 四步直线，依赖已全部就绪）**：
  1. `endRenderEncodingLocked`（`+RenderPass.m`，≈91 行）→ C：用第 27 刀的 `mglRenderPassManager{End,Clear,Discard}…`、
     壳的 `mglPlatformShellGuardedCall`（吃 `@try/@catch`）、第 38 刀的
     `mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer`，`_batching`/`_renderPassManager.state` 走 areas；
  2. MS 循环族（`+DrawStageHost.m` 26+39）→ C：block 换 `fn + ctx`，三个 `_mgl*` 私有 ivar 走"方法 + 壳转发"；
  3. 删 `MGLRenderer+DrawStageHost.m` + 私有头声明 → **文件 16 → 15**；
  4. 之后按 §0.19 表继续（`bindMTLTextureLocked:` → `MGLPipelineCache.m` → `+SwapDiagnostics.m` → `mgl_draw_metal_port.m` → 三厚块）。
- **不要碰的目标**（成本倒挂，已实测）：`syncResourceBindingsForContext:`（§0.23，前置 40 行的 `bindBufferSizeConstantsForRenderEncoder`）；
  §0.20/§0.21 列出的访问器与一行转发；§0.18 里 §0.19/§0.20 已判定为假阳性的名字。

93. **P0-1 第三十九刀：`endRenderEncodingLocked` 里两份重复的 trace 清理块转 C（**−23 行**）**：
     ① 该方法里"清空 fragment texture trace bindings"的逻辑（trace 开时 `mglTraceFragmentTextureTraceBindings` + memset，
     否则 `mglClearFragmentTextureTraceFunctionalFlags`）**原样出现了两遍**（正常路径与 `@catch` 路径）。
     新增 C 入口 **`mglClearFragmentTraceBindingsForRenderer(void *renderer, const char *reason)`**
     （落在 `mgl_trace_strategy.c`）：状态全部来自 areas——`areas.fragment_trace_bindings`（已在 areas 里）、
     `areas.ctx`（取 `mglCurrentRenderProgramKey`）、`areas.pipeline_cache->pipelineProgramName`。
     两处各 11 行 → 各 1 行。
     ② **本刀是"半块"策略的样本**：`endRenderEncodingLocked` 整体（≈91 行）尚需 `mglRenderPassManagerClearRenderPassIdentity`
     的 C 入口与 catch/正常两条路径的守卫化，本轮先把它内部**重复且纯 C** 的部分拿走，控制面（`@try/@catch`）保持不动，
     以零风险换取行数下降；整块搬迁仍按 §0.22 第②步在后续轮次做。
     ③ **度量**：行数 **34,396 → 34,373**、语法 **1,957 → 1,959（+2）**、词汇 3,836（持平）；文件 16、shim 端口 13 不变。
     **语法 +2 的原因**：两处调用点各引入一个 `(__bridge void *)` 桥接（与第 46/59/83 条同款现象），如实记账、不宣称语法收益。
     ④ **oracle**：旧库 = 提交 `bd9cf01` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     下一刀：继续 §0.22 第②步的剩余部分——补 `mglRenderPassManagerClearRenderPassIdentity` 的 C 入口（其体只是
     `mglRenderClearRenderPassIdentity(_state.renderPassIdentityOwner)` 一类调用），再把 `@try/@catch` 用
     `mglPlatformShellGuardedCall` 包成"C 体 + 失败后清理"两步；随后第③④步即可删 `+DrawStageHost.m`。

### 0.25 `endRenderEncodingLocked` 完全转 C 前还差两件（第 61 轮实测）

第 93 刀已把该方法内部**重复且纯 C** 的 trace 清理块拿走（`mglClearFragmentTraceBindingsForRenderer`）。要把整个方法（≈91 行）搬完，
实测还剩两处必须先补：

| 阻塞 | 现状 | 解法 |
|---|---|---|
| `[_renderPassManager clearRenderPassIdentity]` | 无 C 入口（第 27 刀只做了 `End`/`Clear`/`Discard` 三个） | 在 `mgl_render_pass_manager_ops.c` 加第 4 个入口（体应是 `mglRenderClearRenderPassIdentity(cs->renderPassIdentityOwner)` 一类调用） |
| `mglLogRenderPassLifecycle(…, id drawable, …)` | 签名带 **`id drawable`**（定义在 `MGLRenderer.m`），而 `_drawable` 是**私有 ivar**、且**不在 `MGLRendererCoreState`**（该结构只有 `drawBuffers[]` / 尺寸交接通道） | 二选一：① 给日志函数加一个 `void *` 版 C 变体；② 用"方法 + 壳转发"提供 `mglPlatformShellDrawable(void *)`，再让 C 版日志收 `void *` |

**第 62 轮实测修正（重要）**：**item 2 没有"便宜解"**——我原以为可以加个壳助手 `mglPlatformShellLogRenderPassEnd` 包一层，
实测它是**净增行数**（去掉 11 行块、新增 ~20 行壳函数）；另一条路是**把 `mglLogRenderPassLifecycle` 的 `id drawable` 改成 `void *`**，
但那会让 **5–6 个 ObjC 调用点各加一个 `(__bridge void *)_drawable`**（+6 语法）。**结论：item 2 必须作为"整块搬迁"的一部分一起做**
（改签名 + 调用点桥接），**不要**为它单独开一刀；而 item 1（`clearRenderPassIdentity` 的 C 入口，体是
`mglRenderClearFboMatchCache` + 组装 identity + `mglRenderPassManagerStoreIdentity`）是**真正可加性**的前置，值得单独落地。

**结论**：item 1 是小改动（~10 行）且可单独落地；**item 2 与整块搬迁绑定**；做完后第 93 条的"半块"与它们合起来就是 §0.22 第②步的完整内容，
之后第③步 MS 循环族、第④步删除 `+DrawStageHost.m`（**文件 16 → 15**）。**本轮不做代码改动**：可用上下文已不足以在一次闭环内完成
"补两处 + 搬 91 行 + 门禁/CTS/A/B"，按既定纪律不把树留在半成品状态。

94. **P0-1 第四十刀：补第 4 个 manager C 入口 `mglRenderPassManagerClearRenderPassIdentity`（**可加性前置，0 行净变化**）**：
     ① 本刀**只加不减**（按第 62 条修正后的路线：先把"真正可加性"的前置落地，再把整块搬迁作为同一刀完成）：
     在 `mgl_render_pass_manager_ops.c` 里照 manager 的两处 file-static 写成本地 twin——
     `mglRenderPassManagerSyncIdentityView`（把 identity 的 5 个字段写进 `MGLCommandState`）与
     `mglRenderPassManagerStoreIdentity`（必要时 `mglRenderCreateRenderPassIdentityOwner` → `mglRenderUpdateRenderPassIdentity`
     → 失败即销毁 → 再同步 view），然后实现 `mglRenderPassManagerClearRenderPassIdentity(void *renderer)`：
     `mglRenderClearFboMatchCache(cs->renderPassIdentityOwner)` → 组装 `draw_buffers[] = mglRenderEmptyDrawBuffer()` →
     `StoreIdentity`。`areas.command` 提供 `MGLCommandState`，**零新增端口**。
     ② **度量**：`objc_zero.sh` 不变（**16 文件 / 34,373 行 / 语法 1,959 / 词汇 3,836**，因为新增代码全在 `.c` 侧）；
     C 侧新增约 50 行。这是**为下一刀铺路**的提交，如实标注"0 行 ObjC 净变化"。
     ③ **oracle**：旧库 = 提交 `1ddca30` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ④ 下一刀（**整块搬迁，含第 62 条 item 2**）：把 `endRenderEncodingLocked`（≈91 行）搬成 C——用本刀的
     `ClearRenderPassIdentity`、第 27 刀的 `End`/`Clear`/`Discard`、第 93 刀的 `mglClearFragmentTraceBindingsForRenderer`；
     **同一次改动里**把 `mglLogRenderPassLifecycle` 的 `id drawable` 改为 `void *` 并给 5–6 个 ObjC 调用点补
     `(__bridge void *)_drawable`；`@try/@catch` 用 `mglPlatformShellGuardedCall`，catch 路径的清理放在守卫返回失败之后执行。
     做完后第③步 MS 循环族、第④步删 `+DrawStageHost.m`（**文件 16 → 15**）。

95. **P0-1 第四十一刀：`mglLogRenderPassLifecycle` 的 `id drawable` 改 `void *`（**为整块搬迁扫清 item 2；语法 +6，如实记账**）**：
     ① 按第 62/94 条判断，"日志函数的 drawable"必须与整块搬迁一起做；本轮**单独把它先做完**，让下一刀的改动面更小：
     - `MGLRenderer.m` 里的函数签名 `id drawable` → **`void *drawable`**，体内
       `mglPlatformRendererShellTextureForDrawable((__bridge void *)drawable)` → **去掉桥接**（C 指针直传）；
     - 声明（`MGLRenderer+RenderPass_Private.h`）同步改；
     - `+RenderPass.m` 里 **6 个调用点**的 `_drawable` 实参补 `(__bridge void *)_drawable`。
     ② **踩坑**：全局替换把**另一个函数**（`+RenderPass.m:1209` 附近的 trace 辅助函数）里的
     `mglPlatformRendererShellTextureForDrawable(drawable)` 也改掉了 → 编译报 implicit conversion；已按行号定位并单独回改。
     **规则：改签名时不要用全局字符串替换，先按签名行定位再逐个改（第 91/92 条同源）。**
     ③ **度量（如实）**：行数 **34,373 → 34,371**、语法 **1,959 → 1,965（+6）**、词汇 3,836（持平）——**语法是净增**，
     因为 6 个调用点各加一个桥接；本刀**不做进度宣称**，价值是"把下一刀（`endRenderEncodingLocked` 整块搬迁）的改动面缩到最小"。
     ④ **oracle**：旧库 = 提交 `9c3d068` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：**整块搬迁 `endRenderEncodingLocked`**（≈91 行）——本刀 + 第 93 刀（trace 清理 C 入口）+ 第 94 刀
     （`ClearRenderPassIdentity`）+ 第 27 刀（`End`/`Clear`/`Discard`）+ `mglPlatformShellGuardedCall` 已把它的依赖全部凑齐。

96. **第 65 轮：整块搬迁的最后一块拼图（`mglPlatformShellDrawable`）与精确代码骨架**

第 95 刀把日志函数签名改成 `void *drawable` 后，`endRenderEncodingLocked` 的 C 版**只剩一处还需要壳帮忙**：
外侧函数要调用 `mglLogRenderPassLifecycle(…, drawable, …)`，而 `_drawable` 是**私有 ivar**（且不在 `MGLRendererCoreState`）→
必须补一个 **"方法 + 壳转发"**：

```objc
/* MGLRenderer.m */
- (void *)mglDrawablePointer { return (__bridge void *)_drawable; }   /* 方法体内用 ivar 合法 */
/* 壳 TU */
void *mglPlatformShellDrawable(void *renderer)
{ MGLRenderer *r = (__bridge MGLRenderer *)renderer; return r ? [r mglDrawablePointer] : NULL; }
```

**C 版骨架（下一刀照抄，逐段编译）**：

```c
static int mglRendererEndRenderEncodingGuardedBody(void *renderer)   /* 原 @try 体 */
{
    mglRenderPassManagerEndCurrentRenderEncoder(renderer);
    mglRenderPassManagerClearCurrentRenderEncoder(renderer);
    mglClearFragmentTraceBindingsForRenderer(renderer, "end_render_encoding");
    mglRenderPassManagerClearRenderPassIdentity(renderer);
    return 1;
}

void mglRendererEndRenderEncodingLocked(void *renderer)
{
    mglBindingInvalidateLastBoundState(renderer);                    /* cut 22 的 C 函数 */
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) return;
    if (areas.batching) areas.batching->currentCommandBufferHasWork = 1u;
    Framebuffer *endedFramebuffer = cs->renderPassFramebuffer;
    static uint64_t s_log = 0; uint64_t hit = ++s_log;
    if (hit <= 128ull || (hit % 1024ull) == 0ull) {
        mglLogRenderPassLifecycle("end", hit, areas.ctx, cs->currentCommandBufferOwner,
                                  cs->currentRenderEncoderOwner, cs->renderPassStateOwner,
                                  mglPlatformShellDrawable(renderer),   /* ← 需新增的壳入口 */
                                  cs->renderPassFramebuffer, cs->renderPassFramebufferName,
                                  cs->renderPassDrawBuffer, cs->renderPassDrawBufferCount);
    }
    if (!mglPlatformShellGuardedCall(renderer, "end render encoding",
                                     mglRendererEndRenderEncodingGuardedBody)) {
        fprintf(stderr, "MGL ERROR: Exception ending render encoder - ignoring\n");
        mglRenderPassManagerClearCurrentRenderEncoder(renderer);
        mglClearFragmentTraceBindingsForRenderer(renderer, "end_render_encoding_exception");
        mglRenderPassManagerClearRenderPassIdentity(renderer);
    }
    if (endedFramebuffer)
        mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer(renderer, endedFramebuffer, "end_render_pass");
}
```

**语义等价性说明**：原 `@catch` 体在异常时执行三件事（清 encoder、清 trace、清 identity）——C 版由守卫返回 0 后**紧接着执行同样三件事**；
唯一差别是异常对象只被记录成固定文案（原版打印 `exception.reason`），如需保留 reason，可让守卫把 `description` 写进一个 out 参数。

**本轮不做改动**：可用上下文不足以在一次闭环内完成"新增壳入口 + 新 C TU + 删 91 行方法 + 门禁/CTS/A/B"，按纪律不把树留在半成品状态。

97. **P0-1 第四十二刀：`endRenderEncodingLocked` 整块转 C（**−47 行**，§0.22 第②步完成）**：
     ① 按第 96 条骨架落地，一次改动内完成四件事：
     - `MGLRenderer.m` 加 `- (void *)mglDrawablePointer`（方法体内用私有 ivar 合法），声明进
       `MGLRenderer+GPURecovery_Private.h`；壳 TU 加 C 入口 `mglPlatformShellDrawable(void *)`；
     - 新逻辑写进 `mgl_render_pass_manager_ops.c`：`mglRendererEndRenderEncodingLocked(renderer)`（外侧：
       `mglBindingInvalidateLastBoundState` → encoder 判定 → `areas.batching->currentCommandBufferHasWork` →
       限频 `mglLogRenderPassLifecycle`（drawable 走壳入口）→ guarded 调用 → 失败时执行原 `@catch` 的三条清理 →
       末尾 `mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer`）＋ 静态守卫体
       `mglRendererEndRenderEncodingGuardedBody`（原 `@try` 的三条语句）；
     - 删除 ObjC 方法（**65 行**）与私有头声明；4 个文件的调用点（`+DrawStageHost.m`、`+Binding.m`、`+Blit.m`、`MGLRenderer.m`）
       改为 `mglRendererEndRenderEncodingLocked((__bridge void *)self)`。
     ② **语义差异（如实记录）**：原 `@catch` 打印 `exception.reason`，C 版只能记固定文案（`MGL ERROR: Exception ending render encoder - ignoring`），
     并紧接着执行同样的三条清理；异常对象内容如需保真，可让守卫用 out 参数回传 description。
     ③ **度量**：行数 **34,371 → 34,324**、语法 **1,965 → 1,956**、词汇 **3,836 → 3,831**；文件 16、shim 端口 13 不变。
     ④ **oracle**：旧库 = 提交 `e7dc910` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀（§0.22 第③④步，收尾即可删文件）：MS 循环族（`+DrawStageHost.m` 余 26+39）——block `void (^)(void)` 换 `fn + ctx`，
     `_mglInMSSampleDrawLoop` / `_mglForcedMSSampleId` / `_mglMSSamplePlaneOffset` 走"方法 + 壳转发"；
     它们依赖的 `[self endRenderEncodingLocked]` **本刀已变成 C 函数**，`newCommandBufferLocked` 仍需处理。
     完成后 **删除 `MGLRenderer+DrawStageHost.m` → 文件 16 → 15**。

98. **第 67 轮：MS 循环族的精确搬迁配方（§0.22 第③步；下一刀照此执行即可删文件）**

本轮把两个 MS 方法与它们的调用点都看完了，结论是**比预想简单**：

**① block 不是障碍**：`mgl_draw_metal_port.m` 里两处 `drawOnce:^{ … }` 的块体**都是纯 C 调用**（`mglIssueDrawArrays(...)` / `mglIssueDrawElements(...)`，参数全是块外捕获的普通值）。
→ 换成 `fn + ctx` 即可：定义一个携带这些参数的 C 结构体 + 一个 `static void mglMsDrawOnce(void *ctx)`，把两者传给新的 C 入口。

**② 只剩 3 个私有 ivar + 1 个方法需要壳转发**（其余依赖本周期都已 C 化）：

| 依赖 | 现状 | 需要的壳入口 |
|---|---|---|
| `[self emulatedMSColor0TextureForContext:]` | ✅ 已 C（`mglDrawEmulatedMSColor0Texture`，第 21 刀） | — |
| `[self fragmentNeedsPerSampleMSValuesForContext:]` | ✅ 已 C（`mglDrawFragmentNeedsPerSampleMSValues`，第 20 刀） | — |
| `[self endRenderEncodingLocked]` | ✅ 已 C（`mglRendererEndRenderEncodingLocked`，**第 42 刀**） | — |
| `_mglInMSSampleDrawLoop` / `_mglForcedMSSampleId` / `_mglMSSamplePlaneOffset` | 私有 ivar（读+写） | **两个** C 入口即可：`mglPlatformShellMSSampleInLoop(void *)` 与 `mglPlatformShellSetMSSampleState(void *, int in_loop, int32_t forced, int32_t offset)`（`MGLRenderer.m` 里配对应方法，方法体内用 ivar） |
| `[self newCommandBufferLocked]`（仅 `broadcast…` 用） | 大方法 | 一个壳转发 `mglPlatformShellNewCommandBuffer(void *)` |

**③ 新 C 入口（放 `mgl_gpu_recovery.c` 或新建 `mgl_ms_sample_loop.c`）**：
`int mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(void *renderer, GLMContext ctx, void (*draw_once)(void *), void *draw_ctx)`
与 `void mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(void *renderer, GLMContext ctx)`；逻辑照抄原体（`MAX(tex->samples,1)` 用本地 helper、`mglMarkStateDirtyBits`/`fbo->dirty_bits` 直接用 C）。
**④ 收尾**：删两个方法 + `MGLRenderer+Draw_Private.h` 声明，`mgl_draw_metal_port.m` 两处调用点改成 `fn + ctx` 形式，
然后 **`git rm MGL/src/MGLRenderer+DrawStageHost.m` → 文件 16 → 15**（该文件内已无其它方法）。
**本轮不做改动**：可用上下文不足以在一次闭环内完成"3 个壳入口 + 新 C TU + block 改造 + 删文件 + 门禁/CTS/A/B"。

99. **P0-1 第四十三刀：MS 循环族转 C 并删除 `+DrawStageHost.m`（**文件 16 → 15**，§0.22 第③④步完成）**：
     ① 按第 98 条配方一次完成：
     - `MGLRenderer.m` 加三个方法（私有 ivar）：`-mglMSSampleInLoop` / `-mglSetMSSampleState:forced:offset:` / `-mglEnsureNewCommandBuffer`；
       壳 TU 对应加三个 C 入口 `mglPlatformShellMSSampleInLoop` / `mglPlatformShellSetMSSampleState` / `mglPlatformShellNewCommandBuffer`；
     - 新 TU **`mgl_ms_sample_loop.{h,c}`**：`mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(renderer, ctx, fn, fn_ctx)` 与
       `mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(renderer, ctx)`，体照抄原方法（`endRenderEncodingLocked` 已是 C、
       MS color0 / per-sample 谓词已是 C、`_renderPassManager.state` → `areas.command`）；
     - **block → `fn + ctx`**：`mgl_draw_metal_port.m` 里两处 `drawOnce:^{ … }`（块体本就是纯 C 调用）改为两个小结构体
       `MGLMsDrawArraysOnce` / `MGLMsDrawElementsOnce` + `static void mglMsDrawOnce(void *)`；
     - 删除两个方法（26+39 行）、`MGLRenderer+Draw_Private.h` 声明，并 **`git rm MGLRenderer+DrawStageHost.m`**。
     ② **度量**：`objc_zero.sh` **文件 16 → 15**、行数 **34,324 → 34,281**、语法 **1,956 → 1,951**、词汇 **3,831 → 3,821**；
     shim 端口 13 不变（唯一壳 TU 仍为 `MGLPlatformRendererShell.m`）。
     ③ **oracle**：旧库 = 提交 `f2b4e50` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,980/4,980 与 5,513/5,513
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；plain **92/0/2**；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ④ **§0.22 四步直线全部完成**：① ended-pass sampled-copy 刷新（第 38 刀）→ ② `endRenderEncodingLocked`（第 42 刀）
     → ③ MS 循环族（本刀）→ ④ 删除文件（本刀）。
     下一刀：按 §0.19 表继续——`+Binding.m` 的 `bindMTLTextureLocked:`(339) 与 `MGLPipelineCache.m`(446) 是接下来两个大目标；
     手法照旧（§0.14 三种路线 + 手工逐段 + 每段编译 + 整轮 A/B/CTS）。

100. **P0-1 第四十四刀：`bindMTLTextureLocked:`（339 行）整体转 C（`+Binding.m` 431 → 76 行，语法 39 → 17，词汇 32 → 0）**：
     ① 方法体在 `MGLRenderer+Texture.m`，所以按 §0.14 **只能走"方法 + 壳转发"**这一条路：
     - 新 TU **`mgl_texture_bind.{h,c}`**（450 行 C）：`bool mglRendererBindMTLTexture(void *renderer, Texture *tex)`，体照抄原方法；
       `_renderPassManager.state->currentCommandBufferOwner` → `areas.command->currentCommandBufferOwner`（**零新端口**）；
     - 壳 TU 加 4 个端口（声明进 `mgl_renderer_ports.h`）：`mglRendererCreateMTLTextureFromGLTexturePort` /
       `mglRendererCreateFallbackMTLTexturePort`（两者都 `CFBridgingRetain` 把 +1 交回 C）与
       `mglRendererUploadFullCPUTextureDataPort` / `mglRendererUploadDirtyCPUTextureDataPort`（11 参版，`BOOL *` 出参换 `int *`）；
       渲染目标保留 blit 复用已有 `mglRendererEnsureWritableCommandBufferPort`；
     - `bindMTLTexture:` 只留 `METAL_LOCK()` 断言帧，体内直调 C（`METAL_LOCK()` 本身只是 `MGL_ASSERT_GL_THREAD()`）；
       `+RenderPass.m` 三处 `[self bindMTLTextureLocked: tex]` 改直调 C；**退役 `mglRendererBindMTLTexturePort`**，
       两个 C 调用点（`mgl_attachment_binding.c`、`mgl_batch_replay.cpp`）改为直调 `mglRendererBindMTLTexture`。
     ② **ARC → C 的所有权改写（本刀最需要小心的地方）**：`__bridge id` 强局部（`existingTexture`、`__strong id oldTexture`）在 C 里
     换成"借用指针 + `CFRetain` 别名"，且**别名必须在 `mglSafeReleaseMetalObj((void **)&tex->mtl_data)` 之前取**（否则旧纹理会被提前释放，
     后面的保留 blit 就是 use-after-free）；别名用**裸 `CFRetain`/`CFRelease`** 而非 `mglSafeReleaseMetalObj`——后者会记一次
     `mglMetalCountRelease`，而该对象创建时已经计过数，会造成 created/released 记账偏差。创建端口返回的 +1 存进 `tex->mtl_data` /
     `tex->params.mtl_data` 后即归该槽位所有（未入库时要显式 `mglSafeReleaseMetalObj(&newTexture)`）。
     其余等价改写：`NSLog` → `fprintf(stderr, …)`（同一 sink）、`[[NSDate date] timeIntervalSince1970]` →
     `clock_gettime(CLOCK_REALTIME)`（`mgl_gpu_recovery.c` 的 twin）、`mglEnvFlagEnabled` → `mgl_env_flag_enabled`、
     `MIN(a,b)`（宏无括号，`(GLuint)MIN(…)` 会把**比较结果**强转）→ 显式 `mglTextureBindUploadLevelCount()`；
     `mglBindingCreateDefaultSampler` 的 `__bridge_transfer` 版 → 直接收 `mglRenderCreateDefaultSampler` 的 +1（顺带消掉原实现里
     `(__bridge id)` + `CFBridgingRetain` 造成的一次 +1 泄漏）。
     ③ **度量**：行数 **34,281 → 33,995**、语法 **1,951 → 1,938（−13）**、词汇 **3,821 → 3,795（−26）**；文件 15 **不变**
     （`+Binding.m` 仍留 `syncResourceBindingsForContext:` 27 行，按 §0.23 是成本倒挂项）；`MGLRenderer*.m` **30,841 → 30,493**；
     **壳 TU 560 → 629 行 / 语法 79 → 88**，**端口 13 → 16（+3：退役 1、新增 4）**——按 §0.04 硬规本刀**不算 T4 端口净减**
     （与第 47–49 条同类：这 4 个端口要等 `MGLRenderer+Texture.m` 转 C 时才一起退役）。
     ④ **oracle**：旧库 = 提交 `60cfbae` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（`processGLState.slow` 行按惯例过滤，两臂**未过滤**时计数本就不同步：default 5,272/5,271、
     flushy 5,812/5,812），stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**
     （两臂同值，且与上一轮 `ablib43` 的两臂完全相同 → 是 `MGL_BATCH_MAX_DRAWS=1` 下的既存差异，非本刀引入）；
     **CTS 七簇非通过集合 diff 全空**（hotspot 58 · tess 1 · GS 0 · refq 59 · piq 13 · compute 39 · pp 4）；
     28 目标门禁 `GATE=0`。
     ⑤ 下一刀：**`MGLRenderer+SwapDiagnostics.m`（557 行 / 47 语法 / **只有 2 个方法** / `[self …]` 0 处）**——它体内的
     encoder 操作**已经是** `mglRender*` C++ 助手（只差把 `id` + `(__bridge void *)` 换成 `void *`），ivar 依赖只有
     `_renderPassManager`(3) 与 `_drawable`(1)，后者已有 `mglPlatformShellDrawable`；转完即**整文件删除（文件 15 → 14）**。

### 0.26 第 69 轮快照与"下一刀"排序（按"能否整文件消"重排）

第 100 刀后 `+Binding.m` 只剩 76 行（`syncResourceBindingsForContext:` 27 行 + `bindMTLTexture:` 断言帧），文件数仍是 15。
本轮实测各文件 ObjC 面（`objc_zero.sh` 语法 / 词汇 / 行数 + 方法数 + `[self …]` 数）：

| 文件 | 行数 | 语法 | 词汇 | 方法 | `[self …]` | 可消性 |
|---|---|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 6,954 | 423 | 563 | 49 | 143 | 需多刀整块搬 |
| `MGLRenderer+Texture.m` | 6,500 | 297 | 1,088 | 38 | 94 | 转完可退役第 100 刀的 4 个纹理端口 |
| `MGLRenderer.m` | 4,606 | 169 | 276 | 23 | 54 | 主体类，最后处理 |
| `MGLRenderer+Blit.m` | 4,062 | 236 | 761 | 19 | 74 | 需多刀整块搬 |
| `MGLRenderer+BindingState.m` | 2,914 | 130 | 197 | 17 | 44 | 需多刀整块搬 |
| `MGLRenderer+Tessellation.m` | 2,100 | 151 | 278 | 11 | 57 | 需多刀整块搬 |
| `mgl_draw_metal_port.m` | 2,000 | 101 | 97 | **0** | 23 | host-ops 适配层，只剩 `id`/词汇 |
| `MGLRenderer+Compute.m` | 1,245 | 84 | 104 | 11 | 28 | 需整块搬 |
| `MGLRenderer+Buffer.m` | 823 | **13** | 41 | 9 | **4** | ObjC 面最小（9 个 mapper 相扣） |
| `MGLRenderer+Lifecycle.m` | 667 | 94 | 126 | 12 | 14 | T5 合并候选 |
| `MGLPlatformRendererShell.m`（唯一壳） | 629 | 88 | 55 | 18 | 0 | 终态保留 1 个 |
| **`MGLRenderer+SwapDiagnostics.m`** | **557** | **47** | **101** | **2** | **0** | **下一刀：整文件删除** |
| `MGLPipelineCache.m` | 446 | 62 | 91 | 27 | 18 | 状态已在 `areas.pipeline_cache` |
| `MGLRenderPassManager.m` | 416 | 26 | 17 | 28 | 14 | 多为薄转发 |
| `MGLRenderer+Binding.m` | 76 | 17 | 0 | 2 | 10 | 卡在 §0.23（成本倒挂） |

**排序（下一刀 → 之后）**：① `MGLRenderer+SwapDiagnostics.m`（557 行 / 2 方法 / 0 `[self …]` / 体内已是 `mglRender*` C++ 助手，
转完 `文件 15 → 14`）→ ② `MGLRenderer+Buffer.m`（语法只 13、`[self …]` 只 4 处，9 个 mapper 需逐段搬）→
③ `MGLPipelineCache.m`（27 方法但状态指针都已在 areas）→ ④ `MGLRenderer+Compute.m` → ⑤ `+Tessellation.m` / `+BindingState.m` 两个中块
→ ⑥ 三厚块（`+RenderPass.m` / `+Texture.m` / `+Blit.m`）。每刀纪律不变：**抽盘副本 → 手工逐段 → 每段编译 → 整轮 A/B/CTS/门禁 → 三处文档同步**。

101. **P0-1 第四十五刀：`MGLRenderer+SwapDiagnostics.m` 整文件转 C 并删除（**文件 15 → 14**，零新增端口）**：
     ① 选中它的理由（第 100 条第⑤项的实测复核）：557 行里**只有 2 个方法**、`[self …]` **0 处**、体内对 Metal 的调用
     **本来就已经是 `mglRender*` C 门面**（只差把 `id` + `(__bridge void *)` 换成 `void *`），ivar 依赖只有 5 处且全都有 C 家：
     `ctx` → `areas.ctx`、`_activeState` → `areas.core->activeState`（即 `MGL_STATE()` 的 C 写法）、
     `_defaultDrawableWrittenSinceLastSwap` → `areas.core->…`、`_renderPassManager.state` → `areas.command`。**因此不需要任何新端口。**
     ② 落地：
     - 新 TU **`mgl_swap_diagnostics.{h,c}`**（539 行 C + 53 行头）：`mglSwapCopyRenderPassColorToDrawableIfNeeded(renderer, rp_color0,
       drawable_texture, swap_call, trace_swap)` 与 `mglSwapScheduleTextureSampleDiagnostics(renderer, rp_color0, drawable_texture, swap_call)`；
     - **两个 block 换 C**：局部块 `scheduleTextureSample` → `static mglSwapScheduleTextureSample()`（10 个调用点照旧）；
       完成回调块 → `static mglSwapSampleCompletion()` + 堆上下文 `MGLSwapSampleCompletion`，用
       `mglRenderAddCommandBufferOwnerCompletion()` 注册（它的 `MGLRenderCommandBufferCompletion` 签名本来就是 C 的，块只是包装）；
       上下文里 `sample_buffer` 的 +1 从局部搬进上下文、由 `Destroy` 释放；`NSString *sampleTag` → `const char *`（所有调用点都是
       字面量，异步回调期间不会失效），`[sampleTagCopy isEqualToString:@"src.center"]` → `strcmp`；
     - `NSLog` → `fprintf(stderr, …)`；`%@` 在 `mglTraceLog`（C varargs sink）里换成 `%s` + 缺省 `"(null)"`（原来那条 trace 在
       `%@` 下本来就是坏格式，且该路径因常量恒假而不可达，如实记账）；
     - **常量与结构体单点化**：`kMGLSwapPresentDiagnostics`（原 `MGLRenderer+Blit_Private.h` 的 `static const BOOL … = NO`，
       但 `MGLRenderer+Blit.m:2103` 也在用）移进 `mgl_blit_pipelines.h` 成为 `enum { kMglSwapPresentDiagnostics = 0 }`（避免
       `static const` 在被大量 include 的头里触发 unused 警告），`+Blit.m` 一处改名；`MGLScaledBlitParams` 同理由 ObjC 私有头
       移进 `mgl_blit_pipelines.h`（C 侧要用同一份布局），`+Blit.m` 的 3 个使用点类型不变；
     - 删除 `MGLRenderer+SwapDiagnostics.m` 与 `MGLRenderer+SwapDiagnostics_Private.h`，去掉 `MGLRenderer_Private.h` 里的
       `#import`，`MGLRenderer.m` 两处调用点改直调 C（并同步两处"moved to …"注释）。
     ③ **踩坑（新规则）**：删掉一个**被其它 TU include 的头**之后，`make -j8 lib` 能过，但 `make test-all` 会报
     `No rule to make target 'MGL/include/MGLRenderer+SwapDiagnostics_Private.h', needed by 'build/core/arc/…/mgl_draw_metal_port.o'`
     ——**残留 `.d` 依赖**。按既有纪律处理（**不 `make clean`**）：`find build/core build/es -name '*.o' -o -name '*.d' | xargs rm -f`
     后重建，两个库无错、门禁恢复 `GATE=0`。**规则：删头文件 = 必须清 `.o`/`.d` 依赖面。**
     ④ **度量**：文件 **15 → 14**、行数 **33,995 → 33,443（−552）**、语法 **1,938 → 1,895（−43）**、词汇 **3,795 → 3,694（−101）**；
     `MGLRenderer*.m` **30,493 → 29,942**；**端口 16 不变、壳 TU 629 行/88 语法不变**——本刀是**纯 ObjC 面净减**（§0.04 合格）。
     C 侧新增 592 行。
     ⑤ **oracle**：旧库 = 提交 `9ff89b6` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤时 default 5,272/5,271、flushy 5,806/5,806，差异仍只在 `processGLState.slow` 行），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；**CTS 七簇非通过集合 diff 全空**；
     28 目标门禁 `GATE=0`。
     ⑥ 下一刀：按 §0.26 排序——`MGLRenderer+Buffer.m`（823 行 / 语法只 13 / `[self …]` 只 4 处，9 个 mapper 逐段搬）或
     `MGLPipelineCache.m`（446 行 / 27 方法，但状态指针都已在 `areas.pipeline_cache`）。

### 0.27 第 70 轮快照（第 101 刀后，文件 14）与"整文件消"排序

**本轮新增规则（第 101 条第③项）**：**删掉任何被 include 的头文件后，必须清 `.o`/`.d` 依赖面**
（`find build/core build/es -name '*.o' -o -name '*.d' | xargs rm -f`，**永不 `make clean`**），否则 `make test-all`
会因残留 `.d` 里的已删头报 `No rule to make target`（本轮实测：`make -j8 lib` 能过、`make test-all` 失败）。

第 101 刀删掉 `+SwapDiagnostics.m`（**证明"2 方法 + 0 个 `[self …]` + 体内已是 C 门面"的文件可以零端口整文件消**）后的 14 个文件：

| 文件 | 行数 | 语法 | 词汇 | 方法 | `[self …]` | 可消性 |
|---|---|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 6,954 | 423 | 563 | 49 | 143 | 三厚块之一，需多刀 |
| `MGLRenderer+Texture.m` | 6,500 | 297 | 1,088 | 38 | 94 | 三厚块之一；转完可退役第 100 刀的 4 个纹理端口 |
| `MGLRenderer.m` | 4,611 | 173 | 276 | 23 | 54 | 主体类，最后处理 |
| `MGLRenderer+Blit.m` | 4,062 | 236 | 761 | 19 | 74 | 三厚块之一 |
| `MGLRenderer+BindingState.m` | 2,914 | 130 | 197 | 17 | 44 | 中块 |
| `MGLRenderer+Tessellation.m` | 2,100 | 151 | 278 | 11 | 57 | 中块 |
| `mgl_draw_metal_port.m` | 2,000 | 101 | 97 | **0** | 23 | host-ops 适配层，只剩 `id`/词汇 |
| `MGLRenderer+Compute.m` | 1,245 | 84 | 104 | 11 | 28 | 中块 |
| **`MGLRenderer+Buffer.m`** | **823** | **13** | **41** | **9** | **4** | **下一刀：ObjC 面最小，9 个 mapper 逐段搬** |
| `MGLRenderer+Lifecycle.m` | 667 | 94 | 126 | 12 | 14 | T5 合并候选 |
| `MGLPlatformRendererShell.m`（唯一壳） | 629 | 88 | 55 | 18 | 0 | 终态保留 1 个 |
| `MGLPipelineCache.m` | 446 | 62 | 91 | 27 | 18 | 状态已在 `areas.pipeline_cache` |
| `MGLRenderPassManager.m` | 416 | 26 | 17 | 28 | 14 | 多为薄转发 |
| `MGLRenderer+Binding.m` | 76 | 17 | 0 | 2 | 10 | 卡在 §0.23（成本倒挂） |

**排序**：① `MGLRenderer+Buffer.m`（语法 13、`[self …]` 4；按 mapper 逐段搬，搬完即删文件 → 13）→
② `MGLPipelineCache.m`（27 方法但状态指针已在 areas，先搬状态机再搬缓存创建）→ ③ `MGLRenderer+Compute.m` →
④ `+Tessellation.m` / `+BindingState.m` 两个中块 → ⑤ `mgl_draw_metal_port.m`（0 方法，纯 `id`/词汇清扫）→
⑥ 三厚块（`+RenderPass.m` / `+Texture.m` / `+Blit.m`）→ ⑦ `MGLRenderer.m` 与 `MGLRenderer+Lifecycle.m`（T5 并入唯一壳）。

102. **P0-1 第四十六刀：`MGLRenderer+Buffer.m` 整文件转 C 并删除（**文件 14 → 13**，退役 1 个端口、0 新增）**：
     ① 该文件 822 行 / 9 个方法 / ObjC 语法只 13（`[self …]` 4 处），是 §0.27 排序里的第一目标。**先做依赖审计**：
     它用到的渲染器事实**全部**已在状态区或已是 C 入口——`ctx` → `areas.ctx`、`MGL_STATE(ctx)` → 本地
     `mglBufferMapState(&areas)`（dual-proxy 语义）、`_tessellation.nativeTESActive` → `areas.tess_native_tes_active`、
     `_pipelineCache.state->pipelineState` → `areas.pipeline_cache->pipelineState`；**不需要任何新端口**。
     ② 落地：新 TU **`mgl_buffer_map.{h,c}`**（903+100 行 C）承载 9 个 C 入口
     （`mglRendererMapGLBuffersToMTLBufferMap` / `…MapShaderBufferResourcesViaPlan` / `…MapShaderBufferResourcesToBufferMap` /
     `mglRendererMapBuffersToMTL` / `…UpdateDirtyBuffer` / `…CheckForDirtyBufferData` / `…UpdateDirtyBaseBufferList` /
     `…GetVertexBufferIndexWithAttributeSet` / `mglBufferCreateConvertedVertexBufferForAttribKind` + 释放函数）与 3 个原本就在
     该文件里的 C 函数（`mglAdvanceFrameGeneration` / `mglRecordFrameCompleted` / `mglNoteBufferEncoded`）。
     - **26 个 Objective-C 调用点**改直调 C（`+Binding.m` 3、`+RenderPass.m` 11、`+Tessellation.m` 4、`+Compute.m` 3、
       `+BindingState.m` 3、壳 1 端口实现删除）；**退役 `mglRendererMapBuffersToMTLPort`**，唯一 C 调用点
       `mgl_batch_dyn_bind_encode.c` 改直调 `mglRendererMapBuffersToMTL`（端口 16 → 15，按 §0.04 是净减）。
     - 三个"定义在 ObjC TU、声明在 ObjC 头"的函数按 `mgl_renderer_ports.c` 的既有先例用**文件内 `extern` 原型**接上：
       `mglRendererGetValidatedBuffer`（`NSUInteger` → `unsigned long`）、`mglRenderCheckForDirtyBufferData`、
       `mglRenderUpdateDirtyBaseBufferList`、`mglRenderVertexBufferIndexForAttribute`；`mglShouldTraceCall` 是 ObjC 头的
       `static inline`，在 C 侧写同策略本地 twin（`kMGLDiagnosticStateLogs` 恒 0，两条分支本来就编译掉）。
     - **`where` 标签逐字保留**：三条 C 入口分别传 `"-[MGLRenderer(Buffer) checkForDirtyBufferData:]"` /
       `"…updateDirtyBaseBufferList:]"` / `"…getVertexBufferIndexWithAttributeSet:]"`（`strings` 从旧 .o 里读出的
       `__FUNCTION__` 原文），保证 `MGL BUFFER INVALID in %s:` 这类诊断不变。
     - **所有权**：`convertedVertexBufferForAttribKind:…` 原返回 `__bridge_transfer id`（+1，ARC 作用域结束释放），
       C 版返回 `void *` +1 并新增 `mglBufferReleaseConvertedVertexBuffer()` 供调用点释放；它必须用**裸 `CFRelease`**——
       该引用是转换缓存的 `retain()`（创建计数在缓存建立时已记过一次），走 `mglSafeReleaseMetalObj` 会多记一次 release。
     ③ **顺带删掉两个死函数（按 §0.18 三步法核验）**：`mglSnapshotSharedDirtyBuffer` / `mglSnapshotSharedBufferRange`
     在全树（含 .c/.cpp/测试/基准）**零调用点**（清单纯声明与定义各一处），也不在任何 host-ops 函数指针表里 → 直接删除；
     它们包装的 C++ `mglRenderSnapshotShared{DirtyBuffer,BufferRange}` 因此也变成无引用（留待后续 C++ 侧死代码清扫）。
     另：`+Buffer_Private.h` 里 `mglCompletedFrameGeneration`、`mglShouldTraceBufferTransferCall` 亦无使用者，随头文件一起消失。
     ④ **度量**：文件 **14 → 13**、行数 **33,443 → 32,623（−820）**、语法 **1,895 → 1,879（−16）**、词汇 **3,694 → 3,653（−41）**；
     `MGLRenderer*.m` **29,942 → 29,127**；**壳 TU 629 → 625 行 / 88 → 86 语法、端口 16 → 15**；C 侧新增 1,003 行。
     ⑤ **oracle**：旧库 = 提交 `6cac97b` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤时 default 5,278/5,295、flushy 5,816/5,833：差量全在 `processGLState.slow` 行，
     即已知的批处理计数非确定性），stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。删头文件后同样先清 `.o`/`.d`（第 101 条第③项的规则）。
     ⑥ 下一刀：按 §0.28 排序——`MGLPipelineCache.m`（446 行 / 62 语法 / 27 方法，但状态指针已在 `areas.pipeline_cache`）。

### 0.28 第 71 轮快照（第 102 刀后，文件 13）与剩余 13 个文件的排序

第 102 刀验证了 §0.27 的判据（"ObjC 语法/词汇最少者优先"）：`+Buffer.m` 822 行只掉 16 语法，但**整文件消失**，
而且**顺带退役 1 个端口、删掉 2 个死函数**——**文件数与端口数才是这一阶段的主指标**，语法数只是副产品。

| 文件 | 行数 | 语法 | 词汇 | 方法 | `[self …]` | 可消性 |
|---|---|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 6,955 | 423 | 563 | 49 | 143 | 三厚块之一，需多刀 |
| `MGLRenderer+Texture.m` | 6,500 | 297 | 1,088 | 38 | 94 | 三厚块之一；转完可退役第 100 刀的 4 个纹理端口 |
| `MGLRenderer.m` | 4,612 | 173 | 276 | 23 | 54 | 主体类，最后处理 |
| `MGLRenderer+Blit.m` | 4,062 | 236 | 761 | 19 | 74 | 三厚块之一 |
| `MGLRenderer+BindingState.m` | 2,916 | 129 | 197 | 17 | 44 | 中块 |
| `MGLRenderer+Tessellation.m` | 2,101 | 151 | 278 | 11 | 57 | 中块 |
| `mgl_draw_metal_port.m` | 2,000 | 101 | 97 | **0** | 23 | host-ops 适配层，只剩 `id`/词汇 |
| `MGLRenderer+Compute.m` | 1,246 | 84 | 104 | 11 | 28 | 中块 |
| `MGLRenderer+Lifecycle.m` | 667 | 94 | 126 | 12 | 14 | T5 合并候选 |
| `MGLPlatformRendererShell.m`（唯一壳） | 625 | 86 | 55 | 18 | 0 | 终态保留 1 个 |
| **`MGLPipelineCache.m`** | **446** | **62** | **91** | **27** | **18** | **下一刀：状态指针已在 `areas.pipeline_cache`** |
| `MGLRenderPassManager.m` | 416 | 26 | 17 | 28 | 14 | 多为薄转发 |
| `MGLRenderer+Binding.m` | 77 | 17 | 0 | 2 | 10 | 卡在 §0.23（成本倒挂） |

**排序**：① `MGLPipelineCache.m`（转完文件 13 → 12；其 `pipelineState`/程序名/格式都已在 `areas.pipeline_cache`，
先搬缓存创建/查询，再搬 blend 与 reset）→ ② `MGLRenderer+Compute.m` → ③ `MGLRenderer+Lifecycle.m`（T5 并入壳的候选）→
④ `MGLRenderPassManager.m`（28 个薄转发）→ ⑤ `+Tessellation.m` / `+BindingState.m` 两个中块 →
⑥ `mgl_draw_metal_port.m`（0 方法，纯 `id`/词汇清扫）→ ⑦ 三厚块 → ⑧ `MGLRenderer.m`。

103. **P0-1 第四十七刀：`MGLPipelineCache` 按 T5 并入唯一壳 TU（**文件 13 → 12**，端口不变、语法 −2）**：
     ① 为什么不按 §0.28 的"直接转 C"做（**本刀的判断依据，写在文档里以便复核**）：
     - 该类只剩 **15 处消息发送**（`+RenderPass.m` 9、`+Lifecycle.m` 4、壳 2），但 `_pipelineCache.state->…` 这类
       **纯 C 记录读取有 46 处**、`_renderPassManager.state->…` 另有 270 处（后者属于 `MGLRenderPassManager`，见 §0.29 排序）；
     - 真正卡住的是**归档路径**：`binaryArchiveURL`/`loadBinaryArchive`/`saveBinaryArchive` 用的是
       `NSSearchPathForDirectoriesInDomains` + `NSBundle.mainBundle.bundleIdentifier` + `NSFileManager` + `NSURL`，
       而要等价替换成 C 只能靠 `getenv("HOME")`/`CFBundleGetIdentifier`/POSIX，**产物路径与 `NSError.localizedDescription`
       的文案都会变**；偏偏 A/B 口径（`ab_full.py`）把 `BINARY ARCHIVE` 行当作非确定性**过滤掉**——
       即"改动点正好落在验证盲区"，按纪律不该在无专用 oracle 时动它；
     - 结论：**先按 T5 把类并入唯一壳**（终态本来就允许一个平台壳 TU），等它的调用点随各文件转 C 变成 C 调用后，
       再做"类 → C handle"的转换（那时只有 handful 调用点，且可另起 oracle 专门比对归档文件路径与文案）。
     ② 落地：把 `MGLPipelineCache.m` 的 `@interface MGLPipelineCache ()`（2 个私有方法声明）、
     `kMGLPipelineArchiveBuildSchema`（ASan/TSan/普通三档）、`MGLSafeArchivePathComponent` 与整个 `@implementation`
     搬进 `MGLPlatformRendererShell.m` 的 `#ifndef MGL_PLATFORM_SHELL_SMOKE` 块内（smoke 单独编译面不变）；
     壳新增 `#include "mgl_frame_activity.h"`（`MGL_PERF_INC/ADD`）与 `#include "mgl_air_loader.h"`
     （`MGLRenderPipelineDescriptorState`）；`git rm MGL/src/MGLPipelineCache.m`；`MGLPipelineCache.h`（类接口）不变。
     ③ **踩坑（新规则）**：合并后**第一个编译错误是宏冲突**——`MGLRenderer_Private.h:308` 有
     `#define _device ((__bridge id)mglRendererBackendGetDevice(_backend))`，而该类自己的 ivar 也叫 `_device`；
     它原来的 TU 不 include 渲染器私有头所以相安无事，并进壳后 9 处 `_device` 全部被宏吃掉。
     **修法：把该类 ivar 改名 `_cacheDevice`**（头文件里加注释说明原因），而不是 `#undef` 宏。
     **规则：把 ObjC 类并进壳之前，先查它的 ivar 名是否撞上 `MGLRenderer_Private.h` 的 `_view/_layer/_drawable/_device/_commandQueue…` 宏。**
     ④ **度量**：文件 **13 → 12**、行数 **32,623 → 32,621**、语法 **1,879 → 1,877**、词汇 3,653（持平）；
     壳 TU **625 → 1,069 行 / 86 → 146 语法**；端口 15 不变；`MGLRenderer*.m` 29,127 不变。
     ⑤ **oracle**：旧库 = 提交 `148fb5f` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀：按 §0.29 排序 —— 先**消掉最后一个 <100 行的文件** `MGLRenderer+Binding.m`（77 行），
     前置是把它依赖的 `bindBufferSizeConstantsForRenderEncoder`（78 行）与 `syncResourceBindingsForContext:`（27 行）转 C。

### 0.29 唯一壳 TU 的构成、行数上限与移除路径（第 103 刀后实测）

终态允许**一个**平台壳 TU，因此壳的每一块都要有"为什么它是平台代码"和"它怎么消失"。
`MGLPlatformRendererShell.m` 现为 **1,069 行 / 146 语法**，构成如下：

| 行区间 | 块 | 行数 | 为什么留在壳里 / 移除路径 |
|---|---|---|---|
| 1–236 | `MGLPlatformRendererShell` 类（NSView / CAMetalLayer / drawable / GPU capture / swap interval） | 236 | **终态平台代码**：Cocoa 图层与 drawable 只能由 ObjC 持有；随窗口后端一起保留 |
| 241–319 | 渲染器端口 shim（第 100/102 刀已退役 2 个） | 79 | 每个端口在它转发的方法转 C 时退役 |
| 320–458 | **计算/细分宿主入口 10 个（第 108 刀新增）** | 139 | 随其目标（`MGLRenderer.m` / `+RenderPass.m` / `+BindingState.m`）转 C 逐个退役；`Temporaries*` 三个随计划 API 改收 C keep-alive 集退役 |
| 459–600（约） | **draw/细分宿主入口 10 个 + capture 2 个（第 112 刀新增）** | ~150 | 同上一行：8 个随 `+RenderPass.m` / `+Tessellation.m` / `+BindingState.m` 转 C 退役；capture 两个与 capture 段一起留在壳内 |
| 459–529 | 纹理物化端口（4 个，第 100 刀新增） | 71 | `MGLRenderer+Texture.m` 转 C 时一起退役 |
| 530–765 | Batch replay 壳（`@try/@finally` 帧 + flush/port C 入口） | 236 | 异常帧 C 无法表达；flush 驱动完全 C 化后退役 |
| 766–793 | 纹理绑定入口（`bindMTLTexture:` 的锁帧 + GL 入口，第 104 刀并入） | 28 | `bindMTLTexture:` 的 ObjC 调用点全部转 C 后退役 |
| 794–1447 | `MGLRenderer (Lifecycle)` 类目（第 105 刀并入：构造 / 视图 KVO / 窗口通知 / capture / `dealloc`） | 654 | **终态平台代码**（Cocoa 观察者与窗口 API 无法用 C 表达） |
| 1448–1888 | `MGLPipelineCache` 类（第 103 刀并入） | 441 | 转 C handle（见第 103 条第①项的阻塞与解除条件） |

**上限与纪律**：壳**目标 ≤2,400 行**（第 112 刀后实测 **2,051 行**；上限随"新增的端口/入口"上调，
每次上调都必须像本表这样逐块列出移除路径并说明这些入口何时退役；第 105 刀前上限 1,200、第 105 刀后 1,800、
第 108 刀后 2,000）。**第 112 刀起壳已突破 2,000 行，原因是一次性补入 10 个"draw/细分宿主入口"（第 112 刀）**——
它们随 `+RenderPass.m` / `+Tessellation.m` / `+BindingState.m` 转 C 退役，`mglPlatformShellGpuCapture{Start,Stop}`
则随 capture 段留在壳内（它本来就是 Cocoa 面），退役后壳应回到 ~1,300 行。任何把它继续撑大的合并（T5）都必须在同一刀里更新本表并写明移除路径；
若某块本身不是平台代码（例如只是"尚未转 C 的实现"），**优先转 C 而不是并进壳**。
**壳的收缩路径（终态应回到 ~600 行）**：① 渲染器端口随其转发方法转 C 逐个退役；② 计算/细分宿主入口 10 个随
`MGLRenderer.m` / `+RenderPass.m` / `+BindingState.m` 转 C 退役；③ 纹理物化 4 端口随 `+Texture.m` 退役；
④ `MGLPipelineCache` 类转 C handle；⑤ 纹理绑定入口随 ObjC 调用点清零退役；⑥ batch `@try/@finally` 帧等 flush 驱动 C 化后退役；
⑦ 只剩 `MGLPlatformRendererShell` 类（窗口/图层/drawable/capture）与 renderer 生命周期（KVO/通知/构造/卸载）——
这两块是终态允许保留的平台面（合计 ~890 行）。

剩余 **12** 个文件的可消性排序（按"前置成本 ÷ 文件收益"重排，`_renderPassManager.state->` 读数已实测）：

| 文件 | 行数 | 语法 | 阻塞前置 | 建议 |
|---|---|---|---|---|
| `MGLRenderer+Binding.m` | 77 | 17 | `bindBufferSizeConstantsForRenderEncoder`(78 行) + `syncResourceBindingsForContext:`(27 行) 需先转 C | **下一刀**（前置 ~105 行 C 换整文件消除） |
| `MGLRenderPassManager.m` | 416 | 26 | `_renderPassManager.state->` 读数 **270 处**、发送 34 处（多在 `+RenderPass.m`） | 等 `+RenderPass.m` 转 C 后一起做，避免两次改同 270 处 |
| `MGLRenderer+Lifecycle.m` | 667 | 94 | KVO / NSNotification / NSWindow（真正的 ObjC API） | T5 并入壳（并入前先查 ivar 宏冲突，见第 103 条第③项） |
| `MGLRenderer+Compute.m` | 1,246 | 84 | 11 个方法、28 处 `[self …]` | 整块搬（按 `+Buffer.m` 的手法逐段） |
| `mgl_draw_metal_port.m` | 2,000 | 101 | 30+ 处 host 发送 + Foundation（`NSMutableArray`/`NSString`/capture）+ `id` 签名（头文件同改） | 多刀：先搬纯 `id → void *` 的包装，再搬 host 发送 |
| `MGLRenderer+Tessellation.m` / `+BindingState.m` | 2,101 / 2,916 | 151 / 129 | 中块，依赖已大多就绪 | 整块搬 |
| `+RenderPass.m` / `+Texture.m` / `+Blit.m` | 6,955 / 6,500 / 4,062 | 423 / 297 / 236 | 三厚块 | 多刀 |
| `MGLRenderer.m` | 4,612 | 173 | 主体类 | 最后 |

104. **P0-1 第四十八刀：`MGLRenderer+Binding.m` 整文件消失（**文件 12 → 11**），前置两项转 C（端口不变）**：
     ① 按 §0.29 的排序，先消掉最后一个 <100 行的文件：它的两块内容分别被搬走/转走——
     - **`bindBufferSizeConstantsForRenderEncoder`（105 行）转 C** → 新 TU **`mgl_size_constants.{h,c}`**
       （`bool mglRendererBindBufferSizeConstantsForRenderEncoder(void *renderer)`）。C 化要点：`_backend` → `areas.backend`、
       `_device`（宏）→ `mglRendererBackendGetDevice` 不再需要（原 `mglRenderPassCreateBufferWithBytes` 本身就**忽略 device 参数**，
       直接调 `mglRenderCreateBufferWithBytes(..., NULL, &buf)`）、`_renderPassManager.state->currentRenderEncoderOwner` → `areas.command`、
       `MGL_STATE(ctx)` → 本地 dual-proxy twin、`[self …]` → `mglBindingRecordLastBound{Vertex,Fragment}Buffer`（C）。
       **所有权**：`mglRenderCreateBufferWithBytes` 返回 +1 且**不计** `mglMetalCount*`（计数在 `mglRenderCreateBuffer` 那条包装里），
       所以缓存收下之后再释放我们自己那一份用**裸 `CFRelease`**（等价于 ARC 在作用域末尾的释放）；
       若 `mglRendererBackendSetSizeConstantsBuffer` 失败则同样裸释放并把 buffer 置空。
     - **`syncResourceBindingsForContext:`（27 行）转 C** → `mgl_binding_state_ops.{h,c}` 的
       `bool mglRendererSyncResourceBindingsForContext(void *renderer, GLMContext glm_ctx, const MGLResourceSyncWork *done)`；
       `MGLResourceSyncWork` 结构体随之从 `MGLRenderer+Draw_Private.h` **搬进 `mgl_binding_state_ops.h`**（C 侧要按字段读）；
       注意原代码用的是 `MGL_STATE(glm_ctx)`（**实参**，不是 `ctx` ivar），C twin 照此。
     - 剩下的 `bindMTLTexture:`（`METAL_LOCK()` 帧）与 C 入口 `mglRendererBindTexture` **按 T5 并入唯一壳**（新增
       `@implementation MGLRenderer (BindingShell)` 段），`git rm MGL/src/MGLRenderer+Binding.m`。
     ② 调用点：`+RenderPass.m` 的 `[self syncResourceBindingsForContext:…]` → C 调用；
     `mgl_draw_metal_port.m` 的 `[self bindBufferSizeConstantsForRenderEncoder]` → C 调用；两处方法声明删除
     （`+RenderPass_Private.h` / `+Draw_Private.h`），并更新 `MGLRenderer.m` 里两条"moved to …"注释。
     ③ **度量**：文件 **12 → 11**、行数 **32,621 → 32,471（−150）**、语法 **1,877 → 1,853（−24）**、词汇 **3,653 → 3,645（−8）**；
     壳 TU **1,069 → 1,098 行 / 146 → 150 语法**；端口 15 不变；`MGLRenderer*.m` **29,127 → 28,948**；C 侧新增 ~200 行。
     ④ **oracle（含一次口径澄清）**：旧库 = 提交 `a3ae9ec` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与
     5,514/5,514 逐行保序完全一致**，stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     **口径澄清（新增到纪律里）**：本刀 A/B 的 `processGLState.slow` 行数为 367/595（新臂）对 315/326（旧臂），
     而**同一对库背靠背重跑**得到 293/295（新）对 300/296（旧）——说明**该行的条数在多次运行间可差数十到数百条**，
     是纯粹的运行期噪声（它与 `elapsed=` 一样只反映批处理/调度的时机），**不构成信号**：
     判定只用"去掉 `processGLState.slow` 之后的逐行相等"。**今后 A/B 报告里同时给出两臂的 slow 计数与同库重跑计数**，
     不再把它们当作"逐行一致"的脚注。
     ⑤ 下一刀：按 §0.30 排序（`MGLRenderPassManager.m` 仍然等 `+RenderPass.m`；建议 `+Compute.m` 或 `MGLRenderer+Lifecycle.m`）。

**第 126–127 刀更新（2026-09-14 实测）**：壳 **2,026 行 / 285 语法**，端口面 **31 个**——两刀**各净退役 1 个端口**：
`mglRendererDispatchTessControlShaderPort`（第 126 刀）与 `mglRendererDispatchAIRTessEvalVertexRenderPort`（第 127 刀）的
目标方法都已转 C，两个端口与其壳包装一并删除，C 侧 `mglStageDispatchTCS` / `mglStageDispatchAirTESVertex` 改直调
（按 §0.04 的 T4 硬规，这是本周期少见的**净减端口**刀）。最后一个 AIR 端口随 `+Tessellation.m` 整文件删除时退役（31 → 30），
届时壳应再降约 12 行。上限维持 **2,400 行**。

### 0.30 第 73 轮快照（第 104 刀后，文件 11）

`MGLRenderer*.m` 已从基线 34,604 降到 **28,948**；剩下的 11 个文件里，**7 个是"整块或整类"**（可在一到两刀内消失）：

| 文件 | 行数 | 语法 | 词汇 | 方法 | 备注 |
|---|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 6,850 | 413 | 555 | 48 | 厚块；`_renderPassManager.state->` 270 处是它转 C 的顺带收益 |
| `MGLRenderer+Texture.m` | 6,500 | 297 | 1,088 | 38 | 厚块；转完退役第 100 刀的 4 个纹理端口 |
| `MGLRenderer.m` | 4,614 | 173 | 276 | 23 | 主体类，最后 |
| `MGLRenderer+Blit.m` | 4,062 | 236 | 761 | 19 | 厚块 |
| `MGLRenderer+BindingState.m` | 2,916 | 129 | 197 | 17 | 中块 |
| `MGLRenderer+Tessellation.m` | 2,101 | 151 | 278 | 11 | 中块 |
| `mgl_draw_metal_port.m` | 2,001 | 100 | 97 | 0 | host-ops 适配层；多刀 |
| `MGLRenderer+Compute.m` | 1,246 | 84 | 104 | 11 | **下一刀候选**（11 个方法、28 处 `[self …]`） |
| `MGLPlatformRendererShell.m`（唯一壳） | 1,098 | 150 | 146 | — | 见 §0.29（上限 1,200 行） |
| `MGLRenderer+Lifecycle.m` | 667 | 94 | 126 | 12 | T5 候选（KVO/NSNotification 只能 ObjC） |
| `MGLRenderPassManager.m` | 416 | 26 | 17 | 28 | 等 `+RenderPass.m`（270 处 `state->` 一起改） |

**纪律补充（承接第 104 条第④项）**：A/B 报告必须同时给出**未过滤**的 `processGLState.slow` 计数与**同库背靠背重跑**的对应计数；
只有"去掉该行后的逐行相等"才算通过，因为该行条数在多次运行间可差数十到数百条（本刀实测：新 367/595、旧 315/326，
同库重跑 293/295 与 300/296）。

105. **P0-1 第四十九刀：`MGLRenderer+Lifecycle.m` 按 T5 并入唯一壳（**文件 11 → 10**，语法 −3）**：
     ① 该文件是单一 `@implementation MGLRenderer (Lifecycle)`（构造 / 后端回调绑定 / 视图 KVO / 窗口通知 /
     主动纹理预热 / Metal capture / `dealloc` 卸载），639 行里真正"只能 ObjC"的部分是 **Cocoa API**：
     `addObserver:forKeyPath:`、`NSNotificationCenter`、`NSWindow`、`CALayer`/drawable 拉起、`@try/@catch`——
     与壳的既有职责（窗口/图层/drawable）同域，因此按 T5 并入；它驱动的全部实现早已是 C。
     ② 落地：把文件第 21–666 行（KVO 上下文静态量 + 私有 `@interface MGLRenderer (LifecycleBackendBoundary)` +
     整个 `@implementation`）搬进壳的 `#ifndef MGL_PLATFORM_SHELL_SMOKE` 块（放在端口块之后、`MGLPipelineCache` 之前），
     壳新增 `#import "MGLRenderer+Lifecycle_Private.h"`、`#include "mgl.h"`、`#include "draw_command.h"`、
     `#include "mgl_binding_state_ops.h"`；`git rm MGL/src/MGLRenderer+Lifecycle.m`，`MGLRenderer+Lifecycle_Private.h`
     （给调用方的声明）保持不变。
     ③ **ivat/宏口径**：该文件本来就 `#import "MGLRenderer_Private.h"`，所以 `_view`/`_layer`/`_drawable`/`_device`
     这些**宏**的展开与并壳前完全一致——**这次没有出现第 103 刀那种冲突**（冲突只发生在"自带同名 ivar 的类"上，
     类别方法不会）。
     ④ **度量（如实：文件 −1，但语法只是搬运）**：文件 **11 → 10**、行数 **32,471 → 32,462（−9）**、语法 **1,853 → 1,850（−3）**、
     词汇 3,645（持平）；**壳 TU 1,098 → 1,756 行 / 150 → 272 语法**（上限按 §0.29 修订为 1,800，并在下表逐块说明）；
     `MGLRenderer*.m` 28,948 不变（该文件不在 `MGLRenderer*.m` 里）。**本刀的收益是 TU 数与集中度，不是 ObjC 语法净减。**
     ⑤ **oracle**：旧库 = 提交 `549568f` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,272/5,271 与 5,807/5,806 → `processGLState.slow` **291/290 与 293/292**，两臂几乎同值，
     与本刀是纯搬运相符），stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀：§0.31 排序——壳已到 1,756 行，**优先做能"减少语法"的转换**：`MGLRenderPassManager.m`（416 行 / 26 语法，
     34 处发送 + 270 处 `state->` 读数）或直接开 `MGLRenderer+Compute.m`（1,246 行 / 84 语法 / 11 方法）。

### 0.31 第 74 轮快照（第 105 刀后，文件 10）与"减少语法"阶段的排序

文件数已从 53 降到 **10**，但**剩下 9 个文件 + 壳 = 32,462 行 ObjC / 1,850 语法**，其中三厚块占 17,412 行（54%）。
因此本阶段的目标从"消文件"转向**先减少语法，再消文件**（第 105 刀已证明：纯 T5 搬运不减语法）。

| 文件 | 行数 | 语法 | 词汇 | 方法 | 建议 |
|---|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 6,850 | 413 | 555 | 48 | 厚块：按簇分批搬（encoder 生命周期 → 状态处理 → attachment/persistent） |
| `MGLRenderer+Texture.m` | 6,500 | 297 | 1,088 | 38 | 厚块：先搬 upload/readback 簇，再搬 `createMTLTextureFromGLTexture:`（转完退役 4 个纹理端口） |
| `MGLRenderer.m` | 4,614 | 173 | 276 | 23 | 主体类（`_device`/`_commandQueue` 宏对应的 ivar 在类体内）——最后做 |
| `MGLRenderer+Blit.m` | 4,062 | 236 | 761 | 19 | 厚块：按 blit/copy/resolve 簇分批 |
| `MGLRenderer+BindingState.m` | 2,916 | 129 | 197 | 17 | 中块：dyn-bind 计划已在 C，剩采样器级联与 stage copy-back |
| `MGLRenderer+Tessellation.m` | 2,101 | 151 | 278 | 11 | 中块 |
| `mgl_draw_metal_port.m` | 2,001 | 100 | 97 | 0 | **纯语法清扫**：`id` → `void *`（头文件同改）+ host 发送换 C 入口 |
| `MGLPlatformRendererShell.m`（唯一壳） | 1,756 | 272 | 146 | — | 收缩路径见 §0.29（上限 1,800） |
| `MGLRenderer+Compute.m` | 1,246 | 84 | 104 | 11 | **下一刀候选**：11 个方法、24 处 `[self …]`，其中 ~10 处需要新 C 入口 |
| `MGLRenderPassManager.m` | 416 | 26 | 17 | 28 | 类 → C handle（34 处发送 + 270 处 `state->` 读数；和 `+RenderPass.m` 一起做最省） |

**建议顺序**：① `MGLRenderer+Compute.m`（单文件闭环，收益 1,246 行 / 84 语法 / 文件 −1）→
② `MGLRenderPassManager.m`（类 → C handle，顺带把 `+RenderPass.m` 的 270 处读数换成 `areas.command`）→
③ `mgl_draw_metal_port.m`（纯语法，100 语法 + 97 词汇）→ ④ `+Tessellation.m` / `+BindingState.m` →
⑤ 三厚块与 `MGLRenderer.m`（多刀，按簇搬）。

106. **P0-1 第五十刀：`MGLRenderPassManager` 类 → C struct（**文件 10 → 9**，语法 −76）**：
     ① 该类是"**一个 ivar + 28 个纯转发方法**"（`_state` 是 `MGLCommandState` 值，方法体全是 `mglRender*(…)`），
     所以整类转 C：新 TU **`mgl_render_pass_manager.{h,c}`**（511 + 96 行）定义
     `typedef struct MGLRenderPassManager_t { MGLCommandState *state; MGLCommandState command_state; } MGLRenderPassManager;`
     + 28 个 `mglPassManager*` 入口（含 `mglPassManagerCreate/Destroy`：`calloc` + `state = &command_state` + 清 identity；
     `Destroy` 先 `Shutdown` 再 `free`）。原文件里 3 个 file-static（`CreateCommandBuffer` / `CreateRenderEncoder` /
     `SyncRuntimeOwners` / `SyncIdentityView` / `StoreIdentity`）随之内迁（`CreateRenderEncoder` 改名为
     `mglPassManagerMakeRenderEncoder`，避免与**同名对外的** `mglPassManagerCreateRenderEncoder` 撞名——**编译期就报出来了**）。
     ② **`state` 是指针成员，不是值成员**（关键设计点）：这样 9 个 ObjC 文件里 **468 处**
     `…renderPassManager.state->字段` / `…renderPassManager.state`（作实参传给 `const MGLCommandState *` 形参）
     只需把 `.state` 机械换成 `->state`——**一条替换规则同时覆盖"取字段"和"当实参"两种用法**，无需 `&`、无需改形参类型。
     替换脚本实测：`+RenderPass.m` 348、`MGLRenderer.m` 32、`+BindingState.m` 22、`+Blit.m` 17、`+Tessellation.m` 16、
     `mgl_draw_metal_port.m` 15、`+Texture.m` 9、壳 7、`+Compute.m` 2；另有局部变量形（`renderPassManager.state->`，`+Blit.m` 1 处）
     与 `[mglRendererRenderPassManager(r) state]`（壳 1 处）两类**编译器逐个抓出来**的漏网，已单独修。
     ③ **49 处消息发送**按选择器映射成 C 调用（`discardCurrentCommandBuffer` 6、`clearCurrentRenderEncoder` 4、
     `detachCurrentCommandBufferForSubmission` 6、`installNewCommandBufferFromQueue` 3、`commitCommandBufferTransaction:…` 3、
     `releaseDetachedCommandBufferIfOwned:` 4、`hasLastSubmittedCommandBuffer` 2、`waitForLastSubmittedCommandBuffer:` 2、
     `setCurrentDrawUsesRTSampledCopy:` 2、`setRuntimeContext:` 2、`installRenderEncoder:` 2、`updateRenderPassIdentityForContext:` 2、
     `setFboMatchCacheResult:fboName:generation:` 1、`detachPendingEventWithSyncName:` 1、`installNewRenderPassDescriptor` 1、
     `clearCurrentCommandBufferSyncListEntries` 1、`consumeTransactionCreatedCurrentCommandBuffer` 1、
     `incrementDontCareFrameGenerationWithWrap` 1、`setDontCareFrameGeneration:` 1、`shutdown` 1、`createRenderEncoder` 2 等），
     `BOOL` 实参按 `YES/NO → 1/0` 转换。
     ④ 头文件：删 `MGLRenderPassManager.{h,m}`，`MGLRenderer_Private.h` 的 `#import` 换 `#include "mgl_render_pass_manager.h"`；
     壳里的创建/销毁改 `mglPassManagerCreate()` / `mglPassManagerDestroy()`（`= NULL` 代替 `= nil`）；
     `mglRendererRenderPassManager()` 仍定义在 `mgl_draw_metal_port.m`，返回类型换成 C struct 指针。
     ⑤ **度量**：文件 **10 → 9**、行数 **32,462 → 32,020（−442）**、语法 **1,850 → 1,774（−76）**、词汇 **3,645 → 3,621（−24）**；
     壳 TU 1,756 行不变（语法 272 → 233，因为 7 处发送与 7 处 `.state` 换成 C 调用）；端口 15 不变；
     `MGLRenderer*.m` **28,948 → 28,504**；C 侧新增 607 行。
     ⑥ **oracle**：旧库 = 提交 `22f0577` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,270/5,271 与 5,805/5,807 → `processGLState.slow` **289/290 与 291/293**，两臂几乎同值），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑦ 下一刀：按 §0.32 排序——`MGLRenderer+Compute.m`（1,246 行 / 84 语法 / 11 方法）或 `mgl_draw_metal_port.m`（纯语法清扫）。

107. **P0-1 第五十一刀：host-ops 门面的 25 个 `id`/`NSUInteger` 端口改 `void *`（**语法 −42 / 词汇 −46**，文件数不变）**：
     ① `mgl_draw_metal_port.m` 是"0 方法"的 host-ops 适配层，其中 25 个函数**本来就是纯 Metal 门面**
     （`mglRender*` 的一层包装，只因为类型写成 `id`/`NSUInteger`/`BOOL` 才带 ObjC 语法）。这一刀把这批函数的
     **声明与定义同时改成 C 类型**：`id → void *`、`NSUInteger/NSInteger → size_t/int64_t`、`BOOL → bool`、
     `nil → NULL`、`YES/NO → true/false`，并去掉随之多余的全部 `(__bridge …)`；
     `MGLRenderer+DrawSupportUtil.h` 由 ObjC 头改为 C 可用头（`#import <Foundation/Foundation.h>` →
     `<stdbool.h>/<stddef.h>/<stdint.h>`），唯一另一个使用者 `MGLRenderer+RenderPass.m` 照旧 `#import` ✓。
     ② **踩坑（重要，新的纪律）**：把 `(__bridge_retained void *)x` 一概改成 `return x;` 是**错的**——
     ARC 下 `__bridge_retained` 的含义是"给 C 调用方 +1"，而 C 调用方（`mglGsMetalEndBlit` 等）确实 `CFRelease` 它。
     只有"局部变量本来拿着 +1（来自 create）"的那几处等价；**底层调用只是借用**的两处必须显式 `CFRetain`：
     `mglGsMetalBeginBlit`（借来的 encoder）与 `mglStageNativeFactors`（可能直接返回借用的 canonical 缓冲）。
     实测后果：改错后 `make test-regression` 在 `air_geometry_xfb` 崩在
     `-[_MTLCommandEncoder dealloc]: failed assertion 'Command encoder released without endEncoding'`（encoder 被提前 CFRelease）。
     修法是新增 `mglDrawSupportRetainForCaller()`（`CFRetain` 后原样返回）并在这两处调用；**门禁随即回到 92/0/2**。
     **规则：去 `__bridge_retained` 前，先确认那个 `+1` 原本由谁持有——局部（等价）还是 ARC 的 bridge（必须补 `CFRetain`）。**
     ③ 另一处如实记账的语义微差：`mglCachedDefaultTessFactorBuffer` 走"新建"分支时，ARC 版把结果以 **+0（autorelease）**
     交给调用方，C 版现在交给调用方 **+1**（`mglStageCachedFactors` 只借用不释放）→ **每次新建会多一个引用**，
     但该缓冲由后端缓存持有、按 patch_count/levels 复用，泄漏有界；若强行在 C 里释放，未入缓存的失败分支就会 use-after-free，
     因此**保留 +1 并在此记录**，不做"看起来更干净"的改动。
     ④ **度量**：语法 **1,774 → 1,732（−42）**、词汇 **3,621 → 3,575（−46）**；行数 **32,020 → 32,033（+13，新增注释与 helper）**；
     文件数 9 不变（该文件仍有 ~58 语法：host 发送、`NSMutableArray`、capture 的 `NSString`）；
     `mgl_draw_metal_port.m` 单文件 100 → **58** 语法、96 → **50** 词汇。
     ⑤ **oracle**：旧库 = 提交 `c5e281c` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,274/5,273 与 5,804/5,807 → `processGLState.slow` **293/292 与 290/293**），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀：继续同一文件——剩下 58 语法是 **host 发送 + Foundation**（`NSMutableArray *temps`、capture 的 `NSString`、
     `[self bindBuffersToComputeEncoder:…]` 等），补 ~6 个 C 入口后即可把该文件整体改名 `.c`（**文件 9 → 8**）。

### 0.33 第 76 轮交接（第 107 刀后：9 个文件 / 1,732 语法）与逐文件配方

**状态**：文件 **9**（`+RenderPass.m` 6,842 / `+Texture.m` 6,489 / `MGLRenderer.m` 4,607 / `+Blit.m` 4,062 /
`+BindingState.m` 2,916 / `+Tessellation.m` 2,101（`mgl_draw_metal_port.m` 2,014）/ 壳 1,756 / `+Compute.m` 1,246）；
语法 1,732、词汇 3,575、行数 32,033；端口 15、壳 1,756 行（上限 1,800）。
**本轮周期新增的三条纪律**（都已在前面的刀里踩过并修好，务必照做）：
1. **删头文件 = 清 `.o`/`.d`**（第 101 条第③项）：否则 `make test-all` 报 `No rule to make target`。
2. **并类入壳前查 ivar 宏冲突**（第 103 条第③项）：`MGLRenderer_Private.h` 有 `_view/_layer/_drawable/_device/_commandQueue…`
   宏，自带同名 ivar 的类并进来会被宏吃掉（修法是改 ivar 名）。
3. **去 `__bridge_retained` 前确认那份 `+1` 原本归谁**（第 107 条第②项）：局部持有（来自 create）→ 直接 `return` 等价；
   底层只是借用 → 必须补 `CFRetain`（否则调用方的 `CFRelease` 会提前释放，典型症状是
   `Command encoder released without endEncoding`）。

**逐文件配方（按"下一步最省"排序）**：

| 文件 | 剩余 ObjC 面 | 需要的前置 | 预计 |
|---|---|---|---|
| `mgl_draw_metal_port.m`（58 语法） | host 发送 ~25 处、`NSMutableArray *temps`、capture 的 `NSString`、`#import`×4 | 12–15 个 C 入口（`bindMTLProgram:` / `dispatchTessControlShader:` / `dispatchAIRTessEval{Compute,VertexRender}:` / `endRenderEncoding` / `ensureAIRGeometryPassthroughFunctionForProgram:` / `bindStorageImagesForVertexProgram:` / `bindBuffers,TexturesToComputeEncoder:` / `clearStageBindingCopyBack(s):` / `flushStageBindingCopyBacks:` / `isolatedStageBindingBufferForMap:` / `recordStageBindingCopyBack:` / `ensureRasterEncoderForDraw` / `prepareEmulatedIndirectCPURead:`）；`temps` 用 C 侧 retain 数组替换；capture 段整体挪进壳（它是 Cocoa 面） | **2 刀 → 文件 9 → 8** |
| ~~`MGLRenderer+Compute.m`~~ | **已完成（第 109/110/111 刀）：1,246 行 → 删除** | `mgl_compute_bind.{h,c}` + `mgl_compute_dispatch.{h,c}` + 壳内两个 GL 入口 | **文件 9 → 8 ✓** |
| `MGLPipelineCache` 类（在壳内，约 480 行 / 120 语法） | `NSFileManager`/`NSBundle`/`NSURL`/`NSSearchPath…` | C++ 侧 `mglRenderLoadPipelineCacheArchive` 改为**由 path 自建 `NS::URL`**（2 个函数），归档路径改 `getenv("HOME")/Library/Caches` + `CFBundleGetIdentifier`；**必须另起 oracle**：逐字比对两臂 `MGL BINARY ARCHIVE:` 行与归档文件名（A/B 会过滤这类行） | 2 刀 |
| `MGLRenderer+Tessellation.m`（151 语法 / 11 方法） | 中块，依赖多在 C | 按簇搬（factor buffer/捕获/GS 桥） | 3 刀 |
| `MGLRenderer+BindingState.m`（129 语法 / 17 方法） | 采样器级联 + stage copy-back | `materializeSampledSamplerForTexture:` 已是 ObjC 边界的最后一块 | 3 刀 |
| `+Blit.m` / `MGLRenderer.m` / `+Texture.m` / `+RenderPass.m` | 235 / 168 / 289 / 385 语法 | 三厚块 + 主体类 | 各 4–6 刀 |

**每刀不变的闭环**：手工逐段改 → 每段 `clang -fsyntax-only` → `make -j8`（两库）→ `make test-all`（需 `GATE=0`）→
老库独立构建（`git worktree`，借 `config.mk`/`build/aux`/`external/glfw/build`，**永不 `make clean`**）→ `cmp` 两库不同 →
`/private/tmp/run_ab<N>.sh {new,old}` + `ab_full.py`（**只认"去掉 `processGLState.slow` 后的逐行相等"**，并同时报 slow 计数）→
CTS 七簇非通过集合 diff 全空 → 三处文档（§0.0 进度、§5 日志、§0.2x/§0.3x 快照）→ 提交推送（`git push origin main:main`）。

108. **P0-1 第五十二刀：补齐"计算/细分宿主入口"10 个（**可加性前置，ObjC 语法如实 +29、端口 15 → 25**）**：
     ① **为什么先补桥接而不是直接转文件**：算过 4 个"可整文件消"的剩余文件（`+Compute.m` 1,246 / `mgl_draw_metal_port.m` 2,014 /
     `+Tessellation.m` 2,101 / `+BindingState.m` 2,916）的发送面后，它们的**前置是同一批**宿主入口
     （copy-back 家族在四个文件里分别被调用 27 / 3 / 4 / 1 次，`materializeSampledSamplerForTexture:` 4+2 次，
     `bindMTLProgram:` / `endRenderEncoding` / `newCommandBufferLocked` / `processGLStateLocked:` 各若干）：
     先补一次，随后每个文件转 C 都是纯机械替换；若逐文件各补一次，会把这些入口在四个文件里反复改。
     本条与第 94 条同类：**只加不减、如实标注"0 行 ObjC 净减"**。
     ② 新增入口（`mgl_renderer_ports.h` 声明 + 壳 TU 实现，全部是薄转发，无自有状态）：
     `mglRendererBindMTLProgramPort`、`mglRendererEndRenderEncodingPort`、`mglRendererNewCommandBufferLockedPort`、
     `mglRendererProcessGLStateLockedPort`；copy-back 族 `mglRendererClearStageBindingCopyBacksPort` /
     `…ClearStageBindingCopyBackPort` / `…RecordStageBindingCopyBackPort`（8 参）/ `…FlushStageBindingCopyBacksPort`；
     `mglRendererIsolatedStageBindingBufferPort`、`mglRendererMaterializeSampledSamplerPort`（10 参）；
     以及 keep-alive 集合 `mglRendererTemporariesCreate / Add / Release`（ObjC 侧原来是 `NSMutableArray`，
     C++ 侧只当作不透明句柄 `owned.track(...)` 持有）。
     **所有权逐条写明**：`IsolatedStageBindingBufferPort` 方法返回的是 **+0（autorelease）**，C 入口用 `CFBridgingRetain`
     交回 **+1**（调用方用 `mglSafeReleaseMetalObj` 释放）；采样器入口返回**借用**引用；`TemporariesCreate` 返回 +1、`Release` 释放整集。
     ③ **度量（如实：这一刀 ObjC 面是增加的）**：语法 **1,732 → 1,761（+29）**、词汇 **3,575 → 3,586（+11）**、
     行数 **32,033 → 32,166（+133）**；**端口 15 → 25**、壳 TU 1,756 → 1,888 行 / 233 → 262 语法；文件数 9 不变。
     按 §0.04 硬规本刀**不是 T4 净减**，它的价值是"把 4 个文件的转换前置一次做完"；这 10 个入口随其目标
     （`MGLRenderer.m` / `MGLRenderer+RenderPass.m` / `MGLRenderer+BindingState.m`）转 C 而**逐个退役**，
     `Temporaries*` 三个则在计划 API 改收 C keep-alive 集合时退役。
     ④ **oracle**：旧库 = 提交 `5a28d8c` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,274/5,279 与 5,811/5,813 → `processGLState.slow` 293/298 与 297/299），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。新入口在本刀**尚无调用者**（不可达），
     A/B 一致正是"只加不减"应有的结果。
     ⑤ 下一刀：用这批入口转 **`MGLRenderer+Compute.m`**（bind 两族先搬，`+mgl_compute_bind.{h,c}`），
     随后 `mgl_draw_metal_port.m`（补 capture 段进壳 + `NSMutableArray` 换 keep-alive 集）→ 文件 9 → 8 → 7。

109. **P0-1 第五十三刀：compute 缓冲绑定族转 C（`mgl_compute_bind.{h,c}`，`+Compute.m` 1,246 → 901 行）**：
     ① 把 `-[MGLRenderer bindBuffersToComputeEncoder:stage:copyBacks:executionPlan:temporaries:]`
     （含 3 参重载、三个 `MGL_CBIND_*` 宏、运行时数组 size 常量块，共 345 行）整体搬进 C：
     - **两个 ObjC 重载合成一个 C 入口** `mglComputeBindBuffersToEncoder(renderer, stage, encoder, copy_backs, plan, temporaries)`
       （3 参版就是 `plan=NULL, temporaries=NULL`）；
     - **宏 → 上下文 + 静态函数**：`MGL_CBIND_FLUSH_SNAPSHOT/RETAIN_TEMP/EMIT_BUFFER` 改写成
       `mglComputeBindFlushSnapshot/RetainTemp/EmitBuffer`，作用在 `MGLComputeBindCtx{encoder, plan, temporaries, snapshot, snapshot_ok}`
       上；`useComputeBindingSnapshot` 原本是常量 `YES`，因此直接 `mglComputeSetBuffer` 那条分支**照旧保留但不可达**（与转换前一致）；
     - **所有权**：`isolatedStageBindingBufferForMap:` 原本返回 +0（autorelease），新 C 入口（第 108 刀）交回 +1；
       创建缓冲（隔离缓冲、runtime-array size 缓冲）在 C 里统一走 `mglComputeBindHandOffCreated()`——**先交给 keep-alive 集
       （`mglRendererTemporariesAdd`，等价于 ARC 的 `[temporaries addObject:]`），再释放函数自己那份 +1**，正好还原
       "数组持有一份、强局部在作用域末尾释放"的 ARC 配对；没有 plan（直接编码）时则由 encoder 持有（`setBuffer:` 会 retain），
       与 ARC 的释放点一致。
     ② **顺带完成的 C 化**：`MGLStageBindingCopyBack` / `MGLStageBindingCopyBackList` 两个结构体从 ObjC 的
     `MGLRenderer_State.h` **搬进 C 的 `mgl_binding_stage.h`**（`NSUInteger → size_t`，64 位下同类型），
     ObjC 头改为 include 它——这是第 108 刀那批 copy-back 入口能在 C 侧使用的前提；`+Draw_Private.h` 里两个重载的声明删除；
     `mgl_draw_metal_port.m` 的唯一调用点改调 C 入口，并把该函数里的 `NSMutableArray *temps` 换成
     `mglRendererTemporariesCreate()/Release()`（`temporaries_out` 的 +1 契约不变，注释同步改写）。
     ③ **度量**：语法 **1,761 → 1,742（−19）**、词汇 **3,586 → 3,555（−31）**、行数 **32,166 → 31,823（−343）**；
     `+Compute.m` 单文件 **1,246 → 901 行 / 84 → 70 语法**；`mgl_draw_metal_port.m` 58 → 53 语法；文件数 9 不变；
     C 侧新增 470 行（`mgl_compute_bind.{h,c}`）。
     ④ **oracle**：旧库 = 提交 `b368dd7` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,273/5,273 与 5,804/5,805 → `processGLState.slow` **292/292 与 290/291**，两臂几乎同值，
     且这是 compute 路径被 CTS compute 簇与回归覆盖的一刀），stderr `MGL` 行 **307/307 多重集一致**；
     default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：`+Compute.m` 剩下的 **纹理绑定族**（`bindTexturesToComputeEncoder:` 两个重载 + 三个 `MGL_CTEX_*` 宏，
     同样机制、同样用第 108 刀的采样器入口），搬完即可把 `+Compute.m` 整文件删除（**文件 9 → 8**）。

110. **P0-1 第五十四刀：compute 纹理/采样器绑定族转 C（`+Compute.m` 901 → 535 行，语法 70 → 49）**：
     ① 同一套机制再搬一族：`-[MGLRenderer bindTexturesToComputeEncoder:stage:executionPlan:temporaries:]`
     （含 3 参重载、四个 `MGL_CTEX_*` 宏、两个采样/采样器循环、数组纹理补绑循环，共 341 行）→
     `mglComputeBindTexturesToEncoder(renderer, stage, encoder, plan, temporaries)`，与缓冲族**共用**
     `MGLComputeBindCtx` 与 `mglComputeBindFlushSnapshot/RetainTemp`，新增 `mglComputeBindEmitTexture`（kind 2）与
     `mglComputeBindEmitSampler`（kind 3）。
     ② 两处 ObjC 语义细节**照原样保留并记录**：
     - `MGL_CTEX_RETAIN_TEMP` 原本在 `temporaries` 为 nil 时**就地新建**一个 `NSMutableArray`（该集合随函数返回即被 ARC 释放，
       不会回到调用方）——C 版用 `mglRendererTemporariesCreate()` 建本地集合并**在函数返回前 `Release`**，
       行为与"本地数组在返回时消失"一致（这属于原实现的既存小瑕疵，不借转换之机改变语义）；
     - 默认采样器走 `mglRenderCreateDefaultSampler`（+1）→ `mglComputeBindHandOffCreated()` 先交给 keep-alive 集合再释放自己那份
       （与缓冲族同一对 ARC 语义：数组持有、强局部在作用域末尾释放）。
     ③ 顺带删掉三个**已无使用者**的 file-static：`mglComputeCreateDefaultSampler`、`mglComputeSetTexture`、`mglComputeSetSampler`
     （它们只被搬走的两族使用；按"clang 是唯一裁判"逐个删并编译确认），`+Draw_Private.h` 里两个重载声明删除，
     `mgl_draw_metal_port.m` 的纹理调用点改调 C 入口。
     ④ **踩坑（老问题的又一次实例）**：脚本按"正则匹配整条语句"替换多行调用点时只吃到了中间片段，
     留下 `RETURN_FALSE_ON_FAILURE(` 悬挂前缀，编译立刻报 `unterminated function-like macro invocation`；
     **规则照旧：调用点替换后必须马上编译**（本刀的修复就是把前缀与调用并成一条）。
     ⑤ **度量**：语法 **1,742 → 1,718（−24）**、词汇 **3,555 → 3,536（−19）**、行数 **31,823 → 31,455（−368）**；
     `+Compute.m` 单文件 **901 → 535 行 / 70 → 49 语法**；`mgl_draw_metal_port.m` 53 → 50 语法；
     `mgl_compute_bind.{h,c}` 现共 700 行；文件数 9 不变。
     ⑥ **oracle**：旧库 = 提交 `0690c37` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,272/5,269 与 5,807/5,805 → `processGLState.slow` 291/288 与 293/291），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑦ 下一刀：`+Compute.m` 只剩 535 行（`processCompute` / `runComputeDispatchOrchestrationLocked` /
     `mtlDispatchCompute{,Indirect}Locked` / `mglRendererDispatchCompute{,Indirect}` 顶层入口），搬完即可**删文件（9 → 8）**；
     前置已齐（第 108 刀的 `EndRenderEncoding` / `NewCommandBufferLocked` / `EnsureWritableCommandBuffer` 等入口）。

111. **P0-1 第五十五刀：compute 派发编排转 C，**删除 `MGLRenderer+Compute.m`（文件 9 → 8）**：
     ① `+Compute.m` 自第 109/110 刀后只剩 534 行，本刀把编排部分整体搬到新 TU **`mgl_compute_dispatch.{h,c}`**（543+60 行）：
     - `mglComputeProcess(renderer, encoder, copy_backs, plan, temporaries)`（原 `processCompute:` 两个重载：程序/管线创建 →
       绑定两族 → 清 `dirty_bits`）；`mglComputeRunDispatchOrchestrationLocked(...)`（结束渲染编码 → 可写命令缓冲 →
      image/sampled 纹理预绑 → 执行计划事务或直接编码 → `currentCommandBufferHasWork` → copy-back 冲刷/换命令缓冲 → 脏位）；
      以及 `mglComputeMtlDispatch{,Indirect}Locked(...)`（零尺寸短路、可写 image 的 `metal_data_authoritative` 标记、
      间接缓冲的范围校验）；
     - **两个 GL 入口 `mglRendererDispatchCompute{,Indirect}` 留在壳**（它们要走 `mglRendererForContext` 与 `METAL_LOCK()/UNLOCK()`
       帧，需要 MGLRenderer 类型），体改为调用上面的 C 函数——即 **T5 合并**，壳 +46 行；
     - 状态映射全部沿用既有口径：`_renderPassManager->state` → `areas.command`、`_gpuRecovery.commandRecoveryOwner` →
       `*areas.gpu_recovery_command_owner`、`_batching.currentCommandBufferHasWork` → `areas.batching->…`、
       `_deviceResetRequested` → `areas.core->deviceResetRequested`（`atomic_store_explicit` 照旧）、
       `MGL_STATE(ctx)`/`MGL_STATE(glm_ctx)` → 本地 dual-proxy twin。
     ② **新入口 1 个**：`mglPlatformShellSetContext(renderer, glm_ctx)` —— 两个 locked 方法开头都有 `ctx = glm_ctx;`
     （**给渲染器 ivar 赋值**，C 无法直接做），C 侧改为调用该入口，壳里经新方法 `-mglSetActiveContext:` 赋值，
     保证下游经端口调用到的那些方法读到的渲染器上下文与转换前一致。**这是一处容易被忽略的语义点，单独记录。**
     ③ **执行计划的 keep-alive 生命周期**：原方法里 `NSMutableArray *executionTemporaries = [NSMutableArray array]`
     在方法返回时由 ARC 释放（计划编码/派发都发生在方法内），C 版建 `mglRendererTemporariesCreate()` 并在
     所有返回路径（含 4 条失败早退）显式 `Release`，逐一核对过。
     ④ **度量**：文件 **9 → 8**、行数 **31,455 → 30,975（−480）**、语法 **1,718 → 1,673（−45）**、词汇 **3,536 → 3,480（−56）**；
     壳 TU 1,889 → 1,935 行（语法 262 → 266）；端口 25 → 26（新增 `mglPlatformShellSetContext`）；C 侧新增 603 行。
     ⑤ **oracle**：旧库 = 提交 `0fcc8d2` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,277/5,280 与 5,811/5,813 → `processGLState.slow` 296/299 与 297/299），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀：`mgl_draw_metal_port.m`（2,014 行 / 50 语法）——补 capture 段进壳 + 剩余 host 发送换 C 入口，
     之后**文件 8 → 7**；再往后是 `+Tessellation.m`（2,101 / 151 语法，copy-back 族已就绪）。

### 0.37 第 80 轮交接：`mgl_draw_metal_port.m` 的精确剩余清单（下一刀，文件 8 → 7）

`mgl_draw_metal_port.m` 现有 **2,014 行 / 50 语法 / 48 词汇**，ObjC 面只剩：
`#import`×4、`id`×8、`__bridge`×24、`NS[A-Z]*`×28（多为 `NSUInteger`）、`NSLog`×5、`YES/NO`×18、**23 处发送**。
23 处里 **13 处已有 C 入口**（`processGLState`→`mglRendererProcessGLStatePort`、`newCommandBuffer`→`mglPlatformShellNewCommandBuffer`、
`bindMTLTexture`→`mglRendererBindMTLTexture`、`bindMTLProgram`/`endRenderEncoding`/clear·flush copy-back 族→第 108 刀那批、
`bindFragmentBuffersToCurrentRenderEncoder`→既有端口），**剩下 10 处需要新桥接**（签名见下，全部是薄转发）：

| 目标方法 | 定义处 | 需要的 C 入口 |
|---|---|---|
| `flushCommandBuffer:` | `+RenderPass.m:6552` | `mglRendererFlushCommandBufferPort(renderer, finish)` |
| `ensureRasterEncoderForDraw` | `+RenderPass.m:6772` | `mglRendererEnsureRasterEncoderForDrawPort(renderer)` |
| `prepareEmulatedIndirectCPURead:label:` | `+RenderPass.m:6742` | `…PrepareEmulatedIndirectCPUReadPort(renderer, draw_ctx, label, …)` |
| `ensureAIRGeometryPassthroughFunctionForProgram:outputPrimitive:` | `+RenderPass.m:830` | `…EnsureAIRGeometryPassthroughPort(renderer, program, output_primitive)` |
| `dispatchTessControlShader:` | `+Tessellation.m:1141` | `…DispatchTessControlShaderPort(renderer, glm_ctx, …)` |
| `dispatchAIRTessEvalCompute:` | `+Tessellation.m:1410` | `…DispatchAIRTessEvalComputePort(renderer, glm_ctx, …)` |
| `dispatchAIRTessEvalVertexRender:` | `+Tessellation.m`（同上区域） | `…DispatchAIRTessEvalVertexRenderPort(…)` |
| `bindStorageImagesForVertexProgram:` | `+BindingState.m:2748` | `…BindStorageImagesForVertexProgramPort(renderer, program)` |
| `mglCaptureDescriptorForDevice:` / `mglStartCaptureWithDescriptor:error:` / `mglStopCapture` | 壳的 `MGLPlatformRendererShell` 类 | **不建端口**：把整段 GPU capture 代码（含 `getenv("MGL_GPU_CAPTURE")` 与 `NSString`）**移进壳**，暴露 `mglPlatformShellGpuCaptureStart(renderer)` / `…Stop(renderer)` 两个 C 入口 |

**做法**：① 先加这 9 个入口（可加性、单独一刀、如实记账端口 26 → 35）；
② 再逐簇把 host 发送换成 C 调用（`id`→`void *`、`NSLog`→`fprintf`、`NSUInteger`→`size_t`、`YES/NO`→`1/0`、去掉随之多余的 `__bridge`），
每簇编译一次；③ 全部转完后 `git mv mgl_draw_metal_port.m → .c`（去掉 4 个 `#import`）——
**注意第 107 条的教训**：去 `__bridge_retained` 前确认那份 `+1` 原本归谁（局部持有→等价；底层借用→补 `CFRetain`）。
④ 完成后文件 **8 → 7**，语法约 −50、词汇约 −48。

**再往后**（按 §0.33 的表）：`+Tessellation.m`（2,101 / 151 语法，copy-back 族与采样器入口都已就绪）→
`+BindingState.m`（2,916 / 129）→ 三厚块（`+Blit.m` 4,062 / 235、`MGLRenderer.m` 4,616 / 168、`+Texture.m` 6,489 / 289、`+RenderPass.m` 6,842 / 385）
→ 最后是壳内两块（`MGLPipelineCache` 类转 C handle 后壳可回到 ~1,300 行；lifecycle/壳类 ~890 行是终态允许保留的平台面）。

112. **P0-1 第五十六刀：`mgl_draw_metal_port.m` 的第二批宿主入口 + 9 处调用点转 C（该文件 50 → 37 语法）**：
     ① 按 §0.37 的清单补齐 **10 个入口**（声明进 `mgl_renderer_ports.h`、壳内薄转发）：
     `mglRendererFlushCommandBufferPort`、`…EnsureRasterEncoderForDrawPort`、`…PrepareEmulatedIndirectCPUReadPort`、
     `…EnsureAIRGeometryPassthroughPort`、`…DispatchTessControlShaderPort`、`…DispatchAIRTessEvalComputePort`（笔误见下）、
     `…DispatchAIRTessEvalVertexRenderPort`、`…BindStorageImagesForVertexProgramPort`，以及 **GPU capture 的两个入口
     `mglPlatformShellGpuCapture{Start,Stop}`**——后者按 §0.37 的决议**把 capture 段整体搬进壳**
     （`getenv("MGL_GPU_CAPTURE")` + `NSString` + `MTLCaptureDescriptor` + `NSError`），
     `mgl_draw_metal_port.m` 里只留两个转发到壳入口的 host-ops 回调。
     ② 一个小发现（写下来避免下次再猜）：`mglRendererForContext()` 取的 `context->platform_renderer_shell` **就是 `MGLRenderer`
     对象本身**（`@interface MGLRenderer : MGLPlatformRendererShell`），所以 capture 方法可以直接对 `MGLRenderer *r` 发送——
     不需要任何 delegate 关系。
     ③ **9 处调用点**改 C：`mglStageFlushCB`、`mglStageDispatchTCS`、`mglStageDispatchAirTES`（compute/vertex-render 两个）、
     `mglStageEnsurePassthrough`、GS 存储图像绑定、`mglDrawHostEnsureRasterEncoder`、`mglDrawHostPrepareIndirectCPURead`、
     以及两个 capture 回调。
     ④ **度量（如实：总量微增）**：该文件 **50 → 37 语法 / 48 → 41 词汇**（行数 2,014 → 1,983）；
     但全库语法 **1,673 → 1,683（+10）**、词汇 3,480 → 3,478、行数 30,975 → **31,062（+87）**——
     因为 10 个新入口各自带一个 `(__bridge …)`，而它们的收益要等该文件本体整体转 C 才兑现；
     端口（`grep -c "Port("`）**26 → 33**，另有 2 个不带 Port 后缀的 capture 入口；文件数 8 不变。壳 TU 1,935 → **2,051 行**（见 §0.29 的上限修订）。按 §0.04 这是**可加性前置**。
     ⑤ **oracle**：旧库 = 提交 `4e16dff` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,275/5,273 与 5,808/5,813 → `processGLState.slow` 294/292 与 294/299），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀（该文件收尾）：剩下的 ObjC 面是 **2 处 `[self …]`/`[host …]`（已在 ① 里有入口）、
     23 处 `__bridge`、24 处 `NSUInteger`、16 处 `YES/NO`、6 处 `id`、3 处 `NSLog`、4 处 `#import`**，
     以及一批 `self->_ivar` 访问——其中 `_batching`（→ `areas.batching`）、`_backend`（→ `areas.backend`）、
     `_renderPassManager`（→ `areas.command`）已就绪，**只差 tessellation/geometry 子状态**
     （`nativeTESCopyBacks`、`tessVertexCaptureActive`、`tessIndexedDraw`、`tessVertexCaptureOffset`、
     `tessInstanceRecords`、`geometry.expansionActive`）——按"一个状态 struct + 一个端口"的老办法，
     给 `MGLRendererStateAreas` **加字段**（字段不是端口）即可，然后 `git mv mgl_draw_metal_port.m → .c`（**文件 8 → 7**）。

113. **P0-1 第五十七刀：tessellation/geometry 记录 C 化 + `mgl_draw_metal_port.m` 词汇层转 C（词汇 48 → 1）**：
     ① **新增 C 头 `mgl_tessellation_state.h`**：把 `MGLTessellationState`（含 `nativeTESCopyBacks`、`tessVertexCaptureActive`、
     `tessIndexedDraw`、`tessVertexCaptureOffset`、`tessInstanceRecords`、`tessComputeActive`、`pendingGSInput*` 等）
     与 `MGLGeometryState` 从 ObjC 的 `MGLRenderer_State.h` 搬出（`BOOL → bool`、`NSUInteger → size_t`，64 位下同类型），
     ObjC 头改为 include 它——沿用 `mgl_renderer_core_state.h` / `mgl_batching_state.h` / `mgl_command_state.h` /
     `mgl_pipeline_cache_state.h` 的同一条路线。
     ② **状态区加两个字段**（字段不是端口）：`MGLRendererStateAreas.tessellation` / `.geometry` 是**这两个记录本身的地址**
     （壳的 `mglRendererStateAreasPort` 里填 `&r->_tessellation` / `&r->_geometry`），于是 C 侧既能读也能**写**
     （`_tessellation.nativeTESActive = …` 这类赋值正是该文件需要的）；原有的两个标量字段 `tess_native_tes_active` /
     `tess_native_tes_program` 保留给只要这两个值的调用点。
     ③ **词汇层转换**（本刀实际改动的部分）：该文件里 `NSUInteger → size_t`、`nil → NULL`、`YES/NO → 1/0`、
     `NSLog(@"%s", msg) → fprintf(stderr, "%s\n", msg)`（3 处）。**度量**：该文件词汇 **48 → 1**（只剩注释里的一个 `id`），
     全库词汇 **3,478 → 3,438（−40）**；语法 1,683 不变（剩下的 37 处是 `#import`×4、`__bridge`×23、`MGLRenderer *` 局部与 2 处发送），
     行数 31,062 → 31,064。
     ④ **oracle**：旧库 = 提交 `d5d3786` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,277/5,273 与 5,823/5,810 → `processGLState.slow` 296/292 与 309/296），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ **最后一刀的前置已全部就位**（`mgl_draw_metal_port.m` → `.c`）：剩下的 37 处语法是"体量"问题而非"能力"问题——
     需要把 **~100 处 `mglStageHostSelf(renderer)` / `mglDrawHostSelf(renderer)` 局部**换成 `renderer` + 状态区读取
     （`host->ctx` → `areas.ctx`、`host->_backend` → `areas.backend`、`host->_tessellation.X` → `areas.tessellation->X`、
     `host->_geometry.X` → `areas.geometry->X`、`host->_batching.X` → `areas.batching->X`），
     并把 11 处发送换成第 108/112 刀的 C 入口；`#import` → `#include`（`+DrawSupportUtil.h` 已是 C 可用头，
     `MGLRenderer_Private.h`/`+Draw_Private.h`/`+Tessellation_Private.h` 三条要么换成 C 声明、要么随最后一次改动去掉）。
     **建议按"函数簇"分批改**（每个 host-ops 表一次），每批编译一次——这正是第 109–112 刀的节奏。

114. **P0-1 第五十八刀：`mgl_draw_metal_port.m` 的 11 处发送全部转 C（该文件语法 37 → 25）**：
     ① `mglDrawSupportCaptureProcessGL`、`mglStageBindProgram`、`mglStageProcessGL`、`mglStageEndRender`、
     `mglStageClearNativeCB`、`mglStageFlushNativeCB`、`mglGsMetalEnsureCB`、`mglGsMetalBindDrawTextures`（两处纹理绑定）、
     GS 的 `bindFragmentBuffersToCurrentRenderEncoder:`、`mglDrawHostProcessGLStateLocked` —— 全部改调第 108/112 刀的 C 入口；
     其中两个 copy-back 函数需要 tessellation 记录的地址，改为**就地取状态区**：
     `MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);` 之后用 `&areas.tessellation->nativeTESCopyBacks`
     （这正是第 113 刀把记录 C 化 + 加 `areas.tessellation` 字段的用处）。
     ② 顺带把 `mglPlatformShellNewCommandBuffer` 从 `mgl_ms_sample_loop.c` 里的**本地 `extern`** 升格为
     `mgl_renderer_ports.h` 的正式声明（两个 C TU 用它了）。
     ③ **一次失败的尝试（写下来避免重复）**：我先试着用"一次脚本扫描全文件、按函数体把
     `MGLRenderer *self = mglStageHostSelf(renderer);` 换成状态区取用 + 字段改写"的方式**一次性转完该文件**，
     结果是 47 个函数被改写但 **35 个函数仍残留 `self`/`host`**（`self &&`、`(void *)self`、复合表达式里的用法等
     正则覆盖不全），编译报 17 处 `areas` 未声明 + 1 处重定义。**已 `cp` 备份并整体回退**，树留在上一刀已验证的状态。
     **结论（与本文档一贯纪律一致）**：这个文件的收尾要**按 host-ops 簇分批**做（每簇 5–10 个函数、每簇编译一次），
     不要指望一次正则扫描；`self`/`host` 在这类函数里既有空值判断又有字段访问还有实参三种角色。
     ④ **度量**：该文件语法 **37 → 25**、词汇 1（不变）；全库语法 **1,683 → 1,671（−12）**、词汇 3,438（持平）、
     行数 31,063；文件数 8、端口 33（新增的 `mglPlatformShellNewCommandBuffer` 声明不算新端口）不变。
     ⑤ **oracle**：旧库 = 提交 `a2aaac1` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,272/5,275 与 5,819/5,821 → `processGLState.slow` 291/294 与 305/307），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑥ 下一刀：该文件**已无任何 ObjC 消息发送**，剩下 25 处语法是 4 个 `#import`、约 19 个 `__bridge` 与少量 `MGLRenderer *` 局部；
     按簇把局部改成状态区读取（`host->ctx` → `areas.ctx` 等）后即可 `git mv … → .c`（**文件 8 → 7**）。

115. **P0-1 第五十九刀：`_lastDrawPrimitiveMode` 并入核心状态记录（为最后一刀扫清字段）+ 两次脚本化尝试的失败记录**：
     ① 该 ivar 是 `mgl_draw_metal_port.m` 收尾时**唯一没有 C 家**的渲染器字段（其余 `_tessellation` / `_geometry` / `_batching` /
     `_core` / `_backend` / `_renderPassManager` / `_gpuRecovery.commandRecoveryOwner` / `_bindingStateOwner` 都已能经状态区读写）。
     按"一个状态 struct + 一个端口"的老办法，把它并进 **`MGLRendererCoreState`**（`uint32_t lastDrawPrimitiveMode;`），
     `MGLRenderer_Private.h` 里的 ivar 换成宏 `#define _lastDrawPrimitiveMode _core.lastDrawPrimitiveMode`
     （与 `_activeState`/`_defaultDrawableWrittenSinceLastSwap` 同一写法），ObjC 侧一行未改、C 侧从此可写。
     ② **两次脚本化转换失败，如实记录**（都是我自己踩的，不是环境问题）：
     - **第一次**（第 114 条第③项）：按函数体把 `MGLRenderer *self = mglStageHostSelf(renderer);` 整体换成状态区取用；
       47 个函数被改写但 35 个函数残留 `self`/`host`，编译报 17 处 `areas` 未声明 + 1 处重定义；
     - **第二次**（本刀）：改用"屏蔽注释与字符串后的括号配对"来切函数体 + 补全字段映射（含 `_gpuRecovery.commandRecoveryOwner`
       → `*(areas.gpu_recovery_command_owner)`、`_bindingStateOwner` → `*(areas.binding_state_owner)`、`_core.deviceResetRequested`、
       `_lastDrawPrimitiveMode`），并把兜底规则 `\bself\b → renderer` 也加上；结果仍出现
       `redefinition of 'renderer' with a different type`（说明有的函数体切片跨越了下一个函数的声明行）与新的 `areas` 未声明。
     **两次都已 `cp` 备份后整体回退**，树始终停在上一刀已验证的状态。
     **结论（升格为纪律）**：`mgl_draw_metal_port.m` 的收尾**只能按 host-ops 簇人工改、每簇编译**——
     文件里 67 个函数各自声明 `MGLRenderer *self/host`，三种用法（空值判断 / 字段访问 / 实参）交织，
     任何"一次扫描全文件"的脚本都会在函数边界上出错；本文件剩余 25 处语法（4 个 `#import` + 约 19 个 `__bridge` + 少量局部）
     对应的正是这 67 个函数，**按每簇 5–10 个函数推进，约 7–10 刀可清完**。
     ③ **度量**：语法 1,671（持平）、词汇 3,438（持平）、行数 31,063（持平）——本刀是**零度量的可加性前置**
     （把最后一个字段搬进 C 状态记录）；文件数 8 不变。
     ④ **oracle（含一次环境失败）**：旧库 = 提交 `0b48503` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与
     5,514/5,514 逐行保序完全一致**（未过滤 5,281/5,281 与 5,808/5,806 → `processGLState.slow` **300/300 与 294/292**），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**。
     ⚠️ **环境事故如实记录**：第一次跑 `make test-all` 时 `verify-gl-api` 因 `bash scripts/fetch_opengl_registry.sh`
     访问 GitHub 失败（`Error in the HTTP2 framing layer`）而返回 `Error 2`；**重跑即恢复 `GATE=0`**。
     这条与代码无关，但按纪律必须写出——**门禁失败先看是不是网络/工具链，再怀疑自己的改动**。

116. **P0-1 第六十刀：`mgl_draw_metal_port.m` 收尾第一簇（tessellation 状态读写 11 个函数，**剩余 51 个局部**）**：
     ① 按第 115 条第②项确立的纪律**改按簇人工推进**，本刀完成第一簇（tessellation 状态设置/查询）：
     `mglStageBeginNativeTES`、`mglStageEndNativeTES`、`mglStageResetTessDrawState`、`mglStageSetTessCapture`、
     `mglStageSetControlPointIndex`、`mglStageAdoptCaptureAsTCS`、`mglStageSetCurrentFactors`、`mglStageGetTessCapture`、
     `mglStageGetTcsOutput`、`mglStageGetFactors`、`mglStageGetPatchOut`（共 11 个）。
     手法固定为三步（每簇一次编译）：① 删掉 `MGLRenderer *self = mglStageHostSelf(renderer);` 与 `if (!self) return …;`，
     换成 `if (!renderer) return …;` + 就地取状态区；② `self->_tessellation.X` → `areas.tessellation->X`、
     `self->_backend` → `areas.backend`；③ 断言簇内**不再出现 `self`**（脚本里 `assert 'self' not in text`，
     这是本刀唯一"防呆"手段，比事后靠编译器更快）。
     ② **一个重要的度量认识（写下来避免后人误判）**：`MGLRenderer *self/host` 这类**局部**并不计入 `objc_zero.sh` 的"ObjC 语法"
     （计数器认的是 `id`/`__bridge`/`#import`/`NS*`/`@` 等记号），所以**清这 67 个函数不会直接移动语法数字**；
     它们的价值是**让文件能改名 `.c`（文件 8 → 7）**。该文件剩下的 25 处语法其实只有两类：**4 个 `#import`** 与 **约 19 个 `__bridge`**。
     ③ **度量**：语法 1,671、词汇 3,438 **均持平**；行数 31,063 → 31,076（+13，换状态区取用比原来多两行/函数）；
     该文件行数 1,983 → 1,996；**剩余 `mglStageHostSelf(renderer)` 局部 67 → 51**、`mglDrawHostSelf` 未动；
     文件数 8 不变。
     ④ **oracle**：旧库 = 提交 `3edbf89` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,279/5,272 与 5,808/5,804 → `processGLState.slow` 298/291 与 294/290），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ ⚠️ **环境事故（本轮实测）**：本刀的 CTS 跑到 hotspot 簇时 **`/private/tmp` 被整体清空**
     （`p03i_battery.log`、`ablib60/` 与所有 `base_*.txt` 全部消失，仅剩几个系统 socket），
     于是：① **A/B 结果在清空前已取得并记录**（下表照抄自清空前的实测输出，可信）；
     ② **CTS 只完成了 hotspot 簇**，其余六簇未跑；③ 基线文件用**上一刀已核验为"diff 全空"的 `p03h` 运行**重建
     （`base_*.txt`，计数 58/1/0/59/13/39/4 与长期基线一致），用例表 `refq/piq/compute/pp-cases.txt` 由 `p03h` 的 `summary.tsv` 重建；
     ④ A/B 工具（`ab_full.py`、`run_ab.sh`、battery 脚本）一并重建，且 comparator 内置了纪律：
     **只认"去掉 `processGLState.slow` 后的逐行相等"，并同时打印 slow 计数**。
     **规则：`/private/tmp` 只是缓存，任何"验证证据"必须在同一轮里抄进本文档，不能只留在 tmp。**
     ⑥ 下一刀：继续下一簇（建议顺序：`mglStageGetControlPointIndex`/`SetControlPointIndex` 一带 → `mglGsMetal*` 一带
     → `mglDrawHost*` 一带 → 最后 `mglStageHostSelf`/`mglDrawHostSelf` 两个助手自身删除），每簇 5–11 个函数、每簇一次编译；
     全部清完后 `git mv MGL/src/mgl_draw_metal_port.m MGL/src/mgl_draw_metal_port.c`、把 4 个 `#import` 换成 `#include`
     （`+DrawSupportUtil.h` 已是 C 可用头，另两个 ObjC 头需要的声明要么已由 C 头提供、要么补进 `mgl_renderer_ports.h`），
     并去掉随之失效的 `__bridge`（**注意第 107 条：先确认那份 `+1` 原本归谁**）。

117. **P0-1 第六十一刀：`mgl_draw_metal_port.m` 收尾第二至四簇（再 18 个函数，**剩余 33 个局部**）**：
     ① 三簇合并一刀（每簇编译一次，全部通过）：
     - **第二簇（tessellation 查询，8 个）**：`mglStageGetControlPointIndex`、`mglStageEncoderOwner`、
       `mglStageGetTcsOutVerts`、`mglStageGetTcsOutStride`、`mglStageGetTessCaptureOff`、`mglStageGetTessInstRecords`、
       `mglStageGetTessIndexed`、`mglStagePendingGsActive`；
     - **第三簇（stage 辅助与缓冲创建，6 个）**：`mglStageMarkCbHasWork`、`mglStageProcessBuffer`、`mglStageCreateBuffer`、
       `mglStageCreateBufferBytes`、`mglStageCachedFactors`、`mglStageNativeFactors`；
     - **第四簇（pending GS 查询与 GS 缓冲，4 个）**：`mglStagePendingGsInput`、`mglStagePendingGsOff`、
       `mglStagePendingGsStride`、`mglGsMetalMtlForBuffer`。
     手法与第 116 刀一致（早退 + 就地取状态区 + `areas.*` 字段改写），本刀新增两种形态的处理：
     `return self ? X : NULL;` → `if (!renderer) return NULL;` + 直返；`if (self) self->_batching… = 1;` →
     `if (areas.batching) areas.batching->currentCommandBufferHasWork = 1;`（保持原来的空值保护语义）。
     ② **度量**：该文件语法 **25 → 23**（顺手去掉了两处 `(__bridge void *)self`）、词汇 1（持平）、行数 1,996 → 2,021；
     全库语法 **1,671 → 1,669**、词汇 3,438（持平）、行数 31,076 → 31,101；**`mglStageHostSelf` 局部 51 → 33**；文件数 8 不变。
     ③ **oracle**：旧库 = 提交 `7fc6f19` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 **5,271/5,271** 与 5,813/5,807 → `processGLState.slow` **290/290** 与 299/293；
     default 臂两臂行数完全相同，是最干净的一次），stderr `MGL` 行 **307/307 多重集一致**；
     default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ④ 下一刀：继续剩余 33 个局部（建议下一簇：`mglGsMetal*` 一带 → `mglDrawHost*` 一带 → 最后删两个 `*HostSelf` 助手），
     全部清完后改名 `.c`（文件 8 → 7）。

118. **P0-1 第六十二刀：`mgl_draw_metal_port.m` 收尾第五、六簇（再 17 个函数，**剩余 21 个局部：stage 9 / draw 12**）**：
     ① 两簇一刀（各编译一次通过）：
     - **第五簇（GS 金属与状态，7 个）**：`mglGsMetalCmdOwner`、`mglGsMetalRecoveryOwner`、`mglGsMetalNoteDeviceReset`、
       `mglGsMetalSetExpansion`、`mglGsMetalBeginBlit`、`mglGsMetalBindingOwner`、`mglGsMetalRebindFragment`；
     - **第六簇（编码器谓词与剔除捕获，10 个）**：`mglStageEncoderHasCurrent`、`mglStageRasterEmpty`、`mglStageFullyCulled`、
       `mglStageApplyPolygonOffset`、`mglStageSetCtx`、`mglStageClearCullCapture`、`mglStageSetCullCaptureActive`、
       `mglStageStoreCullCapture`、`mglStageLoadCullCapture`、`mglStageDevicePtr`。
     ② 本刀覆盖到的映射形态又多了三种，全部按既有约定处理：
     - `_gpuRecovery.commandRecoveryOwner` → `areas.gpu_recovery_command_owner ? *areas.gpu_recovery_command_owner : NULL`；
     - `_bindingStateOwner` → `areas.binding_state_owner ? *areas.binding_state_owner : NULL`；
     - `_core.deviceResetRequested`（`_Atomic bool`）→ `areas.core->deviceResetRequested`，`atomic_store_explicit` 原样保留；
     - `self->ctx = ctx;`（写渲染器上下文的 ivar）→ **复用第 111 刀的 `mglPlatformShellSetContext(renderer, ctx)`**
       入口（这正是当初为 compute 派发加的那个），而不是新开一条路；
     - `_lastDrawPrimitiveMode` → `areas.core->lastDrawPrimitiveMode`（第 115 刀把该字段并入核心记录的收益）；
     - `mglStageDevicePtr` 里那对 `(__bridge void *)((__bridge id)…)` **两层桥接直接消掉**（返回 `void *`）。
     ③ **度量**：该文件语法 **23 → 21**、词汇 1（持平）、行数 2,021 → 2,036；全库语法 **1,669 → 1,667**、词汇 3,438（持平）、
     行数 31,101 → 31,116；**局部 stage 33 → 9、draw 12（未动）**；文件数 8 不变。
     ④ **oracle**：旧库 = 提交 `0f7bcf9` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,272/5,271 与 5,807/5,806 → `processGLState.slow` **291/290** 与 293/292），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：剩 21 个局部（stage 9 + draw 12；draw 侧含 `mglDrawHostHandleTessellation/Geometry/XFB`、
     `mglDrawHostGuardIssue{Arrays,Elements}`、`mglDrawHostRecord*Submitted`、`mglDrawHostResolve{Element,Indirect}Buffer` 等），
     预计 2 刀清完，随后删两个 `*HostSelf` 助手并改名 `.c`（**文件 8 → 7**）。

119. **P0-1 第六十三刀：`mgl_draw_metal_port.m` 收尾第七、八簇（再 10 个函数，**剩余 13 个局部：stage 6 / draw 7**）**：
     ① **第七簇（stage 侧剩余，3 个）**：`mglGsMetalFillComputeBindings`（只做空值判断，**不需要状态区**）、
     `mglStagePrepareElementIndex`、`mglStageEnsureMtlBufferPort`。
     **第八簇（draw 侧，5 个）**：`mglDrawHostSetLastPrimitiveMode`、`mglDrawHostEncoderOwner`、`mglDrawHostDevice`、
     `mglDrawHostRecordArraySubmitted`、`mglDrawHostRecordElementSubmitted`。
     ② 两处**编译期就抓到的形态细节**（值得记下，因为它们是"另一次失败的开始"）：
     - `mglStagePrepareElementIndex` 里 `mglPreparedElementIndexBuffer` 的**参数与返回类型都是 ObjC 的
       `MGLIndexMetalHandle`（`id`）**，把局部从 `id` 改成 `void *` 后两侧都要补 `(__bridge …)`——
       报错信息是 `implicit conversion … requires a bridged cast`，改回"两边都桥接、局部保持 `void *`"即可；
     - `if (host) { … }` 这类**带缩进块的**写法，改成 `if (!renderer) return;` 后必须保持块结构或把块体对齐，
       本刀用"早退 + 保留一个平凡块"的方式最小化改动面（避免再引入缩进错位）。
     ③ **度量**：该文件语法 **21 → 18**、词汇 1（持平）、行数 2,036 → 2,045；全库语法 **1,667 → 1,664**、词汇 3,438（持平）、
     行数 31,116 → 31,125；**局部 stage 9 → 6、draw 12 → 7**；文件数 8 不变。
     ④ **oracle**：旧库 = 提交 `226d034` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,275/5,276 与 5,806/5,809 → `processGLState.slow` **294/295** 与 292/295），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：剩 13 个局部（stage 6：`mglStage*` 收尾；draw 7：`mglDrawHostHandle{Tessellation,Geometry}`、
     `mglDrawHostGuardIssue{Arrays,Elements}`、`mglDrawHostBindContext`、`mglDrawHostEncode/PrepareCullDistance*`、
     `mglDrawHostWatchdog*`、`mglDrawHostResolve*`），**再来 1–2 刀即可删两个 `*HostSelf` 助手并改名 `.c`**。

120. **P0-1 第六十四刀：`mgl_draw_metal_port.m` 的 67 个类型化局部**全部清零**（局部 13 → 0）**：
     ① 本刀把**最后 13 个** `MGLRenderer *self/host = mgl*HostSelf(renderer);` 全部改写：
     `mglDrawHostHandleTessellation`、`mglDrawHostHandleGeometry`、`mglDrawHostGuardIssueArrays`、
     `mglDrawHostGuardIssueElements`、`mglDrawHostBindContext`、`mglDrawHostEncodeCullDistanceArray`、
     `mglDrawHostEncodeCullDistanceElements`、`mglDrawHostEncodeCullDistanceElementBytes`、
     `mglDrawHostPrepareEncodeCullDistanceElement`、`mglDrawHostWatchdogArrays`、`mglDrawHostWatchdogElements`、
     `mglDrawHostResolveElementBuffer`、`mglDrawHostResolveIndirectBuffer`（另有 `mglDrawHostHandleMS*` 两个内联块）。
     **`mglStageHostSelf` / `mglDrawHostSelf` 的调用点现在为 0**（只剩两个函数定义本身待删）。
     ② **三个新形态**（都由编译器逐个抓出，记下来给下一刀用）：
     - `host->ctx = ctx;`（写上下文 ivar）→ `mglPlatformShellSetContext(renderer, ctx)`（第 111 刀的既有入口）；
     - `mglRendererRenderPassManager(host)->state->X` → `areas.command ? areas.command->X : NULL`
       （`mglDrawHostRecord*Submitted` / `Watchdog*` 的日志参数里一次三连）；
     - **`METAL_LOCK()` 之后才用到 `areas`** 的两个 MS 派发块：取状态区必须插在 `METAL_LOCK()` **之前**（否则变量作用域不覆盖），
       这是"早退替换"批量脚本最容易漏的一类位置。
     ③ **度量**：该文件语法 **18 → 15**、词汇 1（持平）、行数 2,045 → 2,056；全库语法 **1,664 → 1,659**、词汇 3,438（持平）、
     行数 31,125 → 31,136；**类型化局部 13 → 0**；文件数 8 不变。
     ④ **oracle**：旧库 = 提交 `023cb7f` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,337/5,325 与 6,112/6,164 → `processGLState.slow` **356/344** 与 598/650；
     本刀的 slow 计数明显高于往轮，属已记录的运行期噪声，判定仍只看过滤后的逐行相等），stderr `MGL` 行 **307/307 多重集一致**；
     default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ **改名 `.c` 前的最后清单**（本刀查实）：① 删两个 `*HostSelf` 定义（现已无调用者）；
     ② `mglRendererRenderPassManager(MGLRenderer *r)` 与 `mglRendererBackend(MGLRenderer *r)` **两个访问器必须搬进壳 TU**
     （它们直接读 ivar `r->_renderPassManager` / `r->_backend`），C 侧的声明改为接收 `void *`；
     ③ 4 个 `#import` → `#include`（`MGLRenderer+DrawSupportUtil.h` 已是 C 可用头，另三个 ObjC 头所需的声明补进 C 头）；
     ④ 剩余 **9 处 `__bridge`**：`MGLIndexMetalHandle`/`MGLDrawMetalHandle` 两处需要 C 类型化（或保留一层薄包装），
     `.device = (__bridge void *)((__bridge id)…)` 与 `mglPreparedElementIndexBuffer` 两侧的桥接按第 107 条检查所有权后再去。

121. **P0-1 第六十五刀：**`mgl_draw_metal_port` 变成 `.c`（**文件 8 → 7**，行数 31,120 → 29,080）**：
     ① 按第 120 条第⑤项的四件清单一次做完：
     - **删掉两个 `*HostSelf` 助手**（调用点先清零；`mglDrawHostSelf`/`mglStageHostSelf` 的残留使用是
       `if (!mglXHostSelf(renderer)) …` 与 `return mglXHostSelf(renderer) ? … : …` 两类，全部改成 `renderer` 判定）；
     - **两个访问器 `mglRendererRenderPassManager(MGLRenderer *)` / `mglRendererBackend(MGLRenderer *)` 删除**
       （它们直接读 ivar），唯一外部使用者——壳的状态区填充——改为 `r->_renderPassManager->state` / `areas_out->backend = r->_backend`，
       `MGLRenderer_Private.h` 的两行声明换成说明注释（**同理：状态区取用替代访问器，字段不是端口**）；
     - **4 个 `#import` 处理**：删掉 `MGLRenderer_Private.h` / `+Draw_Private.h` / `+Tessellation_Private.h` 三个 ObjC 头，
       `+DrawSupportUtil.h` 改 `#include`（第 107 刀已把它变成 C 可用头），并补上真正的 C 依赖：
       `mgl_encode_context.h`（`MGLEncodeContext`）、`mgl_vertex_attrib_query.h`、`mgl_vertex_attrib_binding.h`、
       `mgl_draw_validate.h`、`mgl_env_flag.h`、`mgl_thread_affinity.h`、`<CoreFoundation/CoreFoundation.h>`；
     - **9 处 `__bridge` 全去**：句柄类型在两个头里本来就是 `#ifdef __OBJC__ typedef id …; #else typedef void *…; #endif`
       （`MGLDrawMetalHandle`、`MGLIndexMetalHandle`），所以 C 侧直接是 `void *`，桥接纯属多余。
     ② **三个 C 化细节**（编译器逐个逼出来，已写进日志）：
     - `METAL_LOCK()/METAL_UNLOCK()` 是渲染器私有宏、本质只是 `MGL_ASSERT_GL_THREAD()`，C 文件里补同义的两行宏；
     - `mglLogDrawWithoutSwapWatchdog` / `mglShouldInspectDrawCall` 声明在 ObjC 头里 → 按既有先例用**文件内 `extern` 原型**接上；
     - `mglVboRangeValidationEnabled()` 是 ObjC 头里的 `static inline`（**含 `#if defined(DEBUG)` 分支**）→ C 侧写**同策略 twin**，
       不能只留 env 分支；
     - 一处 `(NSInteger)` 强转改 `(int64_t)`；`MGLRenderer+DrawSupportUtil.h` 里残留的 `NSInteger` 一并改（该头现在是 C 头）。
     ③ **度量（本刀是"文件数 + 行数"的大跳）**：文件 **8 → 7**、行数 **31,120 → 29,080（−2,040）**、语法 **1,658 → 1,646（−12）**、
     词汇 **3,438 → 3,437**；`mgl_draw_metal_port.c` 自身 2,005 行（不再计入 ObjC 面）。
     ④ **oracle**：旧库 = 提交 `e6d9fd5` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 **5,271/5,275** 与 **5,806/5,805** → `processGLState.slow` **290/294** 与 **292/291**，
     两臂几乎同值），stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：按 §0.33 的表继续——`+Tessellation.m`（2,101 / 151 语法，copy-back 族与采样器入口都已就绪，
     且**它自己的 host-ops 端口已在第 108/112 刀备齐**）。

122. **P0-1 第六十六刀：`MGLRenderer+Tessellation.m` 的词汇层清零（**词汇 278 → 32**，全库词汇 3,437 → 3,191）**：
     ① 该文件 2,101 行、**151 语法 / 278 词汇**，本刀先做"零风险的一半"：
     - **35 处 `NSLog(@"…")` → `fprintf(stderr, "…\n")`**，并把随之多余的 **35 个 `@` 字符串前缀去掉**
       （先换成 `fprintf(stderr, @"…")` 再统一去 `@`；两步都编译验证过，`grep '@'` 复查后文件里只剩
       `@implementation`/`@end` 两处 ObjC 语法必需记号）；
     - `NSUInteger → size_t`、`BOOL → bool`、`nil → NULL`、`YES/NO → 1/0`。
     ② **一个重要的口径澄清（补进 §0.04 的记账面）**：`scripts/objc_zero.sh` 的**语法**计数只认
     `@interface/@implementation/@protocol/@end/@autoreleasepool/@selector/@encode/@property/@synthesize/@try/@catch/@finally/@synchronized`、
     **`[receiver selector]` 形式的消息发送**、`__bridge/__weak/__strong`、`#import` 这几类；
     **`NSLog` 与 `@""` 字符串并不计入语法**（它们计入**词汇**）。所以本刀**语法 151 不变、词汇 278 → 32**——
     这也解释了为什么"词汇层转换"在这份文档里被记为 T2 而不是语法清零手段。
     ③ **度量**：该文件词汇 **278 → 32**、语法 151（不变）、行数 2,101（不变）；全库语法 **1,646（不变）**、
     词汇 **3,437 → 3,191（−246）**、行数 29,080（不变）；文件数 7 不变。
     ④ **oracle**：旧库 = 提交 `ced649d` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,274/5,271 与 5,809/5,806 → `processGLState.slow` 293/290 与 295/292），
     stderr `MGL` 行 **307/307 多重集一致**（这正是"`NSLog` → `fprintf` 同一 sink"的既有结论在本文件的再次确认）；
     default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；**CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：该文件剩下的 **151 语法**几乎全是**消息发送**（`@implementation`/`@end`/`#import` 只占个位数），
     按第 116–120 刀对 `mgl_draw_metal_port` 的做法**按簇转 C**（`prepareTessStageBufferBindings` /
     `bindPreparedTessStageBufferBindings` / `planTessTextureBinds` / `bindPointSizeParamsToComputeEncoder` /
     `ensureTessTextureMetalData` / `flushTessStageBindingInitializationBlit` 等 11 个方法，逐个换成 `mglTess*` C 入口，
     copy-back 族与采样器入口已就绪）。

123. **P0-1 第六十七刀：`MGLRenderer+Tessellation.m` 第一个方法转 C（`ensureTessTextureMetalData:count:ctx:`，为该文件立下手法）**：
     ① 11 个方法里最小的自足方法先做示范：19 行的 `- (BOOL)ensureTessTextureMetalData:count:ctx:` →
     新 TU **`mgl_tess_texture.{h,c}`** 的 `int mglTessEnsureTextureMetalData(void *renderer, const MGLTessTextureBind *binds, uint32_t count, GLMContext draw_ctx)`。
     转换点只有两个：**唯一那处 `[self bindMTLTexture:ptr]` → `mglRendererBindMTLTexture(renderer, ptr)`**（第 100 刀的 C 入口），
     以及 `MGL_STATE(drawCtx)` → 本地 twin `mglTessTextureState(draw_ctx)`（注意这里用的是**实参** drawCtx，不是 `ctx` ivar）。
     **返回值语义照抄**：`binds/drawCtx` 为空时方法返回 `YES`（"无事可做"），C 版返回 `1`。
     ② 三个调用点（第 851/1203/1578 行，参数分别是 `tesTextureBinds` ×2、`tcsTextureBinds` ×1）改调 C 入口，
     ObjC 侧传 `(__bridge void *)self`；方法、`MGLRenderer+Tessellation_Private.h` 的声明（若有）与该方法的注释一并处理。
     ③ **度量**：该文件语法 **151 → 150**、词汇 31、行数 2,101 → 2,080；全库语法 **1,646 → 1,645**、词汇 3,191 → 3,190、
     行数 29,080 → 29,059；C 侧新增约 70 行；文件数 7 不变。
     ④ **oracle**：旧库 = 提交 `9a609f9` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与 5,514/5,514
     逐行保序完全一致**（未过滤 5,274/5,276 与 5,810/5,810 → `processGLState.slow` **293/295** 与 **296/296**），
     stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**；28 目标门禁 `GATE=0`。
     ⑤ 下一刀：同文件按同一手法继续小方法——`bindTessStageBufferBindingsToRenderEncoderOwner:`（22 行）、
     `bindPointSizeParamsToComputeEncoder:`（27 行）、`bindPreparedTessStageBufferBindings:`（31 行）、
     `flushTessStageBindingInitializationBlit:`（37 行）；四个都不需要新桥接（copy-back 族、`mglRenderSetCompute*`、
     `mglRendererBindBufferSizeConstantsForRenderEncoder` 均已就绪），做完后该文件语法可再降约 40。

### 0.49 第 91 轮实测：`+Tessellation.m` 里两个"看起来最容易"的方法其实被 ObjC 类型挡住

本轮本想按第 123 条第⑤项继续搬 `bindTessStageBufferBindingsToRenderEncoderOwner:`（22 行）与
`bindPreparedTessStageBufferBindings:…`（31 行）——两者的**方法体确实全是 C**（只调 `mglTessSetRenderVertexBuffer` /
`mglTessAppendComputeResourceOp`），但**参数类型把它们钉在 ObjC 里**：

| 阻塞 | 现状 | 结论 |
|---|---|---|
| `MGLTessStageBufferBinding` / `…List`（文件内 298–313 行定义） | 字段是 **`id __strong buffer` / `id __strong initialization_source` / `id __strong size_buffer`**，即 **ARC 管理的对象引用** | C 结构体只能收 `void *`（未持有），**等于把 ARC 的持有语义改掉**——这不是机械转换，需要"谁持有、何时释放"的专门设计与 oracle。**先不动** |
| `mglTessAppendComputeBytesOp`（211 行起，`bindPointSizeParamsToComputeEncoder:` 的唯一依赖） | 体内用 **`NSData dataWithBytes:` + `[temporaries addObject:]`** | 有干净出路：C 侧用 **`CFDataCreate(NULL, bytes, length)`** 并交给 `mglRendererTemporariesAdd()`（该入口收 `CFTypeRef`，`Create` 返回 +1 正好对上），plan 的 `bytes` 字段用 `CFDataGetBytePtr()`。**这一条可以做，是 `bindPointSizeParamsToComputeEncoder:`（27 行）的前置** |
| `id __strong` 之外，`+Tessellation.m` 绝大多数方法体已是 C | — | 说明该文件的"搬"不是逻辑问题，而是**少数几个 ObjC 类型（`id` 结构字段、`NSData`/`NSMutableArray`）的收尾问题 |

**下一刀建议**：① 先把 `mglTessAppendComputeBytesOp` 的 `NSData` 换成 `CFData`（C 版助手放 `mgl_tess_texture.c`
或新 `mgl_tess_compute_ops.{h,c}`），② 随之把 `bindPointSizeParamsToComputeEncoder:program:stage:executionPlan:temporaries:`
（27 行，两个内部调用点）转 C；③ 之后再评估 `MGLTessStageBufferBinding*` 两个结构体的 `id __strong` 是否值得
改成"ObjC 侧持有 + C 侧只读 `void *`"（若改，必须在同一刀里给出持有/释放的完整清单与 A/B 之外的验证口径）。

**本轮未改动代码**：按纪律不把树留在半成品状态——探查用的新 TU 已删除，`git status` 干净，工作区仍在 `c3d07fa`（第 123 刀已验证状态）。

124. **P0-1 第六十八刀：`+Tessellation.m` 的 compute-plan 帮手与点尺寸参数转 C（`NSData` → `CFData`）**：
     ① 按 §0.49 的建议落地"可做的那一条"：新 TU **`mgl_tess_compute_ops.{h,c}`**：
     - `bool mglTessAppendComputeBytesOp(plan, temporaries, bytes, length, index)` —— 原静态助手的 `NSData` 换成
       **`CFDataCreate(NULL, bytes, length)`**（+1），交给 `mglRendererTemporariesAdd()`（该入口正收 `CFTypeRef`），
       随后 **`CFRelease` 自己那份**——与 ARC 版"数组持有、强局部在作用域末尾释放"严格配对；
       plan 的 `bytes` 字段用 **`CFDataGetBytePtr(storage)`**（集合持有期间指针有效 ✓）；
     - `void mglTessBindPointSizeParamsToComputeEncoder(renderer, program, stage, plan, temporaries)` ——
       原方法的 `ctx`/`MGL_STATE(ctx)` → 状态区取用 + dual-proxy twin，其余（`_MAX_SHADER_TYPES` 边界、`uses_point_size_params`、
       `mglTessFillPointSizeParams`、`kMGLPointSizeBufferIndex`）原样。
     ② 调用点（第 1164/1556 行的 TCS 与 TES 两处）改调 C 入口；被搬走的静态助手与
     方法本体（约 60 行）+ 其唯一的调用点一并从 ObjC 文件删除。
     ③ **度量**：该文件语法 150（持平，因为搬走的是一处 `[self …]` 与其定义，新增的 C 侧不计）、
     词汇 **27 → 27**、行数 2,080 → **2,025**；全库语法 **1,645（持平）**、词汇 3,190 → **3,186（−4）**、
     行数 29,059 → **29,004（−55）**；C 侧新增约 120 行；文件数 7 不变。
     ④ **oracle**：旧库 = 提交 `dc151fe` 的独立构建（`cmp` 两库不同）；两臂 trace **确定性行 4,981/4,981 与
     5,514/5,514 逐行保序完全一致**（未过滤 **5,273/5,273** 与 5,808/5,807 → `processGLState.slow` **292/292** 与 294/293，
     default 臂两臂完全相同），stderr `MGL` 行 **307/307 多重集一致**；default 臂 **92/0/2**、flushy 臂 **91/1/2**（两臂同值）；
     **CTS 七簇非通过集合 diff 全空**。
     ⚠️ 本轮 `make test-all` 第一次仍因 `scripts/fetch_opengl_registry.sh` 访问 GitHub 失败（`Error in the HTTP2 framing layer`）
     返回 `Error 2`，**重跑即 `GATE=0`**——与第 115 条同一个环境噪声，处置同上。
     ⑤ 下一刀：`+Tessellation.m` 剩 **150 语法**，其中约 40 来自四个小方法（`bindTessStageBufferBindingsToRenderEncoderOwner:`
     被 `MGLTessStageBufferBinding*` 的 `id __strong` 字段挡住，见 §0.49；另外 `flushTessStageBindingInitializationBlit:`、
     `bindPreparedTessStageBufferBindings:`、`planTessTextureBinds:` 需要先处理同族结构体或 `NSMutableArray`），
     其余来自四个大方法（283/269/692/130 行）。**建议下一刀处理 `MGLTessStageBufferBinding*` 的
     `id __strong` 改造**（设计好"ObjC 侧持有 + C 侧只读 `void *`"的清单），因为它一次性解锁 4 个小方法。

### 0.50 第 93 轮实测：给大文件做 `NSLog` → `fprintf` 批量转换时，**格式串里的括号会骗过配平扫描**

第 122 刀在 `+Tessellation.m` 上做 `NSLog → fprintf` 是成功的（35 处），本轮想在 `MGLRenderer+BindingState.m`
（2,916 行 / **28 处 `NSLog`** / 35 `NSUInteger` / 52 `BOOL` / 44 `YES|NO`）复制同一手法，**两次脚本都失败、已全部回退**，
根因值得记下来：

| 尝试 | 做法 | 失败原因 |
|---|---|---|
| ① | 先把 `NSLog(@` 换成 `fprintf(stderr, @`，再用"找到 `")` 再插 `\n`"补换行 | 有的语句结尾不是 `")`（格式串后面还有实参），断言直接失败 |
| ② | 用**配平括号**扫出整个 `NSLog(…)`，再按"首个参数是格式串"重建 | 配平扫描把**字符串字面量里的括号**也算成括号——例如格式串 `"written=[%lld,%lld)"` 里的 `)` 会让扫描提前结束，截断后的 `inner` 无法匹配"格式串 + 其余实参"的正则 |

**正确做法（下一刀照此）**：扫描前先**把字符串字面量与注释屏蔽掉**（保留字符位置），用屏蔽后的副本做括号配平，
再回到原文切片重建；或者退一步**只处理单行 `NSLog`**（`grep -n 'NSLog' |` 每处都在一行内），逐条 `edit` 而不写批量脚本。
**这一条与第 116 条"按簇人工改"同源**：`MGL/` 里剩下的 5 个大文件（`+RenderPass.m` 6,842、`+Texture.m` 6,489、
`MGLRenderer.m` 4,616、`+Blit.m` 4,062、`+BindingState.m` 2,916）**不适合一次性正则批量改**，
每簇/每类记号单独做、每步编译。

**本轮未改动代码**：两次尝试都在写盘前失败（`git status` 干净），工作区仍在 `3daac52`（第 124 刀已验证状态）。

### 0.51 第 94 轮交接：从这里继续（**新会话/新轮次请先读本小节**）

**状态快照（本轮实测，工作区 `5d14092`）**：文件 **7**（53 起）、空 TU 0、行数 **29,004**（43,989 起）、
ObjC 语法 **1,645**（2,268 起）、词汇 **3,186**（4,353 起）；端口 33、唯一壳 TU **2,054 行**（上限 2,400，块表见 §0.29）。
七个文件与本轮实测的语法/词汇/行数：

| 文件 | 语法 | 词汇 | 行数 | 下一刀建议 |
|---|---|---|---|---|
| `MGLRenderer+RenderPass.m` | 385 | 553 | 6,842 | 厚块：按簇搬（encoder 生命周期 → 状态处理 → attachment/persistent） |
| `MGLRenderer+Texture.m` | 289 | 1,086 | 6,489 | 厚块：先 upload/readback 簇，再 `createMTLTextureFromGLTexture:`（转完退役第 100 刀的 4 个纹理端口） |
| `MGLPlatformRendererShell.m`（唯一壳） | 289 | 287 | 2,054 | 见 §0.29：`MGLPipelineCache` 类转 C handle（阻塞=归档路径，需专用 oracle）、lifecycle/壳类为终态平台面 |
| `MGLRenderer+Blit.m` | 235 | 761 | 4,062 | 厚块：blit/copy/resolve 簇 |
| `MGLRenderer.m` | 168 | 275 | 4,616 | 主体类，最后做 |
| `MGLRenderer+Tessellation.m` | 150 | 27 | 2,025 | 四小方法被 `MGLTessStageBufferBinding*` 的 `id __strong` 挡住（§0.49）；先做该结构体改造 |
| `MGLRenderer+BindingState.m` | 129 | 197 | 2,916 | 采样器级联 + stage copy-back；`NSLog` 批量转换见 §0.50 的**屏蔽字符串**配方 |

**每轮闭环（照抄即可）**：
1. 改代码：**按簇人工改、每簇 `clang -fsyntax-only` 或 `make -j8`**；`NSLog` 类批量转换先屏蔽字符串/注释再配平（§0.50）；
   删头文件后必须 `find build/core build/es -name '*.o' -o -name '*.d' | xargs rm -f`（§0.101③）；**永不 `make clean`**。
2. 两个库：`make -j8`（无 error）。
3. 门禁：`make test-all` → 需 `GATE=0` 且 `PASS 92 / FAIL 0 / SKIP 2`；**首跑若在 `verify-gl-api` 因
   `scripts/fetch_opengl_registry.sh` 访问 GitHub 失败（`Error in the HTTP2 framing layer`）→ 直接重跑**（§0.115④）。
4. 旧库：`git -C /Users/fterward/MGL-ab-old checkout --detach <上一提交>` + 清 `.o/.d` + `make -j8 lib`；
   `cmp` 两个 `libmgl.dylib` 必须**不同**。
5. A/B：`/private/tmp/run_ab.sh <ablibN> {new,old}` + `python3 /private/tmp/ab_full.py <new_dir> <old_dir> <new_txt> <old_txt>`；
   **判定只看"去掉 `processGLState.slow` 后的逐行相等"**，并如实报 slow 计数与 stderr `MGL` 行多重集（307/307）。
6. CTS：`TAG=<tag> /private/tmp/run_t4b_battery_z.sh`（七簇），与 `/private/tmp/base_*.txt` 逐簇比对**非通过集合 diff 必须为空**
   （基线计数长期为 58 / 1 / 0 / 59 / 13 / 39 / 4）。
7. 文档三处同步（§0.0 进度行、§5 新增日志条目、§0.2x/§0.3x/§0.4x/§0.5x 快照），**把验证证据抄进文档**
   （`/private/tmp` 会被清空，见 §0.116⑤）。
8. 提交推送：`git push origin main:main`；若 22 端口超时（`Please make sure you have the correct access rights`），
   用 `git -c url."ssh://git@ssh.github.com:443/53453450/MGL-minecraft.git".insteadOf="git@github.com:53453450/MGL-minecraft.git" push origin main:main`。

**已知阻塞（都需要专门设计，不要用脚本硬推）**：
- `MGLTessStageBufferBinding` / `…List` 的 **`id __strong` 字段**（§0.49）→ 需"ObjC 持有 + C 只读 `void *`"的清单与验证口径；
- 壳内 `MGLPipelineCache` 类的**归档路径**（Foundation：`NSSearchPath…`/`NSBundle`/`NSFileManager`/`NSURL`）→
  转 C 会改变归档文件名与 `NSError` 文案，而 **A/B 恰好过滤 `BINARY ARCHIVE` 行**（§0.103①）→ 必须先建专用 oracle；
- `syncResourceBindingsForContext:` 一类"成本倒挂"项（§0.23）→ 除非顺带解锁整文件删除，否则不动。

### 0.52 第 95 轮：预算末轮的收束说明与"下一批必须整块搬"的最后清单

**本轮复验（未改代码，`df24b67`）**：两个库构建无错、`make test-all` **0**（`PASS 92 / FAIL 0 / SKIP 2`）；
文件 **7**、行数 **29,004**、语法 **1,645**、词汇 **3,186**；端口 33、壳 2,054 行。
A/B 与 CTS 的口径证据见第 124 条（同一份代码状态；其后两笔提交都只改文档）。

**语法构成实测（决定下一批怎么做）**——本轮把剩下两个"中块"文件的语法来源拆开数了一遍：

| 文件 | 语法总数 | `[receiver selector]` 消息发送 | `__bridge` | `#import` | `@implementation/@end` |
|---|---|---|---|---|---|
| `MGLRenderer+BindingState.m` | 129 | 43 | **78** | 3 | 2 |
| `MGLRenderer+Tessellation.m` | 150 | 58 | **80** | 5 | 2 |

**结论**：这两个文件的语法**大头是 `__bridge`（60%）**，而 `__bridge` 是"把 ObjC 指针交给 C 入口"的桥，
**只有把宿主代码本身搬成 C 才能连带消掉**——单个 `__bridge` 无法"就地删除"（删了就编不过）。
所以下一批**不能按"小方法"零敲**（本周期第 123/124 刀证明：搬 19–27 行的方法只能降 0–1 个语法），
**必须整块搬**：一次搬掉一个方法簇 + 其 `self`/`__bridge`/`id` 使用面，让语法数字成批下降。

**建议的整块切口（按"语法/行数比"排序）**：
1. `MGLRenderer+BindingState.m` 的 **stage buffer 绑定簇**（`bindStageBufferMapEntriesForStage:` /
   `bindStageFallbackBuffersForStage:` / `finalizeStageBufferPresentMask:` 等，约 400 行、含约 20 个 `__bridge`）；
2. `MGLRenderer+Tessellation.m` 的 **`planTessTextureBinds:` + 三个 `mglTessPlan*`/`mglTessCreateSampler` 静态助手**
   （约 120 行，`id texture/sampler` + `NSMutableArray` → `void *` + keep-alive 集，机制在第 124 刀已跑通）；
3. `MGLRenderer+Blit.m` 的 **采样拷贝/解析簇**（`mgl_blit_sampled_copy.c` 已经存在，可继续把同族方法搬过去）。

**收束说明**：本轮次预算（90 轮）已到末轮，目标**未达成**且**不是阻塞**（没有"同一外部阻塞连续三轮"的情形，
只是剩余量大：5 个厚文件约 24k 行 / 约 1,150 语法，需要大量轮次）。**目标保持 active**，
按 §0.51 的八步闭环与本节的三条整块切口继续即可。

125. **P0-1 第六十九刀：`+Tessellation.m` 的「细分 stage 绑定 / 纹理绑定规划」整簇转 C（新 TU `mgl_tess_stage_bind.{h,c}`）**：
     ① 按 §0.52 的结论**不再按"小方法"零敲，改整块搬**。本刀搬走的是 §0.49 记的那一族阻塞点：
     - 两个原先**只在本文件 `@implementation` 内部**的私有记录 `MGLTessStageBufferBinding` / `...List`
       （含 `id __strong buffer`、`id __strong size_buffer`）→ 落到 C 头 `mgl_tess_stage_bind.h`，
       `id __strong` → `void *`、`BOOL` → `int`。**这正是 §0.49 里"挡住 4 个小方法"的那块石头**：
       字段所有权不变（创建出来的 +1 直接存进记录字段，ARC 的 `__strong` 语义等价）。
     - 五个方法转 C：`-prepareTessStageBufferBindings:stage:copyBacks:`（205 行 / 19 语法）→
       `mglTessPrepareStageBufferBindings`；`-flushTessStageBindingInitializationBlit:` → `mglTessFlushStageBindingInitializationBlit`；
       `-bindTessStageBufferBindingsToRenderEncoderOwner:bindings:` → `mglTessBindStageBufferBindingsToRenderEncoderOwner`；
       `-bindPreparedTessStageBufferBindings:toComputeEncoder:executionPlan:temporaries:` → `mglTessBindPreparedStageBufferBindings`；
       `-planTessTextureBinds:count:ctx:plan:temporaries:`（13 语法）→ `mglTessPlanTextureBinds`。
     ② **没有新增任何端口**：`self` → renderer 句柄；`_renderPassManager->state` / `_tessellation.tessVertexRenderActive`
     → `MGLRendererStateAreas`（`areas.command` / `areas.tessellation`，第 113 刀的那条路正是为此铺的）；
     裸 `ctx` → **`areas.ctx`**；`MGL_STATE(ctx)` → **C twin**（与 `mgl_compute_bind.c` 同一写法）；
     `[self clearStageBindingCopyBack:atIndex:]` / `[self recordStageBindingCopyBack:…]` / `[self endRenderEncoding]`
     → **既有端口** `mglRendererClearStageBindingCopyBackPort` / `mglRendererRecordStageBindingCopyBackPort` /
     `mglRendererEndRenderEncodingPort`；`NSMutableArray *temporaries` → **`void *` + `mglRendererTemporaries*`**
     （该入口收 `CFTypeRef`，句柄本身就是那个数组）。
     ③ **.m 里保留 `id` 版静态助手，C TU 用同名不同前缀的 twin**：`mglTessCreateBuffer` / `…WithBytes` / `…CreateSampler` /
     `…BufferContents` / `…AppendComputeResourceOp` / `…PlanTextureOrBind` / `…PlanSamplerOrBind` /
     `…EncodeBufferCopiesForOwner` / `…SetRenderVertexBuffer` 都只裹一层 `mglRender*` C 入口，若把 .m 的改成 `void *`
     反而会在**剩余的 3 个大方法**（约 40 处调用点）上加 `__bridge`，得不偿失；先例是 `mgl_draw_metal_port.c` 的
     `mglVboRangeValidationEnabled` twin。等三个大方法也搬走，两组 twin 自然合并。
     ④ **所有权口径（本刀最容易错的地方）**：`mglTessPlanTextureBinds` 里那条 `mglTessCreateSampler()` 路径是 **+1**，
     原 ARC 代码靠 `__bridge_transfer` 在作用域末尾释放；C 侧改为「交给 temporaries 集合（集合自己 retain）后
     **显式 `CFRelease` 自己那份**」，**失败分支也要释放**（已写在代码注释里）。
     `mglTextureCreateSamplerForTexParam` 在 C 侧**不再需要 `CFBridgingRetain`**（它本身返回 +1，ARC 版多一次 retain
     是为了抵消 ARC 的自动释放）。
     ⑤ **度量**：`MGLRenderer+Tessellation.m` **2,024 → 1,608 行**、语法 **150 → 130**、词汇 **27 → 17**；
     全库行数 **29,004 → 28,587（−417）**、语法 **1,645 → 1,625（−20）**、词汇 **3,186 → 3,176（−10）**、
     文件数 **7 不变**；新 C TU `mgl_tess_stage_bind.c` **580 行** + 头 **92 行**。
     **语法只降 20（不是搬走的 40）要说清楚**：三个 `.m` 侧的调用点被改写成 C 调用，每处多一个 `(__bridge void *)self`
     / `(__bridge void *)executionTemporaries`（共 9 个），加上两个 `prepareTessBuffer…` 调用点从"1 行消息发送"变成
     "4 行 C 调用 + 1 个 `__bridge`"，净值就是 −20；**这不是回归，是"搬运过程的成本"，等 `.m` 里最后 3 个大方法搬完即消失**——
     与 §0.52 的判断一致（`__bridge` 只能靠宿主文件本身转 C 才消失）。
     ⑥ **oracle（本刀自建）**：旧库 = 提交 `6de380f` 的独立 worktree 构建（`cmp` 两库不同：`libmgl` 4,083,120 vs 4,099,216 字节）；
     `/private/tmp/run_ab.sh` 两臂 + `ab_full.py` 全文比对：**default 臂确定性行 4,981/4,981 逐行一致**（未过滤 5,282/5,280 →
     `processGLState.slow` **301/299**，按既定政策不构成信号）、**flushy 臂 5,514/5,514 一致**（`slow` 335/308）；
     **stderr MGL 行 307/307 多重集一致**；两臂通过数 **default 92/0/2、flushy 91/1/2**（与基线一致）。
     ⚠️ 第一次 A/B 两臂**都**以 `dyld: Library not loaded: @rpath/libglfw.dylib` 退出 134 —— 是**我漏拷 `libglfw.dylib` 进 A/B 目录**
     的环境问题（两臂同时失败就是证据），补齐后即通过；记在此处以免下次再判成回归。
     ⑦ **CTS 七簇**：`hotspot / tess / gs / refq / piq / compute / pp` 非通过集合 **diff 全空**，计数仍是
     **58 / 1 / 0 / 59 / 13 / 39 / 4**（`LC_ALL=C sort` 后比对；注意 `base_*.txt` 是旧排序，不统一 locale 会出现
     `clip_control.` vs `clip_control_ARB` 的**假 diff**）。
     ⑧ **下一刀**：`+Tessellation.m` 剩 **130 语法 / 1,608 行**，只剩 3 个方法 ——
     `-dispatchTessControlShader:`（40 语法）、`-dispatchAIRTessEvalVertexRender:`（32）、`-dispatchAIRTessEvalCompute:`（55），
     外加 `-newTCSStageInBufferForContext:`（19）。**它们的外部调用者只有壳**（`MGLPlatformRendererShell.m` 的
     `mglRendererDispatchTessControlShader/AIRTessEvalCompute/AIRTessEvalVertexRender` 三个端口），
     把三个方法转 C 就能**同时退役这 3 个端口并整文件删掉这个 `.m`（文件数 7 → 6）**，是当前"一次收益最大"的切口。

### 0.53 第 96 轮交接快照（**新会话请先读本节 + §0.51**）

**当前状态**：`MGL/` 内 ObjC **7 个文件 / 0 空 TU / 28,587 行 / 1,625 语法 / 3,176 词汇**；壳 TU 2,054 行（上限 2,400）；
端口 33 个；`make test-all` **0**、CTS 七簇 **diff 全空**、A/B 两臂一致（见第 125 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,842 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · `+BindingState.m` 129/197/2,916 · 壳 `MGLPlatformRendererShell.m` 289/287/2,054 ·
`+Tessellation.m` **130/17/1,608**。

**下一步（按"一刀的收益/风险比"排序）**：
1. **`+Tessellation.m` 收尾（最高优先）**：见第 125 条 ⑧。搬 `-newTCSStageInBufferForContext:` + 三个 dispatch 方法，
   顺带退役壳里 3 个端口、删掉整个 `.m`。注意两个 AIR 方法里还有 `mglTessPlanBufferOrBind` /
   `mglTessCreateSampler` / `mglTessSetRenderVertex*` 那批 `id` 版 twin，会一起搬进 C；
   `planTessTextureBinds` 的两处调用点（现在已是 C 调用）改回 `mglTessPlanTextureBinds` 即可（签名不变）。
2. **`+BindingState.m` stage buffer 绑定簇**（§0.52 的整块切口 ①）：它的语法 60% 是 `__bridge`，
   整簇转 C 才能成批下降；先抽盘再逐簇编译。
3. **`+Blit.m` 采样拷贝/解析簇**：`mgl_blit_sampled_copy.c` 已在，同族继续搬。
4. 壳的 `MGLPipelineCache` 归档路径需要 Foundation→POSIX 改造 + **专属 oracle**（A/B 过滤 `BINARY ARCHIVE` 行）。

126. **P0-1 第七十刀：TCS dispatch 入口转 C 并**净退役 1 个端口**（新 TU `mgl_tess_dispatch.{h,c}`）**：
     ① 按 §0.53 的排序落地"收益最大"的切口的前半段：把 `+Tessellation.m` 的
     `-dispatchTessControlShader:program:contract:`（248 行 / 40 语法）与其唯一被调者
     `-newTCSStageInBufferForContext:…`（130 行 / 19 语法）搬进新 TU，分别成为
     `bool mglTessDispatchControlShader(renderer, glm_ctx, program, contract)` 与文件内静态
     `mglTessDispatchNewTCSStageInBuffer(...)`（返回 +1 的 stage_in buffer）。
     ② **端口净减 1**：`mglRendererDispatchTessControlShaderPort` 的整个存在理由就是"把 C 侧调用转发到 ObjC 方法"，
     方法转 C 后它连同壳里的 12 行包装一起删除，`mgl_draw_metal_port.c` 的 `mglStageDispatchTCS` 改为直调
     `mglTessDispatchControlShader`。按 §0.04 的 T4 硬规（无净减＝拒收），本刀**计入 T4 进度**，端口 **33 → 32**。
     另外两个 AIR TES 端口（`…AIRTessEvalComputePort` / `…AIRTessEvalVertexRenderPort`）仍指向 `.m` 里的方法，**本刀不动**。
     ③ **两处所有权是这刀的全部风险，已在代码注释里写明**：
     - 计算管线句柄是 **+1**，ARC 版靠 `(__bridge_transfer id)tcsPipeline` 在每条 return 路径上释放；
       C 版改为**单一 `done:` 标签**收口（`CFRelease` 一次），因此整个方法体从"15 处 early return"重构成
       `bool ok = false; … goto done;`；
     - temporaries 集合是 `mglRendererTemporariesCreate()` 的 **+1**，同样在 `done:` 释放——放在
       `mglRenderExecuteComputeExecutionPlan` 之后（该调用**同步**编码，命令缓冲此后持有这些 buffer）；
     - copy-back 列表原本在**每条**路径各清一次，C 版在 `done:` 清一次（该清理＝backend 列表复位 + `memset`，幂等）；
       成功路径上它与"保存 tess factor buffer 给 TES patch-draw"这两步的相对次序对调了，两者操作的是 backend 的
       不同槽位，且 A/B 全文 trace 逐行一致（见 ⑤）——**这条次序改动已在 oracle 里被证伪不了**，故记录在案。
     ④ `-newCommandBuffer`（`METAL_LOCK` + `-newCommandBufferLocked` + `METAL_UNLOCK`）在 C 侧走
     `mglRendererNewCommandBufferLockedPort` + `MGL_ASSERT_GL_THREAD()`——与 `mgl_draw_metal_port.c` 对
     `METAL_LOCK` 的既有 twin 一致（该宏只断言 GL 线程）。
     ⑤ **oracle（本刀自建）**：旧库 = 提交 `a8f74f3` 的独立 worktree 构建（`cmp` 两库不同：4,082,800 vs 4,082,224 字节）；
     `ab_full.py` 全文比对：**default 臂确定性行 4,981/4,981 逐行一致**（未过滤 5,281/5,282 → `slow` **300/301**）、
     **flushy 臂 5,514/5,514 一致**（`slow` 321/298，按既定政策不是信号）；**stderr MGL 307/307 多重集一致**；
     两臂 **default 92/0/2、flushy 91/1/2**。**该文件的两个 dispatch 方法覆盖了 CTS 的 tess 簇与 `air_geometry_xfb`
     回归，trace 逐行一致即说明 TCS 编码路径（含 stage_in 打包、间接参数、tess factor 默认值、copy-back 收集）没有偏移。**
     ⑥ **度量**：`MGLRenderer+Tessellation.m` **1,608 → 1,215 行**、语法 **130 → 85**、词汇 17 → 15；
     壳 `MGLPlatformRendererShell.m` **2,054 → 2,042 行**、语法 289 → 287（退役端口包装）；
     全库行数 **28,587 → 28,182（−405）**、语法 **1,625 → 1,578（−47）**、词汇 3,176 → **3,174**、文件数 7 不变；
     新 C TU `mgl_tess_dispatch.c` **470 行** + 头 **40 行**。
     ⑦ **CTS 七簇**：非通过集合 **diff 全空**，计数仍是 **58 / 1 / 0 / 59 / 13 / 39 / 4**；`make test-all` **0**（92/0/2/94）。
     ⑧ **下一刀**：`+Tessellation.m` 只剩 **85 语法 / 1,215 行** 的两个方法
     （`-dispatchAIRTessEvalVertexRender:` 32 语法、`-dispatchAIRTessEvalCompute:` 55 语法）与那批 `id` 版静态助手。
     把它们搬进 `mgl_tess_dispatch.c` 后，该 `.m` 可**整文件删除（文件数 7 → 6）**，同时退役剩下的两个 AIR 端口
     （端口 **32 → 30**）。这是下一个"一刀双收益"的切口，且两个方法的调用者同样只有壳端口一处。

### 0.54 第 97 轮交接快照（**新会话请先读本节 + §0.51**）

**当前状态**：`MGL/` 内 ObjC **7 个文件 / 0 空 TU / 28,182 行 / 1,578 语法 / 3,174 词汇**；壳 TU **2,042 行**（上限 2,400）；
端口面 **32 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂一致（见第 126 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,842 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · `+BindingState.m` 129/197/2,916 · 壳 `MGLPlatformRendererShell.m` 287/287/2,042 ·
`+Tessellation.m` **85/15/1,215**。

**下一步（同 §0.53，第 1 项已推进一半）**：
1. **`+Tessellation.m` 收尾**：把两个 AIR TES dispatch 方法搬进 `mgl_tess_dispatch.c`，删掉整个 `.m`，
   退役两个 AIR 端口。注意这两个方法里还有一批 `id` 版静态助手（`mglTessPlanBufferOrBind`、`mglTessCreateSampler`、
   `mglTessSetRenderVertex{Bytes,Texture,Sampler,Buffer}`、`mglTessDrawPrimitives`、`mglTessBufferContents`、
   `mglTessCreateBuffer{,WithBytes}`、`mglTessBufferLength`、`mglTessTextureInfo`、`mglTESXFBVertexStride`），
   它们与 `mgl_tess_dispatch.c`/`mgl_tess_stage_bind.c` 里已有的 C twin 合并后即可整文件删除。
2. **`+BindingState.m` stage buffer 绑定簇**（§0.52 整块切口 ①）。
3. **`+Blit.m` 采样拷贝/解析簇**（`mgl_blit_sampled_copy.c` 已在）。
4. 壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

127. **P0-1 第七十一刀：AIR TES-vertex render 入口转 C，**再净退役 1 个端口**（端口 32 → 31）**：
     ① 把 `-dispatchAIRTessEvalVertexRender:program:contract:patchCount:instanceCount:baseInstance:`（283 行 / 32 语法）
     搬进 `mgl_tess_dispatch.c`，成为 `bool mglTessDispatchAIRTessEvalVertexRender(renderer, glm_ctx, program, contract,
     patch_count, instance_count, base_instance)`；`mglRendererDispatchAIRTessEvalVertexRenderPort` 与其壳包装（16 行）
     一并删除，`mgl_draw_metal_port.c` 的 `mglStageDispatchAirTESVertex` 改直调。
     ② **本刀新写的 C twin**：`mglTessDispatchBufferLength`、`mglTessDispatchCreateSampler`、
     `mglTessDispatchSetRenderVertex{Bytes,Texture,Sampler,Buffer}`（都直落 `mglRenderSet*ForOwner`）、
     `mglTessDispatchDrawPrimitives`（`MGLRenderDrawPlan` + `mglRenderEncodeDrawForRenderEncoderOwner`）。
     ③ **所有权/顺序**：函数静态「只报一次」标志 `s_tes_vertex_multi_instance_logged` 落到文件作用域；
     采样器 `mglTessDispatchCreateSampler()` 的 **+1 在"已交给渲染编码器"之后立刻 `CFRelease`**（ARC 版是同一轮循环末尾释放，
     两者生命周期边界相同）；`glSampler->mtl_data` 的采样器创建在 C 侧**不再需要 `CFBridgingRetain`**（返回值本就是 +1）。
     ④ **顺带清理**：该方法搬走后 `.m` 里 6 个静态助手（`mglTessCreateBufferWithBytes` / `mglTessCreateSampler` /
     `mglTessTextureInfo` / `mglTessSetRenderVertex{Bytes,Texture,Sampler}`，共 59 行）**只剩定义、没有调用者**，
     而它们的 C twin 已经存在，故一并删除——编译告警 `-Wunused-function` 由 6 条回到 0 条。
     ⑤ **oracle（本刀自建）**：旧库 = 提交 `b923109` 的独立 worktree 构建（`cmp` 两库不同）；`ab_full.py`：
     **default 臂确定性行 4,981/4,981、flushy 臂 5,514/5,514 逐行一致**（未过滤 5,284/5,293 与 5,814/5,821 →
     `slow` 303/312 与 300/307，按既定政策不是信号）；**stderr MGL 307/307 多重集一致**；两臂 **92/0/2、91/1/2**。
     该方法是 isolines/point-mode TES 的 **render-vertex 路径**（CTS tess 簇与 `air_geometry_xfb` 覆盖），
     逐行一致说明 per-patch drawPrimitives、tess factor/patch_out/indirect 槽位绑定与 point-size 参数写入都没有偏移。
     ⑥ **度量**：`MGLRenderer+Tessellation.m` **1,215 → 866 行**、语法 **85 → 62**、词汇 15 → 12；
     壳 **2,042 → 2,026 行**、语法 287 → 285；全库行数 **28,182 → 27,817（−365）**、语法 **1,578 → 1,553（−25）**、
     词汇 **3,174 → 3,171**；端口 **32 → 31**；`mgl_tess_dispatch.c` **533 → 927 行**、头 43 → 53 行；文件数 7 不变。
     ⑦ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑧ **下一刀（收尾刀）**：`+Tessellation.m` 只剩 **62 语法 / 866 行** 的**最后一个方法**
     `-dispatchAIRTessEvalCompute:program:contract:patchCount:instanceCount:baseInstance:`（55 语法 / 约 690 行）
     与 8 个静态助手。搬进 `mgl_tess_dispatch.c` 后即可：**整文件删除该 `.m`（文件数 7 → 6）**、
     退役最后一个 AIR 端口（**31 → 30**）、并删除 `MGLRenderer+Tessellation_Private.h` 里的方法声明
     （它只被 `MGLRenderer_Private.h` 聚合导入）。这是本周期最后一块"细分"拼图，也是"文件数下降"的关键一刀。

128. **P0-1 第七十二刀：`MGLRenderer+Tessellation.m` 整文件消失（文件数 7 → 6），并发现"ARC 隐式 retained 局部"这条铁律**：
     ① 把最后一个方法 `-dispatchAIRTessEvalCompute:program:contract:patchCount:instanceCount:baseInstance:`（约 690 行 / 55 语法）
     搬进 `mgl_tess_dispatch.c`（927 → 1,763 行），成为 `mglTessDispatchAIRTessEvalCompute(...)`；
     随后**整个 `MGLRenderer+Tessellation.m` 与本周期第一个被删的 category 一起消失**，`MGLRenderer+Tessellation_Private.h`
     也一并删除（它只被 `MGLRenderer_Private.h` 聚合导入，改成一行注释说明归属）。
     ② **端口 31 → 31（退役 1、新增 1，按 §0.04 本刀 T4 进度为 0，如实记账）**：
     退役 `mglRendererDispatchAIRTessEvalComputePort`（目标方法已转 C）；但新方法内部调用
     `-[MGLRenderer ensureAIRTessEvalPassthroughFunctionForProgram:]`，该方法仍在 `MGLRenderer+RenderPass.m` 里，
     故新增 **`mglRendererEnsureAIRTessEvalPassthroughPort`**（8 行包装，随 `+RenderPass.m` 转 C 时退役）。
     ③ ⚠️ **本刀最重要的产出是一条铁律**：**ObjC 的 `id` 局部变量是"隐式 retain"，改成 `void *` 后它就只是裸指针**。
     第一次构建后 `make test-all` 在 `air_tessellation_isolines_xfb` **SIGSEGV**，lldb 回溯指向
     `mglTessDispatchBufferContents(xfb_copy_destination)`（`MTL::Buffer::contents` 收到已释放对象）：
     原方法里 `id xfbCopyDestination` 是 ARC 强引用，`glBufferSubData` 把 `buf->data.mtl_data` 换成快照后，
     旧 Metal buffer 仍被 ARC 局部引用着；C 版没有这个引用，于是对象被释放、镜像写入时崩。
     修复方式（已写进代码注释与本节）：**所有"原本是 ARC 强局部"的句柄统一登记进 temporaries 集合**
     （借用句柄 `mglTessDispatchKeepAlive`，自己 +1 创建的 `mglTessDispatchAdopt`＝入集合后放掉创建引用），
     集合在**单一 `done:` 标签**释放——这正是 ARC 的"作用域末尾释放"语义。
     ④ **同一个坑踩了第二次**：同一个测试第二次崩溃在**分离属性 XFB 分支**的
     `mglTessDispatchBufferContents(dest_mtl)`——那里也有一个 `id destMTL` 强局部，同样在 `mglBufferSubData` 之后使用；
     一并登记进保活集合后通过。**这条"逐个 `id` 局部都要判一次"的检查已作为下一批的强制步骤**（见 §0.55）。
     ⑤ 顺带修掉**前两刀遗留的引用泄漏**：`mglTessDispatchControlShader` 与 `mglTessDispatchAIRTessEvalVertexRender`
     里"创建 +1 后又 `addObject:`"的句柄（`tcs_output_buffer` / `tcs_patch_out_buffer` / `indirect_buffer` /
     `tess_factor_buffer` / `out_buffer` / 自建 stage_in buffer）此前只加不减，每次 draw 泄漏一个引用；
     现改为 `mglTessDispatchAdopt`。vertex-render 方法同时重构成**单出口**（`ok` + `done:`）以便统一释放。
     ⑥ **oracle**：旧库 = 提交 `65216b2` 的独立 worktree 构建（`cmp` 两库不同）；`ab_full.py`：
     **default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,274/5,282 与 5,812/5,818 → `slow` 293/301、298/304，
     按既定政策不是信号）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⚠️ **A/B 与 `make test-all` 都没能发现 ③ 的崩溃，是 CTS 的 tess 簇先报出来的**（第一次跑 CTS 时
     `KHR-GL46.tessellation_shader.single.max_patch_vertices` 由 pass 变 **crash**）——
     说明"回归套件 + A/B 双绿"**不等于**没有生命周期回归，CTS 七簇是必需的第三方证据。
     ⑦ **CTS 七簇（修复后）**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4），
     其中 `KHR-GL46.tessellation_shader.single.max_patch_vertices` 与 `air_tessellation_isolines_xfb` 均回到通过。
     ⑧ **度量**：ObjC **文件数 7 → 6**、行数 **27,817 → 26,943（−874）**、语法 **1,553 → 1,491（−62）**、
     词汇 **3,171 → 3,159（−12）**；壳 `MGLPlatformRendererShell.m` 2,026 → **2,018 行 / 285 语法**；
     端口 31（不变）；`mgl_tess_dispatch.c` **1,763 行** + 头 61 行。
     ⑨ **下一刀**：按 §0.55 的排序走 `+BindingState.m` 的 stage buffer 绑定簇（129 语法，60% 是 `__bridge`），
     **搬之前先把该簇里每一个 `id` 局部列出来判"谁保活"**。

### 0.55 第 98 轮交接快照（**新会话请先读本节 + §0.51**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 26,943 行 / 1,491 语法 / 3,159 词汇**；壳 TU **2,018 行 / 285 语法**（上限 2,400）；
端口面 **31 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（见第 128 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,842 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · `+BindingState.m` 129/197/2,916 · 壳 `MGLPlatformRendererShell.m` **285/287/2,018**。

**⚠️ 本周期最重要的铁律（第 128 条 ③④，下一批必须逐条执行）**：
把 ObjC 方法体搬成 C 时，**每一个 `id` 局部变量都要判一次"它现在靠谁活着"**：
1. 来自 `mglRendererBackendGet*` / 结构体字段（`buf->data.mtl_data`）的**借用句柄** → 立刻登记进保活集合
   （`mglTessDispatchKeepAlive` 模式：`mglRendererTemporariesAdd`，集合在单一出口释放）；
2. 自己用 `mglRenderCreate*` 拿到的 **+1 句柄** → 入集合后放掉创建引用（`mglTessDispatchAdopt` 模式），
   **不要**只 `addObject:` 不加释放（那是每次 draw 泄漏一个引用，第 128 条 ⑤ 修的就是这个）；
3. 方法体有多条 early return 时，**先把它改成单出口**（`bool ok` + `done:`），否则释放点必然漏；
4. **特别注意"调用之后还会用到"的句柄**：`glBufferSubData` / `mglRendererBufferSubData` 会把
   `buf->data.mtl_data` 换成快照，之后的 `mglTessBufferContents(old)` 必须有保活引用（两处崩溃都是这一类）。
**验证顺序也要照旧**：`make test-all` 与 A/B **双绿不足以**发现生命周期回归（第 128 条 ⑥），
必须跑 CTS 七簇再看非通过集合 diff。

**下一步排序**：
1. `+BindingState.m` 的 stage buffer 绑定簇（129 语法 / 2,916 行；先做 §0.52 的整块切口 ①）。
2. `+Blit.m` 采样拷贝/解析簇（`mgl_blit_sampled_copy.c` 已在）。
3. `+Texture.m`（289 语法 / 1,086 词汇，词汇最多，需要 `NSLog`→`fprintf` 与 `BOOL`→`bool` 的成批转换）。
4. `+RenderPass.m`（385 语法，最大块）与 `MGLRenderer.m`；`mglRendererEnsureAIRTessEvalPassthroughPort`
   随 `+RenderPass.m` 的 `ensureAIRTessEvalPassthroughFunctionForProgram:` 转 C 一起退役。
5. 壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

129. **P0-1 第七十三刀：`+BindingState.m` 的 stage buffer 绑定驱动整块转 C（新 TU `mgl_stage_buffer_bind.{h,c}`）**：
     ① 搬走两个"逐 stage 绑定驱动"方法：
     `-bindStageBufferMapEntriesForStage:isFragmentStage:bufferMapList:anyBindingPresent:baseBindingPresent:attribBindingReserved:encodeContext:bindingSnapshot:byteScratch:byteScratchUsed:byteScratchCapacity:useSnapshot:maxMetalSlots:allowIsolateWhenGpu:needsCopyBackOnIsolate:`
     （230 行）→ `mglBindingStateBindStageBufferMapEntries(...)`；
     `-bindStageFallbackBuffersForStage:…`（108 行）→ `mglBindingStateBindStageFallbackBuffers(...)`；
     两个仍留在 `.m` 的编码驱动（`-bindVertexBuffersToCurrentRenderEncoder:` / `-bindFragmentBuffersToCurrentRenderEncoder:`）
     改为直调 C 入口（调用点 4 处）。
     ② **宏按既有办法下沉成"小上下文函数"**（`mgl_compute_bind.c` 的先例）：
     `MGL_BIND_SNAP_FLUSH/_COLLECT_BUFFER/_COLLECT_BYTES` → `mglSbSnapFlush/mglSbSnapCollectBuffer/mglSbSnapCollectBytes`；
     `MGL_BIND_STAGE_EMIT_BUFFER/_EMIT_BYTES/_UPDATE/_PERF_SKIP/_CLEAR_BINDING` → `mglSbStageEmitBuffer/mglSbStageEmitBytes/mglSbStageUpdate/mglSbStagePerfSkip/mglSbStageClearBinding`；
     方法内的 `MGL_SMB_*` / `MGL_SFB_*` 两族宏直接展开成对这些函数的调用（不再需要 `#undef` 收尾）。
     其它机械替换：`self`→renderer、`ctx`→`areas.ctx`、`_backend`→`areas.backend`、
     `_bindingStateOwner`→`*areas.binding_state_owner`、`_tessellation.nativeTESCopyBacks`→`areas.tessellation->nativeTESCopyBacks`、
     `(__bridge id)`→裸句柄、`NSLog`→`fprintf(stderr, …)`、`NSInteger/NSUInteger`→`int64_t/size_t`、`YES/NO/nil`→`1/0/NULL`；
     `kMGLVerboseBindLogs`（ObjC 头里的 `getenv("MGL_VERBOSE_BIND") != NULL` 宏）与 `kMGLDefaultStageFallbackBufferSize`、
     `kMGLStageBindingStackScratchSize` 在 C TU 里各写一个等价 twin。
     ③ ⚠️ **按 §0.55 的铁律逐条处理了 `id` 局部**：本簇唯一的 ARC 强局部是 `isolated`
     （`-isolatedStageBindingBufferForMap:…` 的返回值），而端口 `mglRendererIsolatedStageBindingBufferPort`
     **返回 +1（方法当年返回 +0）**。C 侧没有 autorelease pool 兜底，快照里存的还是裸指针，
     故按 `mgl_compute_bind.c` 的既有处置：**在隔离缓冲仍存活时就 flush 快照**（`use_snap` 时），
     之后 `mglSafeReleaseMetalObj(&isolated)` 放掉端口给的 +1。多一次编码是**保序**的（同一批 op、同一个 encoder），
     与 A/B 全文逐行一致（见 ⑤）——这一取舍已写进代码注释。
     ④ 另一个既有先例：`mglRendererGetValidatedBuffer` 只在 ObjC 私有头里声明，按 §0.51 的规则在 C TU 里重复 `extern` 原型；
     `mglBindingStateIsValid` / `mglBindingStateBufferMatches` 是 ObjC 私有头里的 `static inline`，在 C TU 里写成 `mglSbBindingState*` twin。
     ⑤ **oracle（本刀自建）**：旧库 = 提交 `a685b07` 的独立 worktree 构建（`cmp` 两库不同；
     ⚠️ 旧 worktree 因上一刀删了私有头，`.d` 陈旧依赖会让 make 报 `No rule to make target`，**先清 `.o/.d` 再编**）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,314/5,331 与 6,118/6,157 →
     `slow` 333/350、604/643，按既定政策不是信号）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⚠️ 本轮 `make test-all` 第一次仍撞上 `scripts/fetch_opengl_registry.sh` 访问 GitHub 失败（`Couldn't connect to server`），
     **重跑即 0**——第 115/124 条同一环境噪声，处置相同。
     ⑥ **度量（如实记录：本刀 T4 进度为 0）**：`+BindingState.m` **2,916 → 2,553 行**、语法 **129 → 117**、词汇 **197 → 158**；
     全库行数 **26,943 → 26,580（−363）**、语法 **1,491 → 1,479（−12）**、词汇 **3,159 → 3,120（−39）**；
     文件数 **6 不变**；端口 **31 → 31（0 退役 0 新增）**——本刀用的是既有端口（隔离缓冲 / copy-back / 状态判据），
     按 §0.04 的 T4 硬规**不计 T4 进度**，收益是行数与词汇（`BOOL/YES/NO/nil/NSUInteger/NSLog` 一簇）。
     **语法只降 12 的原因**：这两个方法的语法大头是 20 余处 `__bridge` 强转与消息发送，转 C 后它们变成
     同样数量的 `(void *)` 转换（不再是语法），但 4 处调用点各新增一个 `(__bridge void *)self`，净值即 −12。
     ⑦ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑧ **下一刀**：`+BindingState.m` 仍剩 **117 语法 / 2,553 行**，按 §0.56 排序走"采样纹理簇"
     （`applySampledCompatFallbackPlan:` / `materializeSampledSamplerForTexture:` / `applySampledRenderTargetCopyPlan:` /
     `recoverFragmentSampledDepthTexture:` / `emitSampledDiagPortsForProgram:`），
     它**能顺带退役既有端口 `mglRendererMaterializeSampledSamplerPort`**，是本文件里少见的"净减端口"机会。

### 0.56 第 99 轮交接快照（**新会话请先读本节 + §0.51 + §0.55**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 26,580 行 / 1,479 语法 / 3,120 词汇**；
壳 TU **2,018 行 / 285 语法**（上限 2,400）；端口面 **31 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（见第 129 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,842 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · `+BindingState.m` **117/158/2,553** · 壳 `MGLPlatformRendererShell.m` **285/287/2,018**。

**`+BindingState.m` 的剩余方法（按"能退役端口/整块"排序）**：
1. **采样纹理簇（下一刀）**：`recoverFragmentSampledDepthTexture:`（L1611 起，23 语法 / 316 行）、
   `emitSampledDiagPortsForProgram:`（13 语法）、`applySampledCompatFallbackPlan:`、
   `materializeSampledSamplerForTexture:`、`applySampledRenderTargetCopyPlan:`。
   **退役既有端口 `mglRendererMaterializeSampledSamplerPort`**；注意 `applySampledCompatFallbackPlan:` 里
   `[self fallbackSampledTextureForExpectedType:dataKind:]` 与本文件外的采样回退路径仍需保留 ObjC 调用点（先做端口）。
2. **属性/顶点簇**：`bindVertexAttributesFromVAO:`（19 语法 / 344 行）、`bindVertexBuffersToCurrentRenderEncoder:`（16 语法）、
   `bindFragmentBuffersToCurrentRenderEncoder:`（2 语法）、`bindPointSizeParamsIfNeeded:`（3 语法）、
   `finalizeStageBufferPresentMask:`（1 语法，注意 `NSLog` 与 trace 助手）。
3. **存储镜像簇**：`bindStorageImagesForStage:`（10 语法）、`bindStorageImagesForVertexProgram:`、
   `bindSeparateSamplersAndArrayTextures:`（17 语法）、`bindSampledTexturesForStage:`（17 语法，1611 行里最大）。
   它们转 C 后可一并退役 `mglRendererBindStorageImagesForVertexProgramPort`。

**每刀开工前必做（§0.55 铁律 + 本轮新增）**：
1. 列出该簇里**每一个 `id` 局部**并判保活（借用→`KeepAlive`；自建 +1→`Adopt`；多 early return 先改单出口）；
2. 检查**端口返回值与新代码的引用计数**是否一致——本轮的 `mglRendererIsolatedStageBindingBufferPort` 是 **+1**，
   而它替换的方法当年返回 **+0**，这一类"端口比方法多一个 +1"必须显式放掉（且要先保证快照已 flush）；
3. 宏密集处按 `mgl_compute_bind.c` / `mgl_stage_buffer_bind.c` 的"小上下文函数"办法下沉，不要试图让 C TU 包含 ObjC 私有头；
4. 收尾顺序固定：两个库构建（无新告警）→ `make test-all` → **CTS 七簇** → A/B 全文比对 → 度量 → 文档三处同步 → 提交推送。
   ⚠️ 已知环境噪声两个：`fetch_opengl_registry.sh` 访问 GitHub 失败（重跑即过）；旧 worktree 的 `.d` 陈旧依赖（先清 `.o/.d`）。

130. **P0-1 第七十四刀：storage image 绑定驱动转 C，**净退役 1 个端口（31 → 30）**，并靠 CTS 抓出"缓存了编码器属主"这个真 bug**：
     ① `-bindStorageImagesForStage:program:bindStage:` 与 `-bindStorageImagesForVertexProgram:fragmentProgram:`
     搬进新 TU **`mgl_storage_image_bind.{h,c}`**（292 行 + 头 49 行）：
     `mglBindingStateBindStorageImagesForStage` / `mglBindingStateBindStorageImagesForVertexProgram`。
     **端口净减 1**：`mglRendererBindStorageImagesForVertexProgramPort` 与其壳包装（10 行）一并删除，
     `mgl_draw_metal_port.c` 的 GS 路径改直调（按 §0.04，本刀**计入 T4 进度**）。
     ② 按 §0.55/§0.56 的清单逐条处理：文件内静态助手 `mglBindingStateResourceAtOrdinal` /
     `mglBindingStateCreateStorageImageView` 在 C TU 里写 twin（后者的最后一个 ObjC 调用者随本刀消失，故从 `.m` 删除）；
     ObjC 私有头里的 `static inline`（`mglBindingStateQueueResourceBinding` / `…FlushResourceBindings` /
     `…CollectResourceBinding`）与 `MGL_ABORT_TBIND_IF_ENCODER_CLOSED()` 同样写 twin；
     `MGL_STATE(ctx)` 用 `mglSiState(areas)`（core.activeState 优先，再回落 `ctx->active_state`）；
     `NSLog` 沿用 `fprintf`，`RETURN_FALSE_ON_FAILURE` 仍是 C 头 `glm_context.h` 的宏。
     本簇无 ARC 强局部（`texture` 是借用视图、快照 flush 在本函数内完成，编码器那时已 retain）。
     ③ ⚠️ **真 bug（CTS 抓出）**：第一次跑七簇时 compute 簇出现一处 diff——
     `KHR-GL46.shader_image_load_store.advanced-memory-order` 由 `fail` 变成 **pass**。追查发现是**我的翻译错**：
     原方法在每次使用时都重新读 `_renderPassManager->state->currentRenderEncoderOwner`，
     而我在函数开头把它**缓存成局部变量**；ENSURE pass 里 `restoreRenderEncoderAfterTextureUploadForDraw:`
     会**替换**渲染编码器，缓存值此后就是**过期的属主**，入队/flush 打到错的编码器上。
     这正是 §0.14 里 `binding_state_owner` 的那条规矩（"值可能在驱动运行中变化，必须在使用点解引用"）——
     **对 `currentRenderEncoderOwner` 同样成立**，已写成 `mglSiRenderEncoderOwner(&areas)` 逐点读取。
     修复后该用例回到 `fail`（与旧库一致，见 ④ 的两臂复跑）。
     ④ **flaky 用例登记（新增，与 `processGLState.slow` 同级）**：
     `KHR-GL46.shader_image_load_store.advanced-memory-order` **本身是抖动用例**——同一份二进制
     单独跑 5 次全 `fail`，而某次整簇电池里却记成 `pass`。**受控实验**：用电池同一套机制
     （`run_mgl_cts_cases.py` + 同一 caselist/workdir/参数）分别跑 new/old 两臂各一次，**两臂都是 `fail`**；
     重跑整簇电池后 compute 簇也回到**diff 为空**。结论：该用例的状态**不能作为单次信号**，
     判读非通过集合 diff 时必须先做"两臂受控复跑"再定性（本轮已按此办理）。
     ⑤ **oracle**：旧库 = 提交 `0ac1248` 的独立 worktree 构建（`cmp` 两库不同；旧 worktree 先清 `.o/.d` 再编）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,300/5,304 与 5,895/5,875 →
     `slow` 319/323、381/361，按既定政策不是信号）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑥ **度量**：`+BindingState.m` **2,553 → 2,445 行**、语法 **117 → 109**、词汇 **158 → 155**；
     壳 `MGLPlatformRendererShell.m` **2,018 → 2,008 行**、语法 285 → 283；
     全库行数 **26,580 → 26,462（−118）**、语法 **1,479 → 1,469（−10）**、词汇 **3,120 → 3,117（−3）**；
     文件数 **6 不变**；**端口 31 → 30**。新增 C 面 292 + 49 行。
     ⑦ **CTS 七簇（修复后重跑）**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑧ **下一刀**：`+BindingState.m` 剩 **109 语法 / 2,445 行**，按 §0.57 继续"采样纹理簇"
     （可退役既有端口 `mglRendererMaterializeSampledSamplerPort`），注意 `applySampledCompatFallbackPlan:` 里的
     `fallbackSampledTextureForExpectedType:dataKind:` 与 `applySampledRenderTargetCopyPlan:` 里的
     `freshGLSampledRenderTargetCopyForSampling:` 目前**没有**端口——要么先补端口（会让端口数 +2），
     要么把这两个方法一起搬（推荐：一次搬完，避免端口净增）。

### 0.57 第 100 轮交接快照（**新会话请先读本节 + §0.51 + §0.55 + §0.56**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 26,462 行 / 1,469 语法 / 3,117 词汇**；
壳 TU **2,008 行 / 283 语法**（上限 2,400）；端口面 **30 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（见第 130 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,842 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · `+BindingState.m` **109/155/2,445** · 壳 `MGLPlatformRendererShell.m` **283/287/2,008**。

**两条新增判读规矩（第 130 条）**：
1. **`currentRenderEncoderOwner` 必须在使用点重新读取**，不能像 `binding_state_owner` 那样缓存——
   纹理上传路径会 restore（替换）渲染编码器；缓存值会让入队/flush 打到过期属主（本轮真 bug）。
2. **CTS 单次状态不是判决**：`KHR-GL46.shader_image_load_store.advanced-memory-order` 是抖动用例
   （同二进制单独 5 次全 fail、电池里可能记 pass）。任何非通过集合 diff 都要先做**两臂受控复跑**
   （`run_mgl_cts_cases.py` + 同一 caselist/workdir）再定性。

**下一步排序**：
1. `+BindingState.m` 采样纹理簇：`recoverFragmentSampledDepthTexture:`（23 语法 / 316 行）、
   `emitSampledDiagPortsForProgram:`（13）、`applySampledCompatFallbackPlan:`、`materializeSampledSamplerForTexture:`、
   `applySampledRenderTargetCopyPlan:`——**一次搬完**（后两个依赖的 `fallbackSampledTextureForExpectedType:dataKind:`、
   `freshGLSampledRenderTargetCopyForSampling:` 没有端口，分开搬会让端口净增），
   搬完可退役 `mglRendererMaterializeSampledSamplerPort`（→ 29）。
2. `+BindingState.m` 顶点/属性簇（`bindVertexAttributesFromVAO:` 19 语法、两个编码驱动 16+2、`finalizeStageBufferPresentMask:`）。
3. `+Blit.m` 采样拷贝/解析簇；`+Texture.m`；`+RenderPass.m` / `MGLRenderer.m`。
4. 壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

131. **P0-1 第七十五刀：顶点/片元 stage 绑定编码驱动整块转 C，**净退役 2 个端口（30 → 28）**，并记下"选择器调用点要全树搜"**：
     ① 五个方法一次搬完，进新 TU **`mgl_stage_encode_drivers.{h,c}`**（1,051 行 + 头 56 行）：
     - `-bindVertexBuffersToCurrentRenderEncoder:`（16 语法）→ **`mglStageEncodeBindVertexBuffers`**；
     - `-bindVertexAttributesFromVAO:…`（19 语法 / 344 行）→ 文件内静态 `mglStageEncodeBindVertexAttributes`；
     - `-bindPointSizeParamsIfNeeded:…`（3）→ 静态 `mglStageEncodeBindPointSizeParams`；
     - `-bindFragmentBuffersToCurrentRenderEncoder:`（2）→ **`mglStageEncodeBindFragmentBuffers`**；
     - `-finalizeStageBufferPresentMask:…`（1）→ 静态 `mglStageEncodeFinalizePresentMask`。
     **端口净减 2**：`mglRendererBindVertexBuffersToCurrentRenderEncoderPort` 与
     `mglRendererBindFragmentBuffersToCurrentRenderEncoderPort` 及其壳包装一并删除；
     C 侧三处调用者（`mgl_batch_dyn_bind_encode.c` / `mgl_binding_state_ops.c` / `mgl_draw_metal_port.c`）改直调
     （按 §0.04 本刀**计入 T4 进度**）。
     ② 机械替换按既定规则：`self`→renderer、`ctx`→`areas.ctx`、`_backend`/`_device`→`areas.backend`、
     `_batching`→`areas.batching`、`_bindingStateOwner`→`*areas.binding_state_owner`、
     `MGL_BIND_SNAP_*` / `MGL_BIND_STAGE_*` / 方法内的 `MGL_VATTR_*`·`MGL_VPS_*` 宏 → `mglSe*` 函数、
     `NSLog`→`fprintf(stderr, …)`、`NSMutableData` 的 packed-current 池 → **`calloc` 的等长零填充块**（用完即 `free`，
     与 ARC 局部同时机）、ObjC 私有头里的 `static inline`（`mglBindingStateIsValid` / `…BufferMatches` /
     `…TextureSlotCount` / `mglShouldTraceCall`）与常量（`kMGLEnableVertexAllSlotFallback=1`、
     `MGL_BINDING_RESOURCE_STORAGE_SHARED=0`、`kMGLCurrentAttrib*`）各写等价 twin。
     `__FUNCTION__` 这类"给校验器当标签"的实参**照抄选择器字符串**（stderr 文本是 A/B 判据的一部分）。
     ③ ⚠️ **踩坑与规矩**：删掉方法后 `make test-all` 立刻崩在
     `-[MGLRenderer bindVertexBuffersToCurrentRenderEncoder:]: unrecognized selector`——
     调用点在 **`MGLRenderer+RenderPass.m`（3+3 处）**，而我此前只按"端口名 + 方法定义"搜过。
     **新规矩：转换一个方法前，必须全树搜该选择器的所有调用点（`.m` 与 `.c` 都要），
     不能只搜端口名**；本轮已把 6 处改为直调 C 入口后 `make test-all` 回到 0。
     ④ **oracle**：旧库 = 提交 `28b5a5f` 的独立 worktree 构建（`cmp` 两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,340/5,334 与 6,113/6,126 →
     `slow` 359/353、599/612）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑤ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑥ **度量**：`+BindingState.m` **2,445 → 1,681 行**、语法 **109 → 87**、词汇 **155 → 85**；
     壳 `MGLPlatformRendererShell.m` **2,008 → 1,988 行**、语法 283 → 279；
     全库行数 **26,462 → 25,679（−783）**、语法 **1,469 → 1,443（−26）**、词汇 **3,117 → 3,047（−70）**；
     文件数 **6 不变**；**端口 30 → 28**；新增 C 面 1,051 + 56 行。
     ⑦ **下一刀**：`+BindingState.m` 只剩 **87 语法 / 1,681 行**，即"采样纹理簇"
     （`bindTexturesToCurrentRenderEncoder:`、`bindSampledTexturesForStage:`、`recoverFragmentSampledDepthTexture:`、
     `emitSampledDiagPortsForProgram:`、`applySampledCompatFallbackPlan:`、`materializeSampledSamplerForTexture:`、
     `applySampledRenderTargetCopyPlan:`、`bindSeparateSamplersAndArrayTextures:`）。
     开工前先按 §0.57 的排序确认端口账：`materializeSampledSamplerForTexture:` 有既有端口可退役（−1），
     但 `fallbackSampledTextureForExpectedType:dataKind:`（在 `+Texture.m`）与
     `freshGLSampledRenderTargetCopyForSampling:`（在 `+Blit.m`）都没有端口——**整簇一次搬**才能避免端口净增。

### 0.58 第 101 轮交接快照（**新会话请先读本节 + §0.51 + §0.55 + §0.57**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 25,679 行 / 1,443 语法 / 3,047 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（见第 131 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 289/1,086/6,489 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 ·
`+BindingState.m` **87/85/1,681**。

**三条开工规矩（累积，按 §0.55 / §0.57 / §0.58）**：
1. **每个 `id` 局部判保活**：借用→`KeepAlive`、自建 +1→`Adopt`、多 early return 先改单出口；
   端口返回值与方法返回值可能差一个 +1（`mglRendererIsolatedStageBindingBufferPort` 就是 +1）。
2. **`currentRenderEncoderOwner` 在使用点重新读取**，任何"会 restore 编码器"的调用之后都不能用缓存值。
3. **转换前全树搜该选择器的所有调用点**（`.m` + `.c`，不只搜端口名）；删方法后 `make test-all` 是最后一道网。
   另：CTS 单次状态不是判决（`KHR-GL46…advanced-memory-order` 是抖动用例），diff 异常先做**两臂受控复跑**。

**下一步排序**：
1. `+BindingState.m` **采样纹理簇**（87 语法全部在此）：`bindTexturesToCurrentRenderEncoder:`（4）、
   `bindSampledTexturesForStage:`（17）、`recoverFragmentSampledDepthTexture:`（23）、
   `emitSampledDiagPortsForProgram:`（13）、`applySampledCompatFallbackPlan:`、
   `materializeSampledSamplerForTexture:`、`applySampledRenderTargetCopyPlan:`（9）、
   `bindSeparateSamplersAndArrayTextures:`（17）。**整簇一次搬**，搬完 `+BindingState.m` 即整文件消失（文件数 6 → 5）
   且可退役 `mglRendererMaterializeSampledSamplerPort`。注意 `fallbackSampledTextureForExpectedType:dataKind:`
   （`+Texture.m`）与 `freshGLSampledRenderTargetCopyForSampling:`（`+Blit.m`）没有端口，需在**同一刀**里一并搬或补端口。
2. `+Blit.m` 采样拷贝/解析簇（235 语法）。
3. `+Texture.m`（289 语法 / 1,086 词汇，词汇最多）。
4. `+RenderPass.m`（385 语法，最大块）与 `MGLRenderer.m`（168）。
5. 壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

132. **P0-1 第七十六刀：采样纹理回退链转 C（`+Texture.m` 的 5 个方法），为 `+BindingState.m` 的收尾刀解开"无端口被调者"这个结**：
     ① 新 TU **`mgl_sampled_fallback.{h,c}`**：把 `+Texture.m` 里的
     `-fallbackSampledTextureForExpectedType:dataKind:`（68 行）、`-fallbackSampledTextureForExpectedType:`（13）、
     `-fallbackSampledTexture`（38）、`-fallbackCubeSampledTexture`（40）、`-fallbackTextureBufferSampledTexture`（65）
     整体搬成 C：`mglSampledFallbackTextureForExpectedType(renderer, expected_type, data_kind)` /
     `mglSampledFallbackTextureForType` / `mglSampledFallbackTexture` / `mglSampledFallbackCubeTexture` /
     `mglSampledFallbackTextureBuffer`。**动机是端口账**：这五个方法是 `+BindingState.m` 采样簇里
     `applySampledCompatFallbackPlan:`（4 处调用点）唯一的"外部且无端口"被调者；先把它转 C，
     收尾刀就**不需要新增端口**。本刀**端口 28 → 28（T4 中性，如实记账）**。
     ② 机械替换：`self`/`_backend`/`_device` 走状态区；`+Texture.m` 的文件静态
     （`mglTextureCreateTexture` / `…CreateBuffer` / `…CreateBufferTexture` / `…BufferContents` / `…ReplaceRegion`）
     与枚举（`MGL_TEXTURE_USAGE_SHADER_READ=1`、`MGL_TEXTURE_RESOURCE_STORAGE_SHARED=0`、
     `kMGLEnableSampledTextureFallback=YES`）各写 C twin；`NSLog` → `fprintf(stderr, …)`；
     `%@`/`[NSString stringWithUTF8String:]` → `%s`。
     **`@try/@catch`（纹理缓冲纹理创建）改用既有端口 `mglPlatformShellGuardedCallCtx`**（§0.14 路线 ③，
     该回调签名是 `int (*)(void *renderer, void *ctx)`，签名不符会直接编译报错，本轮已修）。
     ③ **所有权（§0.128 铁律的又一次应用）**：5 个入口全部**返回借用指针**（与原方法 ARC 的 +0 一致）：
     `mglRenderCreateTextureFromState` 等给出 **+1**，写进 backend 缓存（Set/Put 内部 retain）之后
     **放掉自己那份**；失败路径同样放掉再返回 NULL。四个调用点改调 C 入口后，`+BindingState.m` 里每处多一个
     `(__bridge id)`（ARC 需要显式桥接）——**这也是该文件语法 87 → 91（+4）的原因**，全库仍降 14。
     ④ **oracle**：旧库 = 提交 `baebfe9` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,323/5,323 与 6,116/6,115 →
     `slow` 342/342、602/601，本刀两臂几乎相同）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⚠️ 本轮 `make test-all` 又一次先撞 GitHub 网络噪声（`fetch_opengl_registry.sh`），**重跑即 0**。
     ⑤ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑥ **度量**：`+Texture.m` **6,489 → 6,256 行**、语法 **289 → 271**、词汇 **1,086 → 1,052**；
     `+BindingState.m` 1,681 → 1,678 行、语法 87 → 91（见 ③）、词汇 85 → 85；
     全库行数 **25,679 → 25,443（−236）**、语法 **1,443 → 1,429（−14）**、词汇 **3,047 → 3,013（−34）**；
     文件数 **6 不变**；端口 **28 不变**；新增 C 面约 330 + 40 行。
     ⑦ **下一刀**：`materializeSampledSamplerForTexture:`（4 语法 / 68 行）→ C 并**退役既有端口
     `mglRendererMaterializeSampledSamplerPort`（28 → 27）**；随后按 §0.59 把其余采样方法逐个搬走，
     最后 `+BindingState.m` 整文件删除（6 → 5）。

### 0.59 第 102 轮交接快照（**新会话请先读本节 + §0.51 + §0.55 + §0.58**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 25,443 行 / 1,429 语法 / 3,013 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（见第 132 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 ·
`+BindingState.m` **91/85/1,678**。

**`+BindingState.m` 收尾清单（8 个方法 / 91 语法）与其"外部被调者"账**（这是决定端口增减的唯一变量）：
| 方法 | 语法 | 外部被调者 | 现状 |
|---|---|---|---|
| `bindTexturesToCurrentRenderEncoder:` | 4 | `bindSampledTexturesForStage:`（同文件） | 同簇一起搬即可 |
| `bindSampledTexturesForStage:` | 17 | 同文件 5 个 + `mglBindingStateResourceAtOrdinal`（弱依赖） | 同簇 |
| `recoverFragmentSampledDepthTexture:` | 23 | `[self bindMTLTexture:]`（有 C 入口）、fallback 链（**已 C**） | 可搬 |
| `emitSampledDiagPortsForProgram:` | 13 | `[self traceSampledTextureReadback:…]`（`+Texture.m`，**无端口**，参数含 `NSString`） | **需先转它或补端口** |
| `applySampledCompatFallbackPlan:` | 0 | fallback 链（**已 C**，本刀解决） | 可搬 |
| `materializeSampledSamplerForTexture:` | 4 | 全部 C | **可搬并可退役 1 端口** |
| `applySampledRenderTargetCopyPlan:` | 9 | `[self freshGLSampledRenderTargetCopyForSampling:…]`（`+Blit.m`，**无端口**） | **需先转它或补端口** |
| `bindSeparateSamplersAndArrayTextures:` | 17 | `materializeSampledSamplerForTexture:`（同文件）、fallback 链（已 C） | 同簇 |

**建议的两刀**：
1. 本清单里"可搬"的 5 个（含 `materializeSampledSamplerForTexture:` → 退役 `mglRendererMaterializeSampledSamplerPort`，28 → 27）；
2. 把 `traceSampledTextureReadback:`（`+Texture.m`）与 `freshGLSampledRenderTargetCopyForSampling:`（`+Blit.m`）
   转 C（后者内部还会碰 `[self currentRenderPassUsesTexture:]`、`[self uploadFullCPUTextureDataIntoTexture:…]`，
   **转之前先列它的外部被调者**），再搬剩下两个方法 → **`+BindingState.m` 整文件消失（6 → 5）**。

**累积开工规矩（§0.55 / §0.57 / §0.58 + 本节）**：
1. 每个 `id` 局部判保活；端口返回值与方法返回值可能差一个 +1。
2. `currentRenderEncoderOwner` 在使用点重新读取。
3. 转换前**全树搜该选择器的所有调用点**（`.m` + `.c`）；删方法后 `make test-all` 是最后一道网。
4. **新：入口返回值契约照抄 ARC**——ObjC 方法返回 +0/借用时，C 入口也必须返回借用指针（创建的 +1 交给缓存后立刻放掉），
   ObjC 调用点用 `(__bridge id)` 桥接（会让该文件语法略升，属正常）。
5. CTS 单次状态不是判决（抖动用例 `KHR-GL46…advanced-memory-order`），diff 异常先做两臂受控复跑。

133. **本轮（第 103 轮）尝试把 `materializeSampledSamplerForTexture:` + `applySampledCompatFallbackPlan:` +
     `bindSeparateSamplersAndArrayTextures:` 转 C 并退役 `mglRendererMaterializeSampledSamplerPort`（28 → 27）——
     **被 CTS 拦下，已整体回滚，未改动代码**（工作区仍在 `2643227`）**：
     ① 转换本身完成且能构建：新 TU `mgl_sampled_bind.{h,c}`（约 470 行）把三个方法搬成
     `mglBindingStateMaterializeSampledSampler` / `…ApplySampledCompatFallbackPlan` /
     `…BindSeparateSamplersAndArrayTextures`，端口包装与声明一并删除，`mgl_compute_bind.c` 两处调用点改直调；
     `make test-all` 通过（92/0/2/94），**A/B 全文逐行一致**（4,981/4,981、5,514/5,514，stderr 307/307）。
     ② **CTS 立刻报出真回归**：hotspot 簇非通过集合从 58 涨到 **65**，多出的全是**深度/深度模板纹理**用例
     （`KHR-GL46.internalformat.copy_tex_image.depth_component24/32`、
     `…internalformat.texture2d.depth_component_unsigned_*`、
     `packed_depth_stencil.stencil_texturing/verify_read_pixels.depth24_stencil8` 等 8 例），
     且状态由 `fail` 变 **`crash`**（进程 -10/-11）。
     ③ **定位过程与证据**：
     - 复现：`KHR-GL46.internalformat.copy_tex_image.depth_component24` 单跑 5 次 ——
       旧库 **5/5 通过**，新库 **2–3/5 段错误**（不是抖动用例，是确定性的内存错误）；
     - lldb 回溯：崩溃在 `objc_retain`，frame#1 = `-[MGLRenderer bindSampledTexturesForStage:…]` 第 520 行
       第 40 列，即 **ARC 对 `mglBindingStateMaterializeSampledSampler()` 返回值的保留**；
     - 插桩：该次调用的 `plan.action=2`（`USE_TEX_PARAMS`）、返回 `ptr->params.mtl_data`（非 NULL），
       其 isa 读出来是 `0x2800000000`（已被释放/复用的对象）；
     - **反证实验**：把该分支临时改成返回 NULL 后**仍然 2/5 崩溃** →
       **崩溃不是这个返回值造成的**，而在同一刀的其他改动里（`applySampledCompatFallbackPlan:` 的调用点重写，
       或 `bindSeparateSamplersAndArrayTextures:` 的整段搬移）。
     - 回滚后复验：`make test-all` **0**、该用例 **5/5 通过**。
     ④ **本轮两条新规矩（已并入 §0.60）**：
     - **"C 入口返回借用指针"这条规矩有盲区**：当 ObjC 调用点会把返回值赋给 `id __strong` 时，ARC 会对它 `retain`；
       若返回的对象恰好已经悬空，崩溃点会出现在**调用点**而不是 C 入口里，容易被误判成"转换错了"。
       排查办法就是本轮的**反证实验**（先让可疑返回值退化为 NULL，看崩溃是否消失）。
     - **A/B（`test_regression`）+ `make test-all` 双绿仍然挡不住这类回归**——本轮两套全绿，**只有 CTS 抓到**；
       与第 128/130 条同一结论，已第三次被验证。
     ⑤ **下一刀建议（把这一刀拆成三步，每步单独跑 CTS）**：
     (a) 先只搬 `applySampledCompatFallbackPlan:`（0 语法、无对象返回，风险最低）；
     (b) 再只搬 `materializeSampledSamplerForTexture:`（退役端口；搬完重点看 depth 用例是否仍 5/5 通过）；
     (c) 最后搬 `bindSeparateSamplersAndArrayTextures:`。
     同时值得单独追一条线索：`Texture::params.mtl_data` 指向的对象在本轮新构建里会**在采样器物化之前**被释放——
     需要查清 `params.mtl_data` 的 retain/release 链（`tex_param.c` 写入、`textures.c` 置 NULL 的路径、
     以及 mglSafeReleaseMetalObj 的调用点），这很可能是 `+BindingState.m` 采样簇能不能安全收尾的关键。

### 0.61 第 104 轮的合成结论（**开工前必读**）

**第 103/104 两轮把 `+BindingState.m` 采样簇的阻塞点收敛为一句话**：
`bindSampledTexturesForStage:` 循环里的 `Texture *ptr` **不能跨 `applySampledCompatFallbackPlan:` 使用**——
该调用（通过其回退链）会分配纹理，之后对 `ptr` 的任何解引用都可能命中失效内存（第 133/134 条，两轮各一次 CTS 拦截）。
旧 ObjC 构建只是**碰巧**把 `ptr->target` 的取值提前了，这个隐患一直在。

**因此下一刀的固定顺序**：
1. 先做**纯修复刀**：在 `applySampledCompatFallbackPlan:` 调用之后重新取 `ptr`（或整段改用调用前的字段快照），
   两处调用点都要；用**8 次深度用例探针** + `make test-all` + CTS 七簇 + A/B 验证，**单独提交**；
2. 再做**转换刀**：`applySampledCompatFallbackPlan:` → C（0 语法，先做）；探针 8 次通过后，
   再 `materializeSampledSamplerForTexture:` → C（退役 `mglRendererMaterializeSampledSamplerPort`，28 → 27）；
   最后 `bindSeparateSamplersAndArrayTextures:` → C，`+BindingState.m` 才可能整文件消失（6 → 5）。

**本轮新增的两条操作习惯**：
- **单用例重复探针**：先用被 CTS 抓到过的那个用例跑 5–8 次（约 2 分钟）判断是否仍有概率性崩溃，再决定是否跑整簇电池；
- **反证实验**：怀疑某个返回值/某次解引用时，把它临时退化为 NULL 或提前预取，观察崩溃是否消失（第 133 条 ③、第 134 条 ③ 各用了一次）。

### 0.60 第 103 轮交接快照（**新会话请先读本节 + §0.51 + §0.55 + §0.58**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 25,443 行 / 1,429 语法 / 3,013 词汇**；
壳 TU **1,988 行 / 279 语法**；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致
（以上全部是提交 `2643227` 的实测值——第 103 轮的改动**已整体回滚**，见第 133 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` 235/761/4,062 · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**开工前必读的三条**（累积 §0.55 / §0.57 / §0.58 / §0.59 + 本轮新增）：
1. 每个 `id` 局部判保活；端口返回值与方法返回值可能差一个 +1。
2. `currentRenderEncoderOwner` 在使用点重新读取（纹理上传会替换编码器）。
3. 转换前**全树搜该选择器的所有调用点**（`.m` + `.c`）；删方法后 `make test-all` 只是最后一道网。
4. C 入口的返回值契约照抄 ARC 的 +0/借用语义。
5. **新：当 C 入口的返回值会被 ObjC 调用点的 ARC `retain` 时，崩溃可能出现在调用点**——
   排查用它法：**先把可疑返回值临时退化为 NULL/NULL 指针，看崩溃是否消失**（第 133 条 ③ 的反证实验）。
6. **判定口径不变**：A/B + `make test-all` 双绿**不足以**判定无回归（已三次被 CTS 打脸），
   **CTS 七簇必须跑**；单次 CTS 状态也不是判决（抖动用例先做两臂受控复跑）。

**下一步（第 133 条 ⑤ 的三步拆分）**：
(a) `applySampledCompatFallbackPlan:`（0 语法）→ C，单独验证；
(b) `materializeSampledSamplerForTexture:`（4 语法）→ C 并退役 `mglRendererMaterializeSampledSamplerPort`（28 → 27），
    验证时**重点跑 depth 用例 5 次**；
(c) `bindSeparateSamplersAndArrayTextures:`（17 语法）→ C。
另需单独查清 `Texture::params.mtl_data` 的 retain/release 链（第 133 条 ⑤ 末段），它是采样簇收尾的前置问题。

134. **第 104 轮：按 §0.60 的三步拆分复做，定位到"`ptr` 跨分配失效"这个根因，但**转换再次被 CTS 拦下并回滚**
     （工作区仍在 `520691f`；本轮**只产出定位结论，未改动代码**）：
     ① **第一步就复现**：这一步只把 `applySampledCompatFallbackPlan:`（47 行 / 0 语法）搬成 C，其余（采样器物化、
     `bindSeparateSamplersAndArrayTextures:`）**保持 ObjC 不动**，深度用例
     `KHR-GL46.internalformat.copy_tex_image.depth_component24` 仍然 **2/5 崩溃**（旧库 5/5 通过）。
     说明**触发点不是采样器那一半**，而是这个"回退规划"调用本身。
     ② **插桩证据（决定性）**：在调用点前后各打印一次 `ptr` 与它的字段：
     - 调用前：`ptr=0x766af5e080 name=2 target=3553`（有效）；
     - **C 被调者收到的是同一个指针**（在被调者入口打印，值完全一致）→ 参数传递没问题；
     - 调用后：调用点的 `ptr` 变成 `0x766a000000 name=0xFFFFFF7F target=0xFFFFFFFF`（无效），
       随后的 `ptr->target` 取值就在 `ptr+0x18` 处 `EXC_BAD_ACCESS`。
     即：**回退链会在调用期间分配纹理（`mglSampledFallbackTextureForExpectedType`），
     而调用点持有的 `Texture *ptr` 不能跨这次分配使用**——旧 ObjC 构建之所以没崩，是编译器把
     `ptr ? ptr->target : 0u` 的取值**提前到调用之前**（同一个 `ptr` 在旧构建里没有被"调用后再读"的形态）。
     ③ **两个单点修复尝试（都只改调用点，不改转换）**：
     - 在回退调用**之前**把 `ptr->target` 预取到局部变量、后面用该局部（8 次复跑 **8/8 通过**）→
       证明"调用后再解引用 `ptr`"确实是崩溃来源；
     - 但把该预取与转换**一起**放开后，8 次复跑仍有 **4/8 崩溃** → 说明调用点后面**还有其它对同一 `ptr` 的解引用**
       （`mglMipDiagEnabled` 段的 `ptr->params`、以及被调 C 函数内部对 `ptr->name/width/height/params.mtl_data` 的读取），
       单点预取不够。
     ④ **结论（下一刀的正确做法）**：转换这一簇之前，必须先修掉"`ptr` 跨回退调用"这个**既有隐患**，
     二选一：**(a)** 回退调用之后**重新取一次** `ptr`（用循环开头那次 `mglTextureForSampledResource(...)` 的同一查法）；
     **(b)** 把该段所需字段在调用前一次性快照成局部量并全部改用局部量（`ptr` 本身不再在调用后解引用）。
     修完必须用**转换后的构建**跑 8 次深度用例探针 + CTS 七簇双证据，再继续搬
     `materializeSampledSamplerForTexture:` 与 `bindSeparateSamplersAndArrayTextures:`。
     ⑤ **新增验证手法（便宜且有效，已并入 §0.61）**：本轮用"**单用例 8 次探针**"代替整簇电池做早期判定——
     被 CTS 抓到过的那个用例在旧库上 5/5 通过、在新构建上概率性崩溃，**重复跑 5–8 次即可在 2 分钟量级判定回归存在**，
     比等 10 分钟电池快得多（但**最终判据仍是七簇电池**）。
     ⑥ 回滚复验：`make test-all` **0**（92/0/2/94）、深度用例 **5/5 通过**、工作区与 `520691f` 一致。

135. **P0-1 第七十七刀：`+Blit.m` 的两个"叶子"路径转 C（新 TU `mgl_blit_drivers.{h,c}`），采样簇改道不再挡进度**：
     ① **换赛道**：§0.61 已把 `+BindingState.m` 采样簇的阻塞点定性为"`Texture *ptr` 跨回退分配"的既有隐患，
     修它需要专门的修复刀（还涉及 arena/快照生命周期的判断）；为不空耗轮次，本轮改从**已解锁**的 `+Blit.m` 取整簇：
     - `-blitFramebufferResolveMsaaSource:…`（116 行 / 4 语法）→ **`mglBlitResolveMsaaSource`**；
     - `-copyImageSubDataCpuToCpu:…`（240 行 / 10 语法）→ **`mglBlitCopyImageSubDataCpuToCpu`**。
     两者在方法调查里都是**叶子**（不调用本文件任何其它 ObjC 方法），因此**零新增端口**；
     唯一调用者 `-mtlBlitFramebuffer:…` 与 `-mtlCopyImageSubData:…` 改成直调 C 入口（全树只此两处）。
     ② 机械替换按既定规则：`self`→renderer、`_device`/`_renderPassManager->state.currentCommandBufferOwner` →
     状态区（**命令缓冲属主在每次使用时重新读取**，第 130 条铁律）、`NSInteger/NSUInteger/BOOL/YES/NO/nil` →
     `int64_t/size_t/int/1/0/NULL`、`NSLog`→`fprintf`、`MAX()` 保留；
     `+Blit.m` 的 6 个文件静态（`mglBlitTextureInfo` / `…SynchronizeTexture` / `…CopyTexture` / `…EndBlitEncoder` /
     `…CopyBufferToTexture` / `…ReplaceTextureRegion`）写成 `mglBd*` twin（照抄实参展开顺序，`MGLOriginValue`/`MGLSizeValue`
     按 `.x/.y/.z`、`.width/.height/.depth` 展开）。
     ③ **`@try/@catch`**：`copyImageSubDataCpuToCpu` 里 `mglBlitReplaceTextureRegion` 的那个 `@try` 改用既有端口
     `mglPlatformShellGuardedCallCtx`（§0.14 路线 ③；回调签名是 `int (*)(void *renderer, void *ctx)`）。
     ④ **ARC 的另一个坑（本轮新记）**：调用点不能把 `id __strong` 局部的地址转成 `void **`
     （`cast of an indirect pointer to an Objective-C pointer to 'void **' is disallowed with ARC`）。
     处置：用一个 `void *` 临时量中转（`void *h = (__bridge void *)readtexid; … &h …; readtexid = (__bridge id)h;`），
     `BOOL *` 出参同理用 `int` 临时量中转。已并入 §0.62 的规矩表。
     ⑤ **oracle**：旧库 = 提交 `006fe32` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,416/5,393 与 6,134/6,199 →
     `slow` 435/412、620/685）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑥ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     两个新告警一栏核实：`+Blit.m` 的 `unused variable 'dstWidth'/'dstHeight'` **在改动前的旧库全量构建日志里同样存在**（2/2），非本刀引入。
     ⑦ **度量**：`+Blit.m` **4,062 → 3,714 行**、语法 **235 → 233**、词汇 **761 → 703**；
     全库行数 **25,443 → 25,095（−348）**、语法 **1,429 → 1,427（−2）**、词汇 **3,013 → 2,955（−58）**；
     文件数 **6 不变**；端口 **28 不变（0 退役 0 新增，T4 中性，如实记账）**；新增 C 面 553 + 60 行。
     ⑧ **下一刀（两条线，任选其一，都已解锁）**：
     (a) `+Blit.m` 继续：`blitFramebufferDirectColorCopyWithState:`（5 语法 / 87 行）需要先把文件内的
     `MGLBlitColorState`（含 `id` 字段）搬到 C 头（`id` → `void *`，`.m` 侧改 `(__bridge id)`）；
     `readTextureRegionViaBlit:`（9 语法 / 72 行，含一个 `@try`）同样可搬；
     (b) `+BindingState.m` 采样簇：先做第 134 条 ④ 的**纯修复刀**（回退调用后重新取 `ptr`），再转换。

### 0.62 第 105 轮交接快照（**新会话请先读本节 + §0.51 + §0.58 + §0.61**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 25,095 行 / 1,427 语法 / 2,955 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；
A/B 两臂逐行一致（见第 135 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **233/703/3,714** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（累积；开工前逐条过）**：
1. 每个 `id` 局部判保活（借用 `KeepAlive` / 自建 +1 `Adopt` / 多出口先改单出口）。
2. `currentRenderEncoderOwner` 与任何"会被 restore 的属主"在**使用点重新读取**。
3. 转换前**全树搜该选择器的所有调用点**（`.m` + `.c`）；删方法后 `make test-all` 只是最后一道网。
4. C 入口的返回值契约照抄 ARC（+0/借用）；**调用点不能把 `id __strong` 局部的地址转成 `void **`**——
   用 `void *` 临时量中转，`BOOL *` 出参也用 `int` 临时量（第 135 条 ④）。
5. **`Texture *` 不能跨"可能分配纹理"的调用使用**（第 133/134 条；`+BindingState.m` 采样簇的阻塞点）。
6. 判定口径：A/B + `make test-all` 双绿**不足以**判定无回归（已三次被 CTS 打脸）；单次 CTS 状态也不是判决；
   先跑**单用例 5–8 次探针**做早期判定，最终仍以七簇电池 diff 为准。

**下一步两条线（都已解锁）**：
1. **`+Blit.m` 续刀**：`MGLBlitColorState` 搬进 C 头（`id`→`void *`）后可搬
   `blitFramebufferDirectColorCopyWithState:`（5 语法）、`blitFramebufferIntegerColorWithState:`（6）、
   `blitFramebufferScaledColorWithState:`（9）；`readTextureRegionViaBlit:`（9 语法，含 `@try` → 守卫调用端口）
   与 `copyImageSubData3DFallback:`（14 语法）也可单独搬。该文件还剩 233 语法。
2. **`+BindingState.m` 采样簇**：先做纯修复刀（回退调用后**重新取 `ptr`**，或整段改用调用前的字段快照），
   探针 8 次 + 七簇通过后，再按 compat → sampler（退役端口 28 → 27）→ separate samplers 的顺序转换，
   最后该文件整文件消失（6 → 5）。

136. **P0-1 第七十八刀：`MGLBlitColorState` 搬进 C 头 + 两条颜色路径转 C（`mgl_blit_color_state.h` / `mgl_blit_color_paths.c`）**：
     ① **先把共享状态搬到 C**：`+Blit.m` 文件内的 `MGLBlitColorState`（30 行，含 `id readtexid; id drawtexid;`）
     搬到新头 **`mgl_blit_color_state.h`**：`id` → `void *`、`NSInteger`→`int64_t`、`NSUInteger`→`size_t`、`BOOL`→`int`。
     这是本刀真正的价值——**后续几条颜色路径（整数色 6 语法、缩放色 9 语法）从此可以直接取该状态**。
     代价是 `.m` 侧 6 处对两个 `id` 字段的使用要显式桥接（`(__bridge void *)readtexid` / `(__bridge id)st->readtexid`）。
     ② 转 C 的两个方法（都在 `mgl_blit_color_paths.c`）：
     - `-resolveIntegerMultisampleTexture:toTexture:srcOrigin:dstOrigin:size:reason:`（57 行 / 6 语法）
       → `mglBlitResolveIntegerMultisampleTexture`（用到既有端口 `mglRendererEnsureWritableCommandBufferPort`）；
     - `-blitFramebufferDirectColorCopyWithState:`（87 行 / 5 语法）→ `mglBlitDirectColorWithState`。
     `+Blit.m` 的计算/拷贝静态助手写成 `mglBc*` twin；`_batching`/命令缓冲属主走状态区（属主每次重新读取）；
     `mglMarkTextureLevelRenderTargetWritten` 是 ObjC 头里的**宏**，C 侧改调 `mglMarkTextureLevelRenderTargetWrittenImpl`
     并显式传 `__FILE__`/`__LINE__` 标签；`MGLMSAAIntegerResolveParams` 在 C TU 内**重复定义**（ObjC 私有头已有同名结构，
     两处都包含会重定义，故只放在 `.c` 里）。
     ③ **度量（如实记账：本刀语法净 +1）**：`+Blit.m` **3,714 → 3,543 行**、语法 **233 → 234（+1）**、词汇 **703 → 650（−53）**；
     全库行数 **25,095 → 24,924（−171）**、语法 **1,427 → 1,428（+1）**、词汇 **2,955 → 2,902（−53）**；
     文件数 6、端口 **28（0 退役 0 新增，T4 中性）**。**语法微增的原因**：两个方法自身语法 11 处，
     而状态结构的 6 个桥接点 + 2 处赋值桥接把其中大部分又加了回来；**收益在行数与词汇，以及"状态已 C 化"这步铺垫**。
     ④ **oracle**：旧库 = 提交 `f922e81` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,365/5,321 与 6,112/6,110 →
     `slow` 384/340、598/596）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑤ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     新告警核查：`+Blit.m` 的 `unused variable 'dstMinY'/'srcW'` 等在**改动前旧库全量构建日志里同样存在**（2/2），非本刀引入。
     ⑥ **下一刀（顺水推舟）**：同一个 `MGLBlitColorState` 已经就位，可继续搬
     `blitFramebufferIntegerColorWithState:`（6 语法 / 124 行，其依赖 `resolveIntegerMultisampleTexture` 本刀已 C 化）与
     `blitFramebufferScaledColorWithState:`（9 语法 / 204 行，需要 `mglDrawableTexture` 的 C 入口与
     `NSMakeRange`→`mglBlitCreateTextureView` 的区间参数化）；两条都做完后 `+Blit.m` 的语法应降到 ~215。

### 0.63 第 106 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.62**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,924 行 / 1,428 语法 / 2,902 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 136 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **234/650/3,543** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表照旧（§0.62 六条）+ 本轮两条**：
7. ObjC 头里的**宏**（如 `mglMarkTextureLevelRenderTargetWritten`）在 C 侧要用它的 `*Impl` 入口并显式传
   `__FILE__`/`__LINE__` 标签；同名的结构若两个头都定义（`MGLMSAAIntegerResolveParams`），C TU 内**只重复一次**放在 `.c`。
8. 把文件内结构搬进 C 头时，先数清 `.m` 侧对 `id` 字段的使用点：**桥接成本可能抵消被搬方法的语法收益**（本轮 +1）。
   搬结构的价值在于**解锁后续方法**，衡量时要把"下一刀能省多少"算进来。

**下一步（三选一，都已解锁）**：
1. **`+Blit.m` 续刀（推荐）**：`blitFramebufferIntegerColorWithState:`（6 语法）+ `blitFramebufferScaledColorWithState:`（9 语法），
   状态已 C 化、`resolveIntegerMultisampleTexture` 已 C 化；缩放的 `NSMakeRange` 需换成 `mglBlitCreateTextureView` 的显式区间参数。
2. `+Blit.m` 其它：`resolvedReadbackTextureForMultisampleTexture:` / `depthFloatTextureForDepthStencilReadback:` /
   `blitFramebufferDepthStencil:`（33 语法，最大块）。
3. `+BindingState.m` 采样簇：先做**纯修复刀**（回退调用后重新取 `ptr` 或整段改用字段快照），
   探针 8 次 + 七簇通过后再按 compat → sampler（退役端口 28 → 27）→ separate samplers 转换。

137. **P0-1 第七十九刀：整数色 blit 路径转 C（`+Blit.m` 续刀，语法 −10）**：
     ① `-blitFramebufferIntegerColorWithState:`（124 行 / 6 语法）→ **`mglBlitIntegerColorWithState`**，
     落在上一刀的 `mgl_blit_color_paths.c` 里——**上一刀把 `MGLBlitColorState` 搬进 C 头的铺垫在这里兑现**：
     本刀不需要任何新 twin、不需要新端口，直接复用 `mglBc*` 与已 C 化的
     `mglBlitResolveIntegerMultisampleTexture`（第 136 条）。
     ② 机械替换照旧：`NSInteger/NSUInteger/BOOL/YES/NO` → `int64_t/size_t/int/1/0`、`NSLog`→`fprintf`、
     命令缓冲属主走状态区并在使用点读取、`mglMarkTextureLevelRenderTargetWritten` 宏 → `…Impl(…, __LINE__)`。
     唯一新增 include 是 `mgl_state_compat.h`（`mglNearlyEqual`）。
     ③ **度量**：`+Blit.m` **3,543 → 3,424 行**、语法 **234 → 224（−10）**、词汇 **650 → 612（−38）**；
     全库行数 **24,924 → 24,805（−119）**、ObJC 语法 **1,428 → 1,418（−10）**、词汇 **2,902 → 2,864（−38）**；
     文件数 6、端口 **28（0 退役 0 新增，T4 中性）**。
     （上一刀语法 +1 的账在本刀补回：+1 → −10，两刀合计 −9。）
     ④ **oracle**：旧库 = 提交 `6c3de70` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,275/5,282 与 5,816/5,816 →
     `slow` 294/301、302/302，本刀 flushy 两臂 slow 计数完全相同）、**stderr MGL 307/307 多重集一致**、
     两臂 **92/0/2、91/1/2**。
     ⑤ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     新告警核查：`+Blit.m:1253-1256` 的 `unused variable 'dstMinY'/'dstMaxY'/'srcW'/'srcH'`
     **在改动前的旧库全量构建日志里同样存在**，非本刀引入。
     ⑥ **下一刀**：`blitFramebufferScaledColorWithState:`（9 语法 / 204 行）——它需要
     `-mglDrawableTexture`（壳类方法，目前**没有 C 入口**）与 `NSMakeRange` 参数化，故要么补一个端口（端口 28 → 29，
     按 §0.04 记为 T4 −1），要么把 `mglDrawableTexture` 一起搬；做完后 `+Blit.m` 语法应降到 ~215。
     另一条已解锁的路仍是 `+BindingState.m` 采样簇的**纯修复刀**（§0.61/§0.62）。

### 0.64 第 107 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.63**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,805 行 / 1,418 语法 / 2,864 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 137 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **224/612/3,424** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62 八条仍然有效）＋ 本轮补充**：
9. **搬"共享状态结构"这一刀要按两刀合算**：第 136 刀语法 +1、第 137 刀 −10（合计 −9）——
   单看铺垫刀会误判为负收益；交接时要把"下一刀能省多少"一并说明。

**下一步（三条线，按收益/风险排序）**：
1. **`+Blit.m` 收尾**：`blitFramebufferScaledColorWithState:`（9 语法 / 204 行）需要 `-mglDrawableTexture` 的 C 入口
   （补端口则 28 → 29，按 §0.04 记 T4 −1；或把该壳方法一并处理），以及 `NSMakeRange` → 显式区间参数；
   还可顺手搬 `blitFramebufferDepthStencil:`（**33 语法，本文件最大块**，依赖 `bindMTLTexture` 端口与
   `endRenderEncoding`/`ensureWritableCommandBuffer` 等既有端口）。
2. **`+BindingState.m` 采样簇**：先做**纯修复刀**（回退调用后重新取 `ptr`，或整段改用调用前字段快照），
   探针 8 次 + 七簇通过后，再按 compat → sampler（退役端口 28 → 27）→ separate samplers 转换，
   最后该文件整文件消失（6 → 5）。
3. `+Texture.m`（271 语法 / 1,052 词汇，词汇最多，需成批 `NSLog`→`fprintf` 与 `BOOL`→`int`）与
   `MGLRenderer.m`（168 语法）；壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

138. **P0-1 第八十刀：`+Blit.m` 最大块 `blitFramebufferDepthStencil:` 转 C（语法 −33，零新增端口）**：
     ① 方法 `-blitFramebufferDepthStencil:srcX0:srcY0:srcX1:srcY1:dstX0:dstY0:dstX1:dstY1:mask:filter:`
     （约 350 行 / **33 语法**，本文件最大块）→ **`mglBlitDepthStencil`**，落在 `mgl_blit_color_paths.c`
     （该 TU 现在同时承载 color / integer / depth-stencil 三条路径，注释已更新）。
     ② **关键取巧：把"只有 ObjC 才拿得到的依赖"换成状态区已有的字段**。方法里 3 处
     `mglBlitCreateRenderEncoder(_renderPassManager, &state)` 看似需要 render-pass-manager 指针
     （第 134 条里 `readTextureRegionViaBlit:` 就是被这个卡住的），但读其实现发现**它只用
     `manager->state->currentCommandBufferOwner`**。于是 C twin 改为
     `mglBcCreateRenderEncoder(areas, state)` 直接取 `areas.command->currentCommandBufferOwner`（**在使用点读取**），
     **完全不需要新端口**。这条已并入 §0.65 的规矩表。
     ③ 机械替换：`ctx`→`glm_ctx`/`areas.ctx`、`self`→renderer、`[self bindMTLTexture:]`→`mglRendererBindMTLTexture`、
     `[self framebufferAttachmentTexture:]`→`mglRendererAttachmentTextureFor(ctx, att)`（均既有 C 入口）、
     `[self endRenderEncoding]`→既有端口、`[self ensureWritableCommandBuffer:]`→既有端口、
     `NSInteger/NSUInteger/BOOL/YES/NO`→C 类型、`NSLog`→`fprintf`、`mglMarkTextureLevelRenderTargetWritten` 宏→`…Impl(…, __LINE__)`；
     另在 C TU 内重复 ObjC 私有头里的 `MGLViewportValue` / `MGLScissorRectValue` 两个 POD 结构（并补 `<math.h>` 与 `MIN`/`MAX`）。
     ④ **度量**：`+Blit.m` **3,424 → 3,072 行**、语法 **224 → 191（−33）**、词汇 **612 → 567（−45）**；
     全库行数 **24,805 → 24,453（−352）**、ObjC 语法 **1,418 → 1,385（−33）**、词汇 **2,864 → 2,819（−45）**；
     文件数 6、端口 **28（0 退役 0 新增，T4 中性）**；`mgl_blit_color_paths.c` 无新告警。
     ⑤ **oracle**：旧库 = 提交 `cb7b347` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,415/5,400 与 6,138/6,178 →
     `slow` 434/419、624/664）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑥ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑦ **下一刀**：`+Blit.m` 剩 **191 语法**，最大两块是 `-mtlBlitFramebuffer:…`（12 语法，含 9 个 `[self …]` 调度）
     与 `-mtlCopyImageSubData:…`（24 语法，含 10 个 `[self …]` 调度）——两者都是**调度器**，
     需要先把它们调用的那些叶子路径搬完（剩下的叶子：`resolvedReadbackTextureForMultisampleTexture:`、
     `depthFloatTextureForDepthStencilReadback:`、`copyImageSubDataFormatConversion:`、
     `copyImageSubData3DFallback:`、`copyImageSubDataPostBlitReadback:`、`readTextureRegionViaBlit:`、
     `blitFramebufferScaledColorWithState:`）。其中 `readTextureRegionViaBlit:` 也可照本刀 ② 的办法解开
     （若它的 pass-manager 用法同样只是取 owner）。

### 0.65 第 108 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.64**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,453 行 / 1,385 语法 / 2,819 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 138 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **191/567/3,072** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62 九条仍然有效）＋ 本轮第十条（很省事的一条）**：
10. **遇到"只有 ObjC 才拿得到"的依赖时，先读它的实现**：本轮 `mglBlitCreateRenderEncoder(_renderPassManager, …)`
    实际只用 `manager->state->currentCommandBufferOwner`，于是 C twin 直接取 `areas.command->currentCommandBufferOwner`
    （在使用点读取），**省掉了一个端口**。同类情况还有 `readTextureRegionViaBlit:` 的
    `mglPassManagerWaitForLastSubmittedCommandBuffer(_renderPassManager, …)`（需确认它是否也只要 owner/state）。

**下一步（按收益排序）**：
1. **`+Blit.m` 剩余叶子路径**（把调度器 `-mtlBlitFramebuffer:`（12 语法）与 `-mtlCopyImageSubData:`（24 语法）留到最后）：
   `readTextureRegionViaBlit:`（9 语法，照规矩 10 试解）、`copyImageSubDataPostBlitReadback:`（22 语法）、
   `copyImageSubData3DFallback:`（14）、`copyImageSubDataFormatConversion:`（14）、
   `blitFramebufferScaledColorWithState:`（9，需 `-mglDrawableTexture` 的 C 入口）、
   `blitFramebufferResolveMsaaSource` 之外的 `resolvedReadbackTextureForMultisampleTexture:`（3）等。
2. **`+BindingState.m` 采样簇**：先做**纯修复刀**（回退调用后重新取 `ptr` 或整段改用调用前字段快照），
   探针 8 次 + 七簇通过后，再按 compat → sampler（退役端口 28 → 27）→ separate samplers 转换，
   最后该文件整文件消失（**6 → 5**，本周期最大的一次文件数下降）。
3. `+Texture.m`（271 语法 / 1,052 词汇，词汇最多）与 `MGLRenderer.m`（168 语法）；
   壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

139. **P0-1 第八十一刀：纹理回读路径转 C（`readTextureRegionViaBlit:`；**零新增端口**，并踩到一个"guarded-call finally 必跑"的坑）**：
     ① `-readTextureRegionViaBlit:region:slice:level:bytes:bytesPerRow:bytesPerImage:reason:`
     （72 行 / 9 语法）→ **`mglBlitReadTextureRegion`**（落在 `mgl_blit_drivers.c`），两处调用点
     （`copyImageSubDataFormatConversion:` 与 `copyImageSubData3DFallback:`）改直调。
     ② **规矩 10 再次奏效（省掉一个端口）**：方法里 `mglPassManagerWaitForLastSubmittedCommandBuffer(_renderPassManager, &state)`
     看名字需要 render-pass-manager，读实现发现**它只是转发**
     `mglRenderWaitCommandBufferOwnerLastSubmitted(manager->state->currentCommandBufferOwner, state)`，
     于是 C twin 直接用 `mglBdCommandBufferOwner(&areas)`（在使用点读取）——**端口 28 不变**。
     ③ ⚠️ **新坑（本轮真 bug，`make test-all` 抓到）**：`@try { copy; endEncoder; } @catch { @try { endEncoder; } @catch {} ; log; }`
     搬到 `mglPlatformShellGuardedCallCtx(…, finally_fn)` 时，我把"catch 里的 endEncoder"写成了 **finally 回调**——
     但 **finally 是无条件执行的**，于是正常路径也再 end 一次，AGX 在
     `-[AGXG16GFamilyBlitContext endEncoding]` 直接崩（`agx_3d_texture_workarounds` 段错误）。
     **正确做法**：body 里 end 并置 `ended` 标志；失败分支再包一层 guarded-call 做"只补一次 end"的清理。
     已写入 §0.66 规矩表第 11 条。
     ④ **度量**：`+Blit.m` **3,072 → 2,993 行**、语法 **191 → 184（−7）**、词汇 **567 → 546（−21）**；
     全库行数 **24,453 → 24,374（−79）**、ObjC 语法 **1,385 → 1,378（−7）**、词汇 **2,819 → 2,798（−21）**；
     文件数 6、端口 **28（0 退役 0 新增，T4 中性）**。
     ⑤ **oracle**：旧库 = 提交 `ae0c402` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,364/5,356 与 6,133/6,149 →
     `slow` 383/375、619/635）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ⑥ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⚠️ 第一次七簇电池是在**未修复**的构建上启动的，已作废重跑（修复后七簇才全绿）——**改了代码就要重跑电池，不能沿用旧电池结果**。
     ⑦ 遗留告警（记在案，非新引入语义）：`mgl_blit_drivers.c` 两处 `-Wpointer-bool-conversion`
     （`src_tex->faces && dst_tex->faces` 恒真）——**原 ObjC 代码就是这个写法**，C 化后编译器才报出来；
     保持与原实现一致，不改语义。

### 0.66 第 109 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.65**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,374 行 / 1,378 语法 / 2,798 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 139 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **184/546/2,993** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62/§0.65 十条仍然有效）＋ 本轮第十一条**：
11. **`mglPlatformShellGuardedCallCtx` 的 `finally_fn` 是无条件执行的**，不能拿它当 `@catch` 用：
    body 里正常收尾（并置标志），失败分支再包一层 guarded-call 只补一次清理（第 139 条 ③）。
12. **改了代码就要重跑七簇电池**：不能沿用"上一次启动但对应旧二进制"的电池结果（第 139 条 ⑥）。

**下一步（按收益排序）**：
1. **`+Blit.m` 剩余叶子**：`copyImageSubDataPostBlitReadback:`（22 语法）、`copyImageSubData3DFallback:`（14）、
   `copyImageSubDataFormatConversion:`（14）、`blitFramebufferScaledColorWithState:`（9，需 `-mglDrawableTexture` 的 C 入口
   或照规矩 10 找替代）、`resolvedReadbackTextureForMultisampleTexture:`（3）、
   `depthFloatTextureForDepthStencilReadback:`（7）；两个调度器（`-mtlBlitFramebuffer:` 12、`-mtlCopyImageSubData:` 24）留到最后。
   注意 `mglBlitScaledPipelineForPixelFormat` / `mglBlitScaledSamplerForFilter` / `mglBlitClearRectDepthState`
   已是 C 入口（`mgl_blit_pipelines.h`）✓。
2. **`+BindingState.m` 采样簇**：先做**纯修复刀**（回退调用后重新取 `ptr` 或整段改用调用前字段快照），
   探针 8 次 + 七簇通过后再按 compat → sampler（退役端口 28 → 27）→ separate samplers 转换，
   最后该文件整文件消失（**6 → 5**）。
3. `+Texture.m`（271 语法 / 1,052 词汇）与 `MGLRenderer.m`（168 语法）；壳的 `MGLPipelineCache` 归档路径。

140. **第 110 轮：修掉第 135 刀埋下的"创建对象 +1 泄漏"（所有权契约刀，无文件/语法变化）**：
     ① **发现**：复盘第 135 刀的 `mglBlitResolveMsaaSource` 时看出一个真 bug——该 C 入口在 MSAA 解析路径上
     `mglBdCreateTexture` 拿到 **+1** 的解析纹理并把它交给调用点的 `void *` 句柄，而 ObjC 调用点用
     `readtexid = (__bridge id)readtexidHandle;` 只是**保留**（retain），**没有接管那个 +1** →
     每次 `glBlitFramebuffer` 的 MSAA 解析都会**泄漏一个纹理引用**（功能正常，但内存只增不减）。
     ② **修法（对称契约，避免"有时 +1 有时借用"）**：C 入口在返回前对**借用**的原纹理补一次 `CFRetain`，
     于是**所有返回路径都带 +1**；调用点改用 `__bridge_transfer` 接管（ARC 会同时释放旧值）。
     这样解析路径与非解析路径的所有权语义一致，既无泄漏也不会过度释放。
     ③ **oracle**：旧库 = 提交 `495a36e` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,335/5,339 与 6,112/6,119 →
     `slow` 354/358、598/605）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ④ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑤ **度量（本刀刻意零变化，如实记账）**：文件 6、行数 **24,375**（+1 注释）、语法 **1,378**、词汇 **2,798**、
     端口 **28**；收益是**修掉一个每帧级别的引用泄漏**与一条可复用契约。
     ⑥ **新增规矩（§0.67 第 13 条）**：**C 入口若可能返回"新创建的对象"，要么把 +1 明确交给调用方接管
     （调用点用 `__bridge_transfer`），要么统一补 retain 使所有路径都是 +1**；绝不能出现
     "`__bridge id` + 有时带 +1"，那必然泄漏。凡是 `*_ptr`/`void *` 出参返回 Metal 对象的刀，都要先问一句：
     **这个句柄带不带 +1、调用点是 `__bridge` 还是 `__bridge_transfer`**。

### 0.67 第 110 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.66**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,375 行 / 1,378 语法 / 2,798 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 140 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 271/1,052/6,256 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **184/546/2,993** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62/§0.65/§0.66 十二条仍然有效）＋ 本轮第十三条（所有权契约）**：
13. **C 入口返回 Metal 对象时的 +1 契约必须明确且对称**：要么统一 +1（借用路径补 `CFRetain`）＋ 调用点
    `__bridge_transfer`，要么统一借用；**绝不允许"`__bridge id` + 有时带 +1"**（第 135 刀的泄漏就是这么来的）。
    转换任何"创建并返回对象"的方法前，先查调用点是 `__bridge` 还是 `__bridge_transfer`。

**下一步（按收益排序）**：
1. **`+Blit.m` 剩余叶子**（两个调度器 `-mtlBlitFramebuffer:` 12 语法 / `-mtlCopyImageSubData:` 24 语法留到最后）：
   - `resolvedReadbackTextureForMultisampleTexture:`（3 语法 / 74 行）与
     `depthFloatTextureForDepthStencilReadback:`（7 语法 / 86 行）——**两者都"创建并返回纹理"**，
     按规矩 13 把入口做成统一 +1 契约（借用/新建都补 retain）＋ 调用点 `__bridge_transfer`；
   - `copyImageSubData3DFallback:`（14 语法 / 410 行）——**零新增端口**（自建 staging 缓冲自己释放），
     依赖的 `readTextureRegionViaBlit` 已 C 化（第 139 刀）；
   - `copyImageSubDataFormatConversion:`（14）、`copyImageSubDataPostBlitReadback:`（22）需要
     `synchronizeRenderPassForTextureReadback` 的 C 入口（补端口则记 T4 −1）；
   - `blitFramebufferScaledColorWithState:`（9）需要 `-mglDrawableTexture` 的 C 入口。
2. **`+BindingState.m` 采样簇**：先做**纯修复刀**（回退调用后重新取 `ptr` 或整段改用调用前字段快照），
   探针 8 次 + 七簇通过后再按 compat → sampler（退役端口 28 → 27）→ separate samplers 转换，
   最后该文件整文件消失（**6 → 5**）。
3. `+Texture.m`（271 语法 / 1,052 词汇）与 `MGLRenderer.m`（168 语法）；壳的 `MGLPipelineCache` 归档路径。

141. **P0-1 第八十二刀：readPixels 的两个"创建并返回纹理"叶子转 C（按规矩 13 做成统一 +1 契约）**：
     ① `-resolvedReadbackTextureForMultisampleTexture:sourceLevel:sourceSlice:sourceDepthPlane:reason:`
     （74 行 / 3 语法）→ **`mglBlitResolvedReadbackTexture`**；
     `-depthFloatTextureForDepthStencilReadback:reason:`（86 行 / 7 语法）→ **`mglBlitDepthFloatTextureForReadback`**。
     两者都落在 `mgl_blit_color_paths.c`（渲染pass/编码器 twin 已在该 TU）。
     ② **本刀是规矩 13 的第一次实战**：两个方法都会**创建并返回**一张纹理（MSAA 解析纹理 / 深度浮点提取纹理），
     也可能**原样返回借用**的输入纹理。C 入口统一为 **+1 契约**：新建的本来就 +1；借用路径返回前
     `CFRetain`；错误路径 `mglSafeReleaseMetalObj` 后返回 NULL。调用点（`+Texture.m` 的 4 处，全部是
     `sourceTexture = …` 强局部赋值）改用 **`(__bridge_transfer id)`** 接管，ARC 同时释放旧值——与原 ARC 语义一致且无泄漏。
     ③ **oracle**：旧库 = 提交 `55dfd97` 的独立 worktree 构建（两库不同；先清 `.o/.d`）；
     `ab_full.py`：**default 4,981/4,981、flushy 5,514/5,514 逐行一致**（未过滤 5,386/5,360 与 6,164/6,163 →
     `slow` 405/379、650/649）、**stderr MGL 307/307 多重集一致**、两臂 **92/0/2、91/1/2**。
     ④ **CTS 七簇**：非通过集合 **diff 全空**（58 / 1 / 0 / 59 / 13 / 39 / 4）；`make test-all` **0**（92/0/2/94）。
     ⑤ **度量（语法净 −2，如实拆账）**：`+Blit.m` **2,993 → 2,836 行**、语法 **184 → 174（−10）**、词汇 **546 → 530**；
     `+Texture.m` 6,256 → 6,248 行、语法 **271 → 279（+8，4 处调用点的 `__bridge_transfer`/`__bridge void *` 桥接）**、
     词汇持平；全库行数 **24,375 → 24,209（−166）**、ObjC 语法 **1,378 → 1,376（−2）**、词汇 **2,798 → 2,782（−16）**；
     文件数 6、端口 **28（0 退役 0 新增，T4 中性）**；新 TU 无告警。
     ⑥ **下一刀**：`+Blit.m` 剩 **174 语法**——`copyImageSubData3DFallback:`（14 语法 / 410 行，**零新增端口**，
     其依赖 `readTextureRegionViaBlit` 已 C 化）、`copyImageSubDataFormatConversion:`（14）、
     `copyImageSubDataPostBlitReadback:`（22，后两者需 `synchronizeRenderPassForTextureReadback` 的 C 入口）、
     `blitFramebufferScaledColorWithState:`（9，需 `-mglDrawableTexture` 的 C 入口）；
     两个调度器（`-mtlBlitFramebuffer:` 12、`-mtlCopyImageSubData:` 24）留到最后。

### 0.68 第 111 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.67**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,209 行 / 1,376 语法 / 2,782 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（第 141 条）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` **279/1,052/6,248** · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **174/530/2,836** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62/§0.65/§0.66/§0.67 十三条仍然有效）＋ 本轮观察第十四条**：
14. **搬"创建并返回对象"的方法时，调用点一侧的语法会小幅上升**（`__bridge_transfer` + `__bridge void *` 桥接）。
    第 141 刀就是 `+Blit.m` −10、`+Texture.m` +8；**要按两文件合算**，别只看被搬文件（同 §0.64 规矩 9 的账法）。

**下一步（按收益排序）**：
1. **`+BindingState.m` 采样簇（本周期最大的一次文件数下降）**：先做**纯修复刀**（回退调用后重新取 `ptr`
   或整段改用调用前字段快照），探针 8 次 + 七簇通过后，再按 compat → sampler（退役端口 28 → 27）→
   separate samplers 转换，最后该文件整文件消失（**6 → 5**）。
2. **`+Blit.m` 续刀**：`copyImageSubData3DFallback:`（14 语法 / 410 行，零新增端口）；
   `copyImageSubDataFormatConversion:`（14）与 `copyImageSubDataPostBlitReadback:`（22）需
   `synchronizeRenderPassForTextureReadback` 的 C 入口（补端口记 T4 −1）；
   `blitFramebufferScaledColorWithState:`（9）需 `-mglDrawableTexture` 的 C 入口。
3. `+Texture.m`（279 语法 / 1,052 词汇，词汇最多）与 `MGLRenderer.m`（168 语法）；
   壳的 `MGLPipelineCache` 归档路径（Foundation→POSIX + 专属 oracle）。

142. **第 112 轮：第四次尝试采样绑定簇（compat + sampler）——仍然被 CTS 拦下并回滚；但本轮把嫌疑范围收窄到
     `recoverFragmentSampledDepthTexture:` 的出参写入**（工作区仍在 `004c9e4`，**未改动代码**）：
     ① 本轮按 §0.61 方案先做"纯修复刀"：在 `bindSampledTexturesForStage:` 里、**调用
     `applySampledRenderTargetCopyPlan:` / `applySampledCompatFallbackPlan:` 之前**把
     `Texture *ptr` 复制成局部快照并把 `ptr` 指向快照（这一段之后只读 `ptr`，所以副本精确），
     再叠加 compat + sampler 两个方法的 C 化（新 TU `mgl_sampled_bind.{h,c}`，端口退役 28 → 27 的版本）。
     ② **结果：仍然 5/8 段错误**（同一用例 `KHR-GL46.internalformat.copy_tex_image.depth_component24`；
     不叠加转换时该用例 8/8 通过）。**说明快照修复并不能解除阻塞**——快照把调用"之后"的读取固定住了，
     但崩溃依旧，意味着**传给这些计划的 `ptr` 在调用"之前"就已经不是有效记录了**。
     ③ **新的嫌疑对象（下一刀从这里查）**：`recoverFragmentSampledDepthTexture:(Texture **)ptrPtr` ——
     它是**唯一**能写穿 `&ptr` 的方法，且只在 **depth / depth-stencil** 用例上走这条路径，
     与本轮崩溃用例全是 depth 用例完美吻合。它内部会把 `*ptrPtr` 指向"配对颜色纹理"或"历史候选纹理"
     （`MGL_STATE(ctx)->recent_sampled_2d_textures[...]`、`mglFindFramebufferColorTexturePairedWithDepth`），
     而下面这些来源**都可能是已经失效的快照/历史槽位**：
     - 历史上第 133/134 轮测得"调用点 `ptr` 在 compat 调用前后发生变化"（`0x766af5e080` → `0x766a000000`）
       也与"`ptr` 早在 recover 阶段就被写成了失效值"一致（当时打印的 pre 值是寄存器里的旧值，
       内存槽里已经是坏值——正好解释了"指针在调用前后不一致"这个反常现象）。
     ④ **下一刀的正确查法**：先给 `recoverFragmentSampledDepthTexture:` 的**每个 `*ptrPtr` 写入点**加断言/日志
     （或在 C 侧用 `mglRendererObjectPointerLikelyValid` 之类的现成判据过滤），确认它在 depth 用例上写出的
     到底是哪一类纹理、是否已失效；**先修这个方法（或它的候选来源），再回头转换采样簇**。
     ⑤ 回滚复验：`make test-all` **0**（92/0/2/94）、该 depth 用例 **3/3 通过**、工作区与 `004c9e4` 一致、
     度量不变（6 文件 / 24,209 行 / 1,376 语法 / 2,782 词汇 / 28 端口）。
     ⑥ **代价与教训**：这是采样簇第四次被拦（第 103/104/112 轮两次转换尝试 + 第 112 轮修复尝试）。
     已把"先证明 `ptr` 在进入该段之前就是有效的"写进 §0.69 的开工检查清单——**不要再用"在调用后补救"的思路**。

144. **第 114 轮（P0-1 第八十四刀）：`+Blit.m` 的两个 copyImageSubData 叶子转 C——只新增 1 个端口**：
     ① **切口**：`-copyImageSubDataFormatConversion:…`（224 行 / 9 语法）与
     `-copyImageSubData3DFallback:…`（400 行 / 9 语法）整块搬进既有 TU `mgl_blit_drivers.c`，
     新入口 `mglBlitCopyImageSubDataFormatConversion` / `mglBlitCopyImageSubData3DFallback`
     （签名与既有 `mglBlitCopyImageSubDataCpuToCpu` 对齐）；两个方法删除，调用点（dispatcher
     `-(void)mtlCopyImageSubData:…` 内）改成 C 调用。**本刀未新建 TU**——同族入口放同一文件。
     ② **端口只 +1**：唯一需要新桥的是 `synchronizeRenderPassForTextureReadback:reason:`
     （定义在 `+RenderPass.m`），新增 `mglRendererSynchronizeRenderPassForTextureReadbackPort`；
     `endRenderEncoding` / `flushCommandBuffer` 都已有端口，`@try/@catch` 用既有
     `mglPlatformShellGuardedCallCtx`。**端口面 28 → 29**（T4 如实记 +1）。
     ③ **零 areas 改动的关键发现**：`_capability` 不是 ivar，而是宏
     `#define _capability _core.capability`（`MGLRenderer_Private.h:312`），所以 C 侧直接
     `areas.core->capability` 就够（`MGLRendererCoreState` 已含 `MGLCapability capability`）——
     **又一次靠"先读依赖实现"（§0.62 规矩 11）省掉了端口/结构改动**。
     ④ **guarded call 的忠实用法**：两个方法共 4 个 `@try/@catch`，全部按 §0.14 路线 ③ 转成
     `mglPlatformShellGuardedCallCtx`；其中 3D 读源的那个 `@try` 体内原本有 4 处
     `return YES`，改成 ctx 上的 `early_exit` 标志由外层统一收口（**体内部一律不 `free`**
     staging，外层四条路径各释放一次，避免双重释放）。NSException 打印成 `caught exception`
     （A/B 未触发这些路径，stderr 多重集不受影响）。
     ⑤ **度量（两文件合算）**：`+Blit.m` 语法 **159 → 145**、词汇 **530 → 340**、行 **2,687 → 2,044**；
     壳 TU（新端口）**1,988 → 1,999 行 / 279 → 282 语法**；全库 **24,060 → 23,428 行（−632）**、
     语法 **1,361 → 1,350（−11）**、词汇 **2,737 → 2,592（−145）**；文件数 **6**、空 TU **0**、端口 **28 → 29**。
     ⑥ **验证（四件套全绿）**：`make -j8` 无错（0 error；新代码里 4 处 `-Wpointer-bool-conversion`
     是 `faces` 数组名的忠实复刻，与原 ObjC 表达式同源）；`make test-all` **0**（92/0/2/94）；
     **CTS 七簇 diff 全空**（58/1/0/59/13/39/4）；A/B（新库 vs `62fce97`）两臂 default **92/0/2**、
     flushy **91/1/2**，确定性行 **4981/4981** 与 **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑦ **代价与教训（第十七条）**：删掉 18 语法，净收益只有 **−11**——差额被"调用点改 C 调用时新增的
     `(__bridge void *)` 桥接"和"壳里新端口实现自带的 3 处语法"吃掉。**刀前先算三处账**：
     被删方法、调用点桥接、壳端口实现；只看被搬文件会高估收益。

143. **第 113 轮（P0-1 第八十三刀）：`+Blit.m` 的 `mtlCopyTexSubImageViaTextureBlit:` 叶子转 C——零新增端口**：
     ① **切口**：`-(BOOL)mtlCopyTexSubImageViaTextureBlit:tex:destTexture:slice:level:xoffset:yoffset:x:y:width:height:`
     （**140 行 / 17 语法**）整块搬进既有 TU `mgl_blit_drivers.c`，新入口
     `mglBlitCopyTexSubImageViaTextureBlit(void *renderer, void *ctx, Texture *tex, void *destTexture, …)`；
     原方法删除，唯一调用点（约 1421 行）改成 C 调用。
     ② **零新增端口**：入口内部全部复用既有端口与 twin——`mglRendererAttachmentTextureFor`、
     `mglRendererBindMTLTexture`、`mglRendererEndRenderEncodingPort`、`mglRendererEnsureWritableCommandBufferPort`、
     `mglBd*` 系列 twin，以及 ObjC 私有头 `static inline` 的 C 孪生（新增 twin
     `mglBdMarkTextureLevelMetalFilled`，对应 `mglMarkTextureLevelMetalFilled`）；新 include
     `mgl_texture_bind.h` / `mgl_texture_readback_clear.h` / `mgl_blit_sampled_copy.h`，并按既有做法把
     ObjC 私有头的声明在 C TU 内重述为文件局部 `extern`。**端口面 28 → 28**。
     ③ **度量（净减，无需分摊记账）**：`+Blit.m` 语法 **174 → 159**、词汇 **530 → 485**、行 **2,836 → 2,687**；
     全库 **24,209 → 24,060 行（−149）**、语法 **1,376 → 1,361（−15）**、词汇 **2,782 → 2,737（−45）**；
     文件数 **6**、空 TU **0**、壳 TU **1,988 行 / 279 语法**、端口 **28** 均不变。
     ④ **验证（四件套全绿）**：`make -j8` 无错（新 TU 与 `+Blit.m` 无新增警告）；`make test-all` **0**
     （`PASS: 92  FAIL: 0  SKIP: 2 / 94`）；**CTS 七簇非通过集合 diff 全空**
     （hotspot 58 / tess 1 / gs 0 / refq 59 / piq 13 / compute 39 / pp 4，与 `base_*.txt` 逐行一致）；
     A/B oracle（新库 vs `MGL-ab-old` @ `96d0aa0`）两臂 default **92/0/2**、flushy **91/1/2**，
     确定性行 **4981/4981** 与 **5514/5514** 逐行一致、stderr MGL 多重集 **307/307**。
     ⑤ **代价与教训**：本刀是**纯收益刀**（−15 语法 / −149 行 / 0 新端口），没有新教训；
     但本轮踩到一次**工具纪律**坑并已修正——**批测结果的读取必须用 `summary.tsv`（`NR>1 && $3!="pass"`）
     过滤，绝不能 `ls $dir/*.txt | head -1`**：目录里的 `command.txt` 只有一行 glcts 命令行，
     照它做 diff 会得到"59 行全变"的假回归。已把这条写进 §0.70 规矩表第十六条。

### 0.69 第 112 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.68**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,209 行 / 1,376 语法 / 2,782 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；CTS 七簇 **diff 全空**；A/B 两臂逐行一致（`004c9e4` 的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **174/530/2,836** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。

**规矩表（§0.62/§0.65/§0.66/§0.67/§0.68 十四条仍然有效）＋ 本轮第十五条**：
15. **"在调用之后补救"解决不了本簇的问题**（第 112 轮实测）：快照修复 + 转换仍然 5/8 崩溃 ⇒
    嫌疑在**进入该段之前**就产生的失效 `ptr`。开工前必须先证明"`ptr` 在进入采样循环时是有效记录"，
    即查 `recoverFragmentSampledDepthTexture:` 的每个 `*ptrPtr` 写入点（**只在 depth/depth-stencil 用例上走**）。

**下一步（把采样簇的嫌疑对象查清，再谈转换）**：
1. **第一优先：审 `recoverFragmentSampledDepthTexture:` 的 `*ptrPtr` 写入点**（第 142 条 ③④）：
   给每个写入点加日志/有效性判据（现成有 `mglRendererObjectPointerLikelyValid`、
   `mglRendererTextureLooksLikeSampledColor2D` 等），在 `KHR-GL46.internalformat.copy_tex_image.depth_component24`
   上跑 5–8 次，确认写出的纹理来源（配对着色纹理 / 历史候选）以及是否失效；**先修它**。
2. 采样簇其余方法（compat → sampler → separate samplers）在 1 修好之前**不要再试**。
3. 并行可做的安全刀（都已解锁，零新增端口）：
   `+Blit.m` 的 `copyImageSubData3DFallback:`（14 语法 / 410 行）；
   `+Texture.m` 里词汇最多的一批（1,052 词汇，主要是 `NSLog`→`fprintf`、`BOOL`→`int` 的成批替换）；
   `+RenderPass.m` / `MGLRenderer.m` 的叶子方法。

### 0.70 第 113 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 24,060 行 / 1,361 语法 / 2,737 词汇**；
壳 TU **1,988 行 / 279 语法**（上限 2,400）；端口面 **28 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十三刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **159/485/2,687** · 壳 `MGLPlatformRendererShell.m` 279/287/1,988 · `+BindingState.m` **91/85/1,678**。
（`+Blit.m` 已从 174/530/2,836 降到 159/485/2,687，**再切 2–3 刀即可降到 100 以下**。）

**规矩表（§0.62/§0.65/§0.66/§0.67/§0.68/§0.69 十五条仍然有效）＋ 本轮第十六条**：
16. **批测结果的读取方式要固定**（第 113 轮实测）：每个簇目录里只有 `summary.tsv` / `summary.json` 是结果，
    `command.txt`（也就是 `ls *.txt | head -1` 抓到的那个）只是 glcts 命令行。
    正确取法：`LC_ALL=C awk -F'\t' 'NR>1 && $3!="pass" {print $2"\t"$3}' $dir/summary.tsv | LC_ALL=C sort`
    再与 `/tmp/base_<cluster>.txt` 比。用错文件会得到"整簇全变"的**假回归**，白跑一轮排查。

**采样绑定簇（`+BindingState.m`）——本轮只读调研结论（嫌疑点从"某处"缩到"四个具体写入点"）**：
- `Texture *ptr` 在本文件里只有 **5 处**被赋值：475 `ptr = mglTextureForSampledResource(...)`（初始化）、
  **952 `ptr = pairedColor;`**（ENTER_INSAMPLER 分支）、**1026 `ptr = recoverTexture;`**（history 路径）、
  **1130 `ptr = recoverTexture;`**（RT 路径），以及 1152 `*ptrPtr = ptr;`（唯一写穿出参处）。
  952 的来源是 `mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &pairedFboName)`；
  1026 的来源是探针候选 `cand`；1130 的来源是 `pairedColor`。
- **关键怀疑（比 §0.69 更具体）**：这些赋值在 ObjC 里都是 `Texture *__strong` 局部赋值（**ARC 会 retain 新值**），
  转成 C 之后**这个 retain 消失**。若其中某个纹理只被"当时那个 strong 局部"保活，
  转换后 `ptr` 就会在其后的 plans 执行期间失效——这与"快照修复（只 memcpy、不 retain）无效"和
  "在 plans 之前就已经坏掉"两个实测现象都吻合。
- **下一刀的正确做法：先做"调用边界"判定实验，别再直接做整段转换**：
  写一个**只做转发**的 C 入口（`mglSampledCompatFallbackPlan(void *r, Texture *ptr, void *texture, …)` /
  `mglSampledSamplerMaterialize(void *r, Texture *ptr, …)`，函数体就是
  `[(__bridge MGLRenderer *)r applySampledCompatFallbackPlan:ptr …]`），把两个调用点改成 C 调用。
  **它不减少任何语法**（不算正式刀），但精确重现"ObjC 指针穿过 C 边界（参数不 retain、返回 `void *`）"这一语义差：
  - 若该版本**崩** ⇒ 结论是"参数需要 C 边界保活"，正式刀的 C 入口在入口处加
    `CFRetain` / `__bridge_retained` 即可，转换就能正常做；
  - 若该版本**不崩** ⇒ 结论是"差异在被转方法体内部"（`id` 局部的 strong 语义 / 返回值 +0↔+1），
    再往 `sampler = (__bridge id)(glSampler->mtl_data)`（1373）与
    `texture = (__bridge id)mglSampledFallbackTextureForExpectedType(...)`（1323）这两条借用路径上查。
  探针：`KHR-GL46.internalformat.copy_tex_image.depth_component24` 连跑 8 次（不转换时 8/8 通过）。

**下一步（优先级顺序）**：
1. 采样簇按上面的判定实验走（**先判定再转换**；第 142 条 ②"别再试整段转换"仍然有效）。
2. 并行安全刀（零新增端口、已解锁）：`+Blit.m` 的 `copyImageSubData3DFallback:`（14 语法 / 410 行）、
   两个 dispatcher（`-mtlBlitFramebuffer:` 12 语法 / `-mtlCopyImageSubData:` 24 语法）；
   `+Texture.m` 的词汇批量替换（1,052 词汇，主要是 `NSLog`→`fprintf`、`BOOL`→`int`）；
   `+RenderPass.m` / `MGLRenderer.m` 的叶子方法。
3. 需要新端口的刀排在后面：`copyImageSubDataFormatConversion:` / `copyImageSubDataPostBlitReadback:`
   （要 `synchronizeRenderPassForTextureReadback` 的 C 入口，+1 端口）、
   `blitFramebufferScaledColorWithState:`（要 `-mglDrawableTexture` 的 C 入口）。

145. **第 115 轮（P0-1 第八十五刀）：`+Blit.m` 的 copyImageSubData 收尾——后置回读叶子 + 整个 dispatcher 一起转 C**：
     ① **切口**：`-copyImageSubDataPostBlitReadback:dstTexture:…`（256 行 / 8 语法）与
     `-(void)mtlCopyImageSubData:srcTexture:…`（dispatcher，212 行 / 24 语法）一起搬进
     `mgl_blit_drivers.c`，新入口 `mglBlitCopyImageSubDataPostBlitReadback` / `mglBlitCopyImageSubData`；
     两个方法删除，唯一调用点（`+Texture.m` 的 `mglCopyImageSubDataFenceEntry`，原
     `[renderer mtlCopyImageSubData:…]`）改成 C 调用，`MGLRenderer+Texture_Private.h` 里的方法声明删除，
     `+Texture.m` 补 `#include "mgl_blit_drivers.h"`。
     ② **两个"白捡"的桥**（都是先查再动手的结果）：
     - dispatcher 第一句 `ctx = glm_ctx;` 不需要新端口——**已有 `mglPlatformShellSetContext`**
       （注释就写着"the Objective-C compute entry points did `ctx = glm_ctx;` before dispatching"）；
     - `bindMTLTexture:` 也没有 port——它早已退役，C 侧直接调 `mglRendererBindMTLTexture`
       （`mgl_texture_bind.h`，本刀前 8 个方法已在用）。
     唯一新增的桥是 `endRenderPassIfFramebufferChangedForNonDraw:` →
     `mglRendererEndRenderPassIfFramebufferChangedForNonDrawPort`（签名 `(void*, uint64_t)`）。
     **端口面 29 → 30**。
     ③ **guarded call 的三个 @try**：PostBlitReadback 的两个（都在 slice 循环里，catch 里
     `readbackOK=false; break;`）复用第 144 刀就建好的 `MglBdGetBytesCtx` / `mglBdGetBytesGuarded`；
     dispatcher 那个大 `@try`（包住整个 slice 循环 + `endBlitEncoder`）用新的
     `mglBdCopyImageDispatchGuarded`，其 `@catch` 里**无条件**再结束一次 encoder（原代码就是无条件，
     且那次也带自己的 `@try/@catch`）——用 `mglBdEndBlitEncoderGuarded` 包一层，没有引入规则 9 的
     `finally` 误用。`RETURN_ON_FAILURE` 在 C 侧照用（它的 `__FUNCTION__`/`__LINE__` 会变，但 A/B 日志里
     `^failure ` 行为 0，不受影响——本刀已实测）。
     ④ **度量（两文件合算）**：`+Blit.m` 语法 **145 → 109**、词汇 **340 → 239**、行 **2,044 → 1,570**；
     壳 TU **1,999 → 2,008 行 / 282 → 284 语法**；`+Texture.m` **不变**（`[renderer …]` 换成
     `(__bridge void *)renderer`，一进一出）；全库 **23,428 → 22,963 行（−465）**、
     语法 **1,350 → 1,316（−34）**、词汇 **2,592 → 2,491（−101）**；文件数 **6**、空 TU **0**、端口 **29 → 30**。
     ⑤ **验证（四件套全绿）**：`make -j8` **0 error**、无新增警告（新代码里 3 类既有警告均与第 144 刀同源：
     `faces` 数组名的 pointer-bool-conversion 是忠实复刻）；`make test-all` **0**（92/0/2/94）；
     **CTS 七簇 diff 全空**（58/1/0/59/13/39/4）；A/B（新库 vs `ad7f99b`）两臂 default **92/0/2**、
     flushy **91/1/2**，确定性行 **4981/4981** 与 **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑥ **教训**：本刀 −34 是"叶子 + 它的唯一调用者一起搬"的收益——**把 dispatcher 和它调用的叶子放在同一刀**
     能让调用点的 `(__bridge void *)` 桥接一次性消化掉，避免第 144 刀那种"被调用点慢慢啃"的−11。

### 0.71 第 114 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.70**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 23,428 行 / 1,350 语法 / 2,592 词汇**；
壳 TU **1,999 行 / 282 语法**（上限 2,400）；端口面 **29 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十四刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
`+Blit.m` **145/340/2,044** · 壳 `MGLPlatformRendererShell.m` 282/287/1,999 · `+BindingState.m` **91/85/1,678**。

**`+Blit.m` 的剩余地图（本刀后重算，语法 / 行数）**——下一次开这个文件直接照此表选点：
| 段 | 语法 | 行数 | 转 C 需要的桥 |
|---|---|---|---|
| 28 个 `static` helper（34–369） | ~40 | ~336 | 底层全是 `mglRender*` C 函数，twin 好写 |
| `-freshGLSampledRenderTargetCopyForSampling:` | 11 | 161 | `uploadFullCPUTextureDataIntoTexture`（有端口）、`restoreRenderEncoderAfterTextureUpload`（有端口）、`currentRenderPassUsesTexture`（**需 +1**）；**被 `+BindingState.m` 调用（采样簇）** |
| `-resolveBlitFramebufferAttachments:` | 12 | 144 | `mglDrawableTexture`（**需 +1**）、`mglEnsureLayerDrawableSizeAtLeastWidth`（**需 +1**）、`mglNextDrawable`（**需 +1**）、`bindMTLTexture`（有） |
| `-blitFramebufferScaledColorWithState:` | 11 | 202 | `mglDrawableTexture`（同上，共用） |
| `-(void)mtlBlitFramebuffer:…`（dispatcher） | 18 | 418 | `endRenderEncoding`/`ensureWritableCommandBuffer`/`flushDrawBuffer` **都已有端口** → 但要先转上面两个被调方法 |
| `-(void)mtlCopyTexSubImage:…`（dispatcher） | 10 | 179 | `bindMTLTexture`（有）、`mtlReadDrawable`（**需 +1**）、`copyTextureUploadWithDedicatedCommandBuffer`（**需 +1**）、`dataWithLength`（→`malloc`） |
| `-copyImageSubDataPostBlitReadback:` | 8 | 256 | `flushCommandBuffer`（有）、`synchronizeRenderPassForTextureReadback`（**本刀已加**）→ **零新增端口** |
| `-(void)mtlCopyImageSubData:…`（dispatcher） | 24 | 226 | 上述全部 + `endRenderPassIfFramebufferChangedForNonDraw`（**需 +1**） |
| `mglRendererBlitFramebuffer`（已是 C 函数） | 0 | 25 | 随文件一起挪走即可 |

**规矩表（§0.62/§0.65/§0.66/§0.67/§0.68/§0.69/§0.70 十六条仍然有效）＋ 本轮第十七条**：
17. **"删掉多少语法"≠"净减多少语法"，开工前算三处账**（第 114 轮实测）：被删方法（18）＋ 调用点改成
    C 调用时新增的 `(__bridge void *)` 桥接 ＋ 壳里新端口实现自带的语法（本刀 +3），三者合计才是净变化
    （本刀 18 − 4 − 3 = **11**）。只看被搬文件会高估收益；**新增端口一定会在壳里记 +2~3 语法**。

**下一步（优先级顺序）**：
1. **`+Blit.m` 最划算的一刀**：`-copyImageSubDataPostBlitReadback:`（8 语法 / 256 行，**零新增端口**——
   它要的两个桥现在都有了）＋ dispatcher `-(void)mtlCopyImageSubData:…`（24 语法 / 226 行，需
   `endRenderPassIfFramebufferChangedForNonDraw` **+1 端口**）。合计 32 语法，是本文件剩下最大的一块。
2. 然后 `mtlBlitFramebuffer:` 簇（dispatcher 18 ＋ 两个被调方法 12+11，需 `mglDrawableTexture` /
   `mglEnsureLayerDrawableSizeAtLeastWidth` / `mglNextDrawable` 三个端口）。
3. 最后 helper 簇 ＋ `freshGLSampledRenderTargetCopyForSampling:`（注意：它是 `+BindingState.m`
   采样簇的依赖，先转它不影响采样簇安全，但**改它的契约会牵动采样簇**，动之前先看第 142/143 条）。
4. 采样绑定簇仍按 §0.70 的"调用边界判定实验"走，未判定前不要做整段转换。

146. **第 116 轮（P0-1 第八十六刀）：`+Blit.m` 的 blitFramebuffer 附着解析叶子转 C——新增 3 个 drawable 端口**：
     ① **切口**：`-resolveBlitFramebufferAttachments:srcX0:srcY0:srcX1:srcY1:dstX0:dstY0:
     dstX1:dstY1:outState:outReadAttachment:`（144 行 / 12 语法）搬进 `mgl_blit_drivers.c`，新入口
     `mglBlitResolveFramebufferAttachments`；方法删除，dispatcher `-(void)mtlBlitFramebuffer:…` 的调用点改成 C 调用。
     `blitFramebufferScaledColorWithState:` 那一簇**故意留到下一刀**——它要一整套 render-encoder twin
     （`mglBlitCreateRenderEncoder` / `mglBlitSetRender*` / `mglBlitDrawPrimitives` / `mglBlitEndRenderEncoder`，
     还要处理 `NSMakeRange`），与本刀的最小切口不同源。
     ② **端口 +3（30 → 33）**：`_drawable` **是 property 宏**（`#define _drawable self.drawable`，与
     `_capability` 那条 `_core.capability` 不同，**没有 core 字段可借**），所以本刀实测的桥是
     `mglRendererNextDrawablePort`（`-mglNextDrawable` 自己就会 `self.drawable = …`，端口只调它）、
     `mglRendererDrawableTexturePort`（`self.drawable.texture`，**无 drawable 时返回 NULL**）、
     `mglRendererEnsureLayerDrawableSizeAtLeastWidthPort`（BOOL）。
     ③ **两处语义合并（都等价，但要在注释里写清）**：
     - 原代码 `if (!_drawable || ![self mglDrawableTexture])` 是**两次**属性读；合并成"`drawableTexture` 为 NULL"
       一次判断即可（无 drawable ⇒ `nil.texture` ⇒ NULL），且随后那句 `readtexid = [self mglDrawableTexture]`
       直接复用同一次查询。
     - 原代码 `MAX(0, MAX(dstX0, dstX1))` 是 **int** 运算再转 `NSUInteger`；C 侧显式写成两个 int 三元
       （`max_dst_x > 0 ? max_dst_x : 0`）再转 `size_t`，**不能用 `mglBdMaxSize`**（后者是 size_t 版，
       负数语义会被拓宽成巨大值）。
     ④ **度量（两文件合算）**：`+Blit.m` 语法 **109 → 97**、词汇 **239 → 209**、行 **1,570 → 1,419**；
     壳 TU **2,008 → 2,037 行 / 284 → 291 语法**（3 个端口）；全库 **22,963 → 22,841 行（−122）**、
     语法 **1,316 → 1,311（−5）**、词汇 **2,491 → 2,463（−28）**；文件数 **6**、空 TU **0**、端口 **30 → 33**。
     ⑤ **验证（四件套全绿）**：`make -j8` **0 error**、警告集合与上一刀**逐条一致**（29 条，无新增）；
     `make test-all` **0**（92/0/2/94）；**CTS 七簇 diff 全空**（58/1/0/59/13/39/4）；
     A/B（新库 vs `fc542a8`）两臂 default **92/0/2**、flushy **91/1/2**，确定性行 **4981/4981** 与
     **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑥ **教训（第十九条）**：**一个端口值 2~3 点语法**（本刀 3 个端口在壳里 +7，被搬文件 −12，净只有 −5）。
     凡是"被搬代码要写 renderer 的 ivar"的地方，先用 `grep '#define _xxx'` 看它是宏还是真 ivar：
     宏指向 `_core.*`/areas 的算零端口（第 144 刀），指向 property（`self.drawable`）的**必须**按
     "每个 property 一个端口"预算——**别等写完才发现净收益被壳吃掉**。

### 0.72 第 115 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.71**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 22,963 行 / 1,316 语法 / 2,491 词汇**；
壳 TU **2,008 行 / 284 语法**（上限 2,400）；端口面 **30 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十五刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
壳 `MGLPlatformRendererShell.m` 284/287/2,008 · `+Blit.m` **109/239/1,570** · `+BindingState.m` **91/85/1,678**。

**`+Blit.m` 只剩 109 语法，两刀可以清零**（照此表选点，别再逐个啃）：
| 段 | 语法 | 行数 | 转 C 需要的桥 |
|---|---|---|---|
| `-(void)mtlBlitFramebuffer:…`（dispatcher） | 18 | 418 | `endRenderEncoding`/`ensureWritableCommandBuffer`/`flushDrawBuffer` **都已有端口**；但它调的下面两个方法得先转 |
| `-resolveBlitFramebufferAttachments:` | 12 | 144 | `mglDrawableTexture`（**+1**）、`mglEnsureLayerDrawableSizeAtLeastWidth`（**+1**）、`mglNextDrawable`（**+1**）、`bindMTLTexture`（无端口，直调） |
| `-blitFramebufferScaledColorWithState:` | 11 | 202 | `mglDrawableTexture`（同上，共用） |
| 28 个 `static` helper（34–369） | ~40 | ~336 | 底层全是 `mglRender*`，多数 twin 已在 `mgl_blit_drivers.c`；缺的按第 144 刀的办法补 |
| `-freshGLSampledRenderTargetCopyForSampling:` | 11 | 161 | `uploadFullCPUTextureDataIntoTexture`/`restoreRenderEncoderAfterTextureUpload` 有端口；`currentRenderPassUsesTexture`（**+1**）；**被 `+BindingState.m` 的采样簇调用** |
| `-(void)mtlCopyTexSubImage:…`（dispatcher） | 10 | 179 | `bindMTLTexture` 直调；`mtlReadDrawable`（**+1**）、`copyTextureUploadWithDedicatedCommandBuffer`（**+1**）、`dataWithLength`→`malloc` |
| 文件头 13 个 `#import` | 13 | — | 随文件删除一起消失 |

**规矩表（§0.62/§0.65/§0.66/§0.67/§0.68/§0.69/§0.70/§0.71 十七条仍然有效）＋ 本轮第十八条**：
18. **把 dispatcher 和它唯一的被调叶子放进同一刀**（第 145 轮实测）：第 144 刀只搬叶子，收益被调用点的
    `(__bridge void *)` 桥接吃掉（18 → 净 11）；第 145 刀把 dispatcher 一起搬，同一处桥接一次性消化
    （叶子 8 + dispatcher 24 = 32 → 净 34，因为 dispatcher 里那 12 处桥接调用直接变成了 C 直调）。
    **开工前先查"这个方法被谁调、调用点是不是也在这刀里"**；调用点在别的文件时，用
    `mglPlatformShellSetContext` 这类**既有**入口先查一遍再谈新增端口（第 145 轮靠这一步省了 2 个端口）。

**下一步（优先级顺序）**：
1. **清掉 `+Blit.m` 的倒数第二刀**：`mtlBlitFramebuffer:` + `resolveBlitFramebufferAttachments:` +
   `blitFramebufferScaledColorWithState:`（41 语法 / 764 行，+3 端口）。
2. **清掉 `+Blit.m` 的最后一刀**：28 个 helper + `freshGLSampledRenderTargetCopyForSampling:` +
   `mtlCopyTexSubImage:` + 文件头（约 68 语法，+3 端口）→ **文件数 6 → 5**。
   注意 `freshGLSampledRenderTargetCopyForSampling:` 是采样簇的依赖，动它的契约前先看第 142/143/144 条。
3. `+BindingState.m`（91 语法）：仍按 §0.70 的"调用边界判定实验"走；stage buffer 绑定簇（不涉 depth-recover）
   可以先做。
4. `MGLRenderer.m`（168）· `+Texture.m`（279）· `+RenderPass.m`（385）· 壳（284）。

147. **第 117 轮（P0-1 第八十七刀）：`+Blit.m` 的 scaled color blit 叶子转 C——零新增端口**：
     ① **切口**：`-blitFramebufferScaledColorWithState:`（202 行 / 11 语法）搬进 `mgl_blit_drivers.c`，
     新入口 `mglBlitFramebufferScaledColorWithState(void *renderer, MGLBlitColorState *st)`；方法删除，
     dispatcher `-(void)mtlBlitFramebuffer:…` 的调用点改成 C 调用。**端口面 33 → 33（零新增）**——
     `mglDrawableTexture` 用第 146 刀刚加的端口，其余全是既有 C 函数。
     ② **顺带补齐了 12 个 render-encoder twin**（下次转 `mtlBlitFramebuffer:` / `mtlCopyTexSubImage:` 直接用）：
     `mglBdCreateTextureView`（把 `NSMakeRange(level,1)`/`NSMakeRange(slice,1)` 拆成两个
     `(offset, count)` 参数对，C 里没有 `NSRange`）、`mglBdDefaultRenderPassState`、`mglBdRenderPassAttachment`、
     `mglBdCreateRenderEncoder`（**直接收 command buffer owner**，因为 .m 传 `_renderPassManager` 只是为了读
     `->state->currentCommandBufferOwner`，C 侧用 `mglBdCommandBufferOwner(&areas)`）、
     `mglBdSetRenderPipeline/Bytes/Texture/Sampler`、`mglBdSetRenderViewport`/`SetRenderScissor`（收标量，
     不再需要 `MGLViewportValue`/`MGLScissorRectValue` 这两个只在 ObjC 私有头里定义的类型）、
     `mglBdDrawPrimitives`、`mglBdEndRenderEncoder`。
     ③ **踩到规矩 7 的又一例**：`mglMarkTextureLevelRenderTargetWritten(tex, level)` 是 **ObjC 头里的宏**，
     C 侧编译报 `call to undeclared function`；改用它的 `mglMarkTextureLevelRenderTargetWrittenImpl(tex, level,
     __func__, __LINE__)`（与本仓 `mgl_texture_readback_clear.c` 的既有写法一致）即可。**开工前先 grep 宏名**。
     ④ **度量（单文件记账，壳不变）**：`+Blit.m` 语法 **97 → 86**、词汇 **209 → 154**、行 **1,419 → 1,214**；
     壳 TU **2,037 行 / 291 语法不变**；全库 **22,841 → 22,636 行（−205）**、语法 **1,311 → 1,300（−11）**、
     词汇 **2,463 → 2,408（−55）**；文件数 **6**、空 TU **0**、端口 **33**。
     ⑤ **验证（四件套全绿）**：`make -j8` **0 error**；**警告如实记账**：`unused function` 从 17 条涨到 29 条
     （新增 13 条全是本刀搬走的 render-encoder helper——它们的唯一使用者就是被删的方法，
     这些 helper 会随最后一刀整文件删除，属预期），新增 2 条 `unused variable 'dst_min_y'/'dst_max_y'`
     （忠实复刻原 ObjC 里同样未使用的 `dstMinY`/`dstMaxY`）；`make test-all` **0**（92/0/2/94）；
     **CTS 七簇 diff 全空**（58/1/0/59/13/39/4）；A/B（新库 vs `ba0547c`）两臂 default **92/0/2**、
     flushy **91/1/2**，确定性行 **4981/4981** 与 **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑥ **教训**：**零端口刀 = 纯收益刀**（本刀 −11，第 146 轮 −5 被 3 个端口吃掉 7 点）。
     选点的顺序应当是：先转"只依赖既有端口/既有 C 函数"的叶子，把需要新端口的那些**留到最后一起算账**。

### 0.73 第 116 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.72**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 22,841 行 / 1,311 语法 / 2,463 词汇**；
壳 TU **2,037 行 / 291 语法**（上限 2,400）；端口面 **33 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十六刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
壳 `MGLPlatformRendererShell.m` 291/289/2,037 · `+Blit.m` **97/209/1,419** · `+BindingState.m` **91/85/1,678**。

**`+Blit.m` 只剩 97 语法**（按 §0.72 的表选点，本刀已完成其中 `resolve…Attachments:` 一行）：
- `-(void)mtlBlitFramebuffer:…`（dispatcher，18 语法）＋ 它只调剩下的三个方法：转它必须**同时**转
  `blitFramebufferScaledColorWithState:`（11）与 `mtlCopyTexSubImage:`（10），否则要补端口；
- `blitFramebufferScaledColorWithState:`（11 语法 / 202 行）—— 需要的新 twin：`mglBdCreateTextureView`
  （替代 `NSMakeRange` 版）、`mglBdDefaultRenderPassState`、`mglBdRenderPassAttachment`、
  `mglBdCreateRenderEncoder`（用 `areas.command->currentCommandBufferOwner` 替 `_renderPassManager`）、
  `mglBdSetRenderPipeline/Bytes/Texture/Sampler/Viewport/Scissor`、`mglBdDrawPrimitives`、
  `mglBdEndRenderEncoder`；`mglDrawableTexture` 已有端口（本刀加的）；
- 28 个 `static` helper（约 40 语法）＋ `freshGLSampledRenderTargetCopyForSampling:`（11）＋
  `mtlCopyTexSubImage:`（10）＋ 文件头 13 个 `#import` → 清掉后**文件数 6 → 5**。

**规矩表（§0.62/§0.65/§0.66/§0.67/§0.68/§0.69/§0.70/§0.71/§0.72 十八条仍然有效）＋ 本轮第十九条**：
19. **开工前先把"要写的 ivar 是宏还是 property"查清**（第 146 轮实测）：`grep '#define _<ivar>' include/MGLRenderer_Private.h`。
    指向 `_core.*` / areas 的 ⇒ 零端口（第 144 刀）；指向 `self.<property>` 的 ⇒ **每个 property 一个端口**，
    而且**端口在壳里要记 +2~3 点语法**（第 146 轮 3 个端口 = 壳 +7，把被搬文件的 −12 吃成净 −5）。
    预算公式（第 145 轮第十八条）因此要再加一项：**净收益 = 被删方法 − 调用点桥接 − 壳里新端口实现**。

**下一步（优先级顺序）**：
1. **`+Blit.m` 倒数第二刀**：`mtlBlitFramebuffer:` + `blitFramebufferScaledColorWithState:` + `mtlCopyTexSubImage:`
   （39 语法 / 799 行；`mtlDrawableTexture` 与 render-encoder twin 都已在手，`mtlReadDrawable` /
   `copyTextureUploadWithDedicatedCommandBuffer` 需 +2 端口）。
2. **`+Blit.m` 最后一刀**：28 个 helper + `freshGLSampledRenderTargetCopyForSampling:` + 文件头 → **文件数 6 → 5**。
   注意 `freshGLSampledRenderTargetCopyForSampling:` 是采样簇依赖，动它契约前先看第 142/143/144 条。
3. `+BindingState.m`（91 语法）：sample 簇仍按 §0.70 的"调用边界判定实验"走；stage buffer 绑定簇可先做。
4. `MGLRenderer.m`（168）· `+Texture.m`（279）· `+RenderPass.m`（385）· 壳（291）。

148. **第 118 轮（P0-1 第八十八刀）：`mtlBlitFramebuffer:` dispatcher 转 C——零新增端口，`+Blit.m` 只剩 68 语法**：
     ① **切口**：`-(void)mtlBlitFramebuffer:srcX0:srcY0:srcX1:srcY1:dstX0:dstY0:dstX1:dstY1:mask:filter:`
     （**416 行 / 18 语法**，本文件最大的 dispatcher）搬进 `mgl_blit_drivers.c`，新入口
     `mglBlitFramebufferDispatch`；方法删除，同文件那个已经是 C 的 `mglRendererBlitFramebuffer`
     直接调用它（不再发消息）。**端口面 33 → 33（零新增）**——它调的
     `mglBlitResolveFramebufferAttachments` / `mglBlitFramebufferScaledColorWithState` /
     `mglBlitDepthStencil` / `mglBlitIntegerColorWithState` / `mglBlitDirectColorWithState` /
     `mglBlitResolveMsaaSource` **全部已在 C 里**，`flushDrawBuffer` / `endRenderEncoding` /
     `ensureWritableCommandBuffer` 都有端口。这正是 §0.72 那张表"先转叶子再转 dispatcher"的兑现。
     ② **本刀新增的三个文件局部 extern**（都是 ObjC 私有头里的纯 C 符号，按既有模式重述）：
     `mglEnvFlagEnabled`（**注意返回类型是 `BOOL`，即 macOS 上的 `signed char`——声明成 `int` 会读到
     w0 高位的垃圾**，所以显式写 `extern signed char`）、`isColorAttachment`、`getFBOAttachment`；
     新 include 三个头：`<math.h>`（`fabs`）、`mgl_blit_clip.h`（`MGLBlitAxis`/`mglClipBlitAxis`）、
     `mgl_blit_pipelines.h`（`kMglSwapPresentDiagnostics`）。
     ③ **所有权（本刀最容易出错的地方）**：`readtexid` 在 ObjC 里是
     `readtexid = (__bridge_transfer id)readtexidHandle;`（ARC 在作用域结束时释放那个 +1），
     而 MSAA 入口按 §0.62 规矩 13 返回**统一 +1**（借用路径也会 `CFRetain`）。C 侧没有 ARC，
     所以**每一条 return 路径都要 `mglSafeReleaseMetalObj(&readtexid)`**（本刀 5 条路径：裁剪为空、
     plan 失败、integer/scaled 命中、以及正常落到 direct 路径）——**漏一条就是泄漏，多一条就是过度释放**。
     ④ **度量（单文件记账，壳与端口都不变）**：`+Blit.m` 语法 **86 → 68**、词汇 **154 → 101**、
     行 **1,214 → 795**；全库 **22,636 → 22,217 行（−419）**、语法 **1,300 → 1,282（−18）**、
     词汇 **2,408 → 2,355（−53）**；文件数 **6**、空 TU **0**、端口 **33**。
     ⑤ **验证（四件套全绿）**：`make -j8` **0 error**；`unused function` 唯一警告 **29 → 27**；
     `make test-all` **0**（92/0/2/94）——**第一次跑挂在 `scripts/fetch_opengl_registry.sh` 的
     GitHub `HTTP2 framing layer` 失败上（环境噪音），重试第二次通过**；
     **CTS 七簇 diff 全空**（58/1/0/59/13/39/4）；A/B（新库 vs `8d87254`）两臂 default **92/0/2**、
     flushy **91/1/2**，确定性行 **4981/4981** 与 **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑥ **教训（第二十一条，与第十六条并列的工具纪律）**：**批测目录必须按时间戳取最新**。
     本刀为了排查 `make test-all` 先 `pkill` 了正在跑的批测，然后重启；于是同一 `TAG=p047` 下留下了
     **两个** hotspot 目录（被打断的 16:32 只有 22 个 case、完整的 16:37 有 1328 个）。
     `ls -d $R/mgl-$TAG-$c-* | head -1` 抓到的是**被打断的那个**，于是"非通过 58 → 2"读起来像
     56 个用例被修好——**这是假改善，比假回归更危险**（它看起来是好消息）。
     正确取法：`ls -d ... | sort | tail -1`（或换一个 TAG）。**杀过批测就必须换 TAG 或显式取最新目录。**

### 0.74 第 117 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.73**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 22,636 行 / 1,300 语法 / 2,408 词汇**；
壳 TU **2,037 行 / 291 语法**（上限 2,400）；端口面 **33 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十七刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
壳 `MGLPlatformRendererShell.m` 291/289/2,037 · `+Blit.m` **86/154/1,214** · `+BindingState.m` **91/85/1,678**。

**`+Blit.m` 只剩 86 语法，两刀清零**：
- **刀 A（倒数第二刀）**：`-(void)mtlBlitFramebuffer:…`（dispatcher，18 语法 / 418 行）＋
  `-(void)mtlCopyTexSubImage:…`（10 语法 / 179 行）＝ 28 语法；它们的被调方法**都已经在 C 里**
  （`resolve…Attachments:` 第 146 刀、`scaledColorWithState:` 第 147 刀、`mglBlitDepthStencil` /
  `mglBlitIntegerColorWithState` / `mglBlitDirectColorWithState` / `mglBlitResolveMsaaSource` 更早）。
  仍缺的桥：`mtlReadDrawable`（**+1 端口**）、`copyTextureUploadWithDedicatedCommandBuffer`（**+1 端口**）、
  `dataWithLength`（→ `malloc`）。**12 个 render-encoder twin 已在 `mgl_blit_drivers.c` 就位**。
- **刀 B（最后一刀）**：28 个 helper ＋ `freshGLSampledRenderTargetCopyForSampling:`（11 语法）＋
  文件头 13 个 `#import` → **文件数 6 → 5**。注意 `freshGLSampledRenderTargetCopyForSampling:` 被
  `+BindingState.m` 的采样簇调用（`[self freshGLSampledRenderTargetCopyForSampling:ptr …]`），
  转它时要**保留同名语义的 C 入口**，别顺手改契约（见第 142/143/144 条）。

**规矩表（§0.62/§0.65–§0.72 十九条仍然有效）＋ 本轮第二十条**：
20. **零端口刀是纯收益刀，先做**（第 147 轮实测）：第 146 轮被 3 个 drawable 端口吃掉 7 点（净 −5），
    第 147 轮零端口（净 −11）。**排序原则**：先转"只用既有端口 + 既有 C 函数"的叶子；
    需要新端口的放一批，**一次算清壳里的 2~3 点/端口**再决定值不值得。
    另外：**搬走一个方法的唯一使用者后，`.m` 里那批 static helper 会变成 `unused function` 警告**——
    这是预期的（它们随最后一刀整文件消失），要在日志里如实记录，不要当成回归。

**下一步（优先级顺序）**：
1. `+Blit.m` 刀 A（28 语法，+2 端口）→ 刀 B（约 68 语法，0~2 端口）→ **文件数 6 → 5**。
2. `+BindingState.m`（91 语法）：stage buffer 绑定簇可先做；sample 簇仍按 §0.70 的"调用边界判定实验"走。
3. `MGLRenderer.m`（168）· `+Texture.m`（279）· `+RenderPass.m`（385）· 壳（291）。

149. **第 119 轮（P0-1 第八十九刀）：`mtlCopyTexSubImage:` 转 C——`+Blit.m` 只剩 58 语法**：
     ① **切口**：`-(void)mtlCopyTexSubImage:tex:slice:mipmapLevel:xoffset:yoffset:x:y:width:height:`
     （179 行 / 10 语法）搬进 `mgl_blit_drivers.c`，新入口 `mglBlitCopyTexSubImage`；方法删除，
     唯一调用点（`+Texture.m` 的 `mglCopyTexSubImageFenceEntry`）改成 C 调用，
     `MGLRenderer+Texture_Private.h` 里的声明删除。
     ② **端口 +2（33 → 35）**：`mtlReadDrawable:pixelBytes:bytesPerRow:bytesPerImage:fromRegion:` →
     `mglRendererMTLReadDrawablePort`（**`MGLRegionValue` 直接穿端口**，C 结构无需拆），
     `copyTextureUploadWithDedicatedCommandBuffer:…`（12 个参数）→
     `mglRendererCopyTextureUploadWithDedicatedCommandBufferPort`（`MGLSizeValue`/`MGLOriginValue` 同样直穿）。
     `mgl_renderer_ports.h` 因此新增 `#include "mgl_region_value.h"`——**第一次往端口头里加 C 结构类型**。
     ③ **两处 ObjC 设施在 C 侧的替代**：`NSMutableData dataWithLength:` 是零填充分配 → `calloc(1, size)`；
     `mglBlitCreateBufferWithBytes(_device, …)` 的 `_device` 参数**被 helper 忽略**（helper 体里就是
     `(void)device;`），所以直接用既有 twin `mglBdCreateBufferWithBytes`，**不需要 device 端口**。
     ④ **所有权**：`uploadBuffer` 原来是 strong 局部（ARC 在 return 前释放），C 侧在调用上传端口之后
     显式 `mglSafeReleaseMetalObj(&uploadBuffer)`；两个 `calloc` 缓冲区在**每一条** return 路径上都 `free`
     （本刀 9 条路径）。
     ⑤ **度量（两文件合算）**：`+Blit.m` 语法 **68 → 58**、词汇 **101 → 63**、行 **795 → 616**；
     壳 TU **2,037 → 2,078 行 / 291 → 298 语法**（2 个端口）；全库 **22,217 → 22,079 行（−138）**、
     语法 **1,282 → 1,278（−4）**、词汇 **2,355 → 2,326（−29）**；文件数 **6**、空 TU **0**、端口 **33 → 35**。
     ⑥ **验证（四件套全绿）**：`make -j8` **0 error**；`make test-all` **0**（92/0/2/94）；
     **CTS 七簇 diff 全空**（58/1/0/59/13/39/4）——本次同时校验了每个簇的
     `summary.json` 里 **`completed == total`**（1328/140/136/223/30/152/5，即第二十一条的完整性判据）；
     A/B（新库 vs `d6d3b17`）两臂 default **92/0/2**、flushy **91/1/2**，确定性行 **4981/4981** 与
     **5514/5514** 逐行一致、stderr 多重集 **307/307**。
     ⑦ **教训**：又是"端口吃掉收益"的一例（被搬文件 −10，壳 +7，净 −4）——**第十九条的预算公式再次生效**。
     因此**最后一刀（27 个 helper + `freshGLSampledRenderTargetCopyForSampling:` + 文件头）要尽量把
     新端口压到 1 个**（只剩 `currentRenderPassUsesTexture`），否则"删掉整个文件"的结构收益会被壳吃光。

### 0.75 第 118 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.74**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 22,217 行 / 1,282 语法 / 2,355 词汇**；
壳 TU **2,037 行 / 291 语法**（上限 2,400）；端口面 **33 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4）；A/B 两臂逐行一致（第八十八刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
壳 `MGLPlatformRendererShell.m` 291/289/2,037 · `+BindingState.m` **91/85/1,678** · `+Blit.m` **68/101/795**。

**`+Blit.m` 最后一刀的内容（68 语法 → 0，文件数 6 → 5）**：
- `-(void)mtlCopyTexSubImage:…`（10 语法 / 179 行）：`bindMTLTexture` 直调、`mtlReadDrawable`（**+1 端口**）、
  `copyTextureUploadWithDedicatedCommandBuffer`（**+1 端口**）、`dataWithLength` → `malloc`；
- `-freshGLSampledRenderTargetCopyForSampling:`（11 语法 / 161 行）：`uploadFullCPUTextureDataIntoTexture` /
  `restoreRenderEncoderAfterTextureUpload` 已有端口，`currentRenderPassUsesTexture`（**+1 端口**）；
  **它被 `+BindingState.m` 的采样簇调用**（调用点已改成 C 调用即可，别改契约）；
- 27 个 `static` helper（约 34 语法，绝大多数现在已 `unused`）＋ 文件头 13 个 `#import`。

**规矩表（§0.62/§0.65–§0.73 二十条仍然有效）＋ 本轮第二十一条**：
21. **批测目录按时间戳取最新**（第 148 轮实测）：`ls -d $R/mgl-<TAG>-<cluster>-* | sort | tail -1`。
    同一 TAG 下若曾 `pkill` 过批测，会留下**被截断的目录**（`summary.json` 里 `completed` 远小于
    `total`）；用 `head -1` 抓它会得到"整簇 fail 全消失"的**假改善**——比假回归更危险。
    判据：**先看 `summary.json` 的 `completed` 是否等于 `total`**，不等就换目录/换 TAG。

**下一步（优先级顺序）**：
1. **清掉 `+Blit.m`**（上一节那三块，约 68 语法，+3 端口）→ **文件数 6 → 5**，这是本轮最大的一次结构收益。
2. `+BindingState.m`（91 语法）：stage buffer 绑定簇可先做；sample 簇仍按 §0.70 的"调用边界判定实验"走。
3. `MGLRenderer.m`（168）· `+Texture.m`（279）· 壳（291）· `+RenderPass.m`（385）。

### 0.76 第 119 轮交接快照（**新会话请先读本节 + §0.51 + §0.61 + §0.69 + §0.75**）

**当前状态**：`MGL/` 内 ObjC **6 个文件 / 0 空 TU / 22,079 行 / 1,278 语法 / 2,326 词汇**；
壳 TU **2,078 行 / 298 语法**（上限 2,400）；端口面 **35 个**；`make test-all` **0**；
CTS 七簇 **diff 全空**（58/1/0/59/13/39/4，各簇 `completed == total`）；A/B 两臂逐行一致（第八十九刀的实测值）。

**逐文件剩余（语法 / 词汇 / 行数）**：
`+RenderPass.m` 385/553/6,843 · `+Texture.m` 279/1,052/6,248 · `MGLRenderer.m` 168/275/4,616 ·
壳 `MGLPlatformRendererShell.m` 298/298/2,078 · `+BindingState.m` **91/85/1,678** · `+Blit.m` **58/63/616**。

**`+Blit.m` 最后一刀（58 语法 → 0，文件数 6 → 5）——开工清单**：
1. `-freshGLSampledRenderTargetCopyForSampling:`（11 语法 / 161 行）：**唯一需要的新端口是
   `currentRenderPassUsesTexture:`（+1）**；`uploadFullCPUTextureDataIntoTexture` /
   `restoreRenderEncoderAfterTextureUpload` 都有端口。**它被 `+BindingState.m` 的采样簇调用**
   （`[self freshGLSampledRenderTargetCopyForSampling:ptr …]`，约 1477 行）——调用点改成 C 调用，
   **别改契约**（见第 142/143/144 条）。
2. 27 个 `static` helper（约 34 语法）：绝大多数 Twin 已在 `mgl_blit_drivers.c`
   （`mglBdTextureInfo` / `mglBdCreateBuffer` / `mglBdCreateBufferWithBytes` / `mglBdCreateTexture` /
   `mglBdCreateTextureView` / `mglBdReplaceTextureRegion` / `mglBdGetTextureBytes` / `mglBdCreateRenderEncoder` /
   `mglBdDefaultRenderPassState` / `mglBdRenderPassAttachment` / `mglBdEndRenderEncoder` / `mglBdSetRender*` /
   `mglBdDrawPrimitives` / `mglBdEndComputeEncoder`? / `mglBdSetCompute*`? / `mglBdDispatchThreads`? /
   `mglBdEndBlitEncoder` / `mglBdCopyTexture` / `mglBdCopyTextureToBuffer` / `mglBdCopyBufferToTexture` /
   `mglBdSynchronizeTexture`）；**缺的只可能是 compute 那几个**（`mglBlitSetComputePipeline/Texture/Bytes`、
   `mglBlitDispatchThreads`、`mglBlitEndComputeEncoder`），要先查它们是否还有调用者。
3. 文件头 13 个 `#import` + `@end`：文件删除后自动消失。
4. 删完后要**同时**把 `MGLRenderer+Blit_Private.h` 里对应的声明清掉，并确认
   `MGLRenderer.m` 的 `@interface MGLRenderer (Blit)` 声明不残留（否则会多出
   `-Wincomplete-implementation`），最后把 `MGLRenderer+Blit.m` 从构建里移除。

**规矩表（§0.62/§0.65–§0.75 二十一条仍然有效）**：本轮没有新规矩——第十九条（端口预算）与
第二十一条（批测目录取最新 + 校验 `completed == total`）已经覆盖了本轮遇到的全部情况。

**下一步（优先级顺序）**：
1. 清掉 `+Blit.m`（上一节清单）→ **文件数 6 → 5**。
2. `+BindingState.m`（91 语法）：stage buffer 绑定簇先做；sample 簇仍按 §0.70 的"调用边界判定实验"走。
3. `MGLRenderer.m`（168）· `+Texture.m`（279）· 壳（298）· `+RenderPass.m`（385）。
