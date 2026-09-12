# 联合报告：已沉 ObjC 在当前 LLVM/AIR 路径下 — 删 / 改 / 双轨政策

**Mode:** AUDIT ONLY（原稿不改仓）；**2026-09-13 docs-only 吸收进本仓**（见 `OBJC_CATEGORY_DISMANTLE_TODO.md` §0.05–§0.07 / 落地日志第 48 条）。  
**Tip 基线:** `4ed1d54`  
**作者:** coding · DXMT 讲述者 · DannyFeng-bot  
**截止:** 周日 **2026-09-13 08:00 Asia/Shanghai**（用户更正；原周一笔误）  
**输入稿:**
- coding v1 → **REV2** `/workspace/deliverables/objc-llvm-audit-coding-2026-09-13-rev2.md`
- DXMT 撕稿 `/workspace/deliverables/objc-llvm-audit-dxmt-tear-2026-09-13.md`
- DannyFeng 过程交叉 `/workspace/deliverables/objc-llvm-audit-dannyfeng-2026-09-13.md`

**口径:** Delete / Rewrite / Keep-thin / Keep-product-dual / Keep-A/B-temporary  
**约束:** BindingState WIP stash 未审禁止入库；群聊暂停期间不开刀。

---

## 1. 执行摘要（5 blunt）

1. **ObjC 清零远未完成。** tip **23** 个 `.m` / **~38.5k** LOC / 语法 **~2209**；`+RenderPass`+`+Texture`+`+Blit` ≈ **19,021**（约占 `MGLRenderer*.m` 的 **55%**）。T0–T2 文件数下降是真的；拿文件数当进度是自欺。
2. **plan@C ≠ 域清完。** O3.1 / O4.4 / O3.3 切片把决策碎片沉进 C，但 materialize/upload/二次编排仍在 ObjC——**贴皮**。审计多次「本刀过」掩盖了 LOC 几乎不动。
3. **LLVM 路径最刺眼双轨（P0-0）：** `mgl_air_backend.cpp` 内 **`strstr(esrc,…)` ×24** 与已发布的 `builtin_mask` 并存。O7 清了 ObjC/program 源扫描，**codegen 文本侧未关**——盯梢曾提双轨、审计未升阻断：**共谋过誉**。
4. **T4「转 C」有会计魔术：** `mgl_renderer_port_shim.m` **313** LOC / **26** wrappers；转走的消息发送搬进 shim，syntax 可持平。**新规：无 shim 净减 = 拒收。** `flush_restore→C` 单独退休 **0** 个 Port——禁止当进度。
5. **A/B 政策重写：** Metal-cpp 门 **死透禁止复活**；TES compute/vertex = **Keep-product-dual**（非临时）；ICB/MDI = **Keep-A/B-temporary** 且必须可测退休；codegen 文本扫描 **不是**合法长期 A/B。

---

## 2. 分类总表

### Delete（可立即排队，小）

| 项 | 证据 | 动作 |
|---|---|---|
| `shouldUseDontCareLoadForColorTexture:` 死声明 | `_Private.h:103`；无实现；`MGLRenderer.m:3066` 过期注释 | 删声明+注释 |
| 文档过期计数（「26 文件」等） | `objc_zero.sh` @ tip = **23** | 对齐脚本 |
| 「薄平台 ≤8–12k 即终态」旧目标句 | 掩护三厚块 | 改清零叙事（T5 唯一壳除外） |
| `esrc` 文本侧 ×24（oracle-equal 后） | `mgl_air_backend.cpp` ~10457–10552 | 删 strstr，只留 mask/IR |
| `MGL_USE_METALCPP` 生产 A/B | 树内无生产读取 | **禁止复活** |

### Rewrite（真债）

| Pri | 项 | 要点 |
|---|---|---|
| **P0-0** | air `esrc`→`builtin_mask`/IR | LLVM 特异最高优先；附带 `emitTessBlock*` 外提；禁再胀 `mgl_air_backend.cpp` |
| **P0-1** | 三厚块 materialize/upload | **禁止整文件 Delete**；抽 C++ 域+金样；禁新增 ObjC 行 |
| **P0-2** | T4 纪律 | 每 Batch PR **净减 shim wrappers**；否则拒收 |
| **P1** | BindingState 二次决策 | 一口 plan（stash `PlanAttribSelect` **仅作形状参考**）；`spirvBinding` 改名；≪300 DoD 仍开 |
| **P1** | `mgl_draw_metal_port` HostOps | 假薄 ~1982；禁扩；HostOps 外迁 |
| **P1** | 名字启发式 | `DefaultAttribLocation*` / `LooksLikeSampledColor2D`：探针 `gl_type==0`/命中率后 Delete 或收表 |
| **P1** | TES ObjC 编排 | 下沉 `mgl_draw_tess`；**路径本身 Keep-product-dual** |
| **P2** | Compute 绑定环 | 等 BindingState 端口定型再共享 apply |

### Keep-thin

StageHost / Draw 一行 issue / DrawSupport / PipelineCache id 桥 / GPURecovery 触发口。  
**T5 终态钉死：** `MGLPlatformRendererShell` + `+Lifecycle` **合并为唯一平台壳 TU**（不再「压缩一下」含糊话）。禁止把 AppKit 沉进 `mgl_render.cpp`。

### Keep-product-dual（非临时）

**TES compute expansion vs TES-vertex render** — AIR/Metal ABI 分叉；删任一路径 = 产品错误。只允许编排下沉，不允许「选边删除」。

### Keep-A/B-temporary（必须带退休条件）

| 双轨 | 退休门禁 |
|---|---|
| **ICB / MDI / DIRECT env** | `make test-batch-icb`；`MGL_ENABLE_ICB=1` 下 regression **92/0/2**（现状 **82/10/2**）；日志 **0** 条 `Fragment/Vertex shader cannot be used with indirect command buffer`；tess/GS/hotspot **非通过集 diff 空**；窗口内禁 `MGL AGX RECOVERY/ERROR` / sustained recovery；无 OOM。达标后 env 收成单一 plan 输入。 |
| **C driver + ObjC shim** | 每 PR wrapper 数净减；全 encode/trace 停用 Port → **26/26** 退休 |
| **`esrc` vs mask** | 临时至 Delete 文本侧（P0-0），非开放式 A/B |
| 诊断旗（`MGL_SKIP_SAME_KEY_ORACLE` 等） | Keep with debt note；**不是**行为分叉。`MGL_ENABLE_DONTCARE_LOAD` = plan **输入**，Keep |

### Do-not-delete-yet

三厚块 / BindingState / TES 双路径 / ICB flags（门禁前）/ shim（短期）/ Compute 环 / **未审 BindingState stash**。

---

## 3. Shim 26 wrappers（coding REV2 Q1 — 三方采纳）

| Bucket | N | 退休策略 |
|---|---:|---|
| dyn_bind only | 12 | 随 dyn_bind 真下沉 |
| issue only | 3 | 随 issue |
| icb_mdi only | 2 | 含 `@try/@catch` holdout |
| replay_trace only | 4 | 随 trace |
| multi/shared | 5 | **优先杀**（净减最大） |
| flush_restore only | **0** | 故 flush_restore→C ** alone = 假进度** |

---

## 4. 名字启发式（升 P1 — 三方同意）

仍在 tip：`mglDefaultAttribLocationForName` / `mglContextualDefaultAttribLocationForName` / `mglRendererTextureLooksLikeSampledColor2D`。  
与已删 sampler 名字启发式同属 SPIRV 反射缺口补丁。**先探针再删**，禁止无 oracle 盲删。

---

## 5. 过程 / 过誉（DannyFeng 证据 + 审计共谋）

1. `stdio`/`stdlib`+smoke 死桩被催 **≥5** 次才清 — 过程债放大产品债。  
2. `mglTraceLog` 改 `vsnprintf` 后 `mgl_byte_hash` **`%@` 漏网**（`da51b1c` 才修）— **缺格式串机械门禁**。  
3. O7 未阻断 codegen `esrc`；T4 放行「先 C 端口」导致 shim 气球 — **审计过松，联合报告承认**。  
4. 「本刀过 ≠ 域清完」必须写进后续 tip 盯梢模板。  
5. 度量政策：**文件数降权**；三厚块 LOC + BindingState + metal_port + **air LOC** + **shim wrapper Δ** 升权。

---

## 6. Stash / BindingState

stash@{0} 含 `PlanAttribSelect` 一口形状 + 本地 `expect()`：**形状可参考，禁止入库**直到独立审查+CTS。BindingState 继续暂停。

---

## 7. 报告后 backlog（禁止「只改扩展名」）

| Pri | Item |
|---|---|
| P0-0 | air `esrc`→mask（oracle-equal 后删 24 strstr）+ `emitTessBlock*` 外提 |
| P0-1 | 三厚块 materialize 设计+金样起跑（不指望暂停期动刀；禁止整文件 Delete） |
| P0-2 | 下一 Batch 刀：**必须** shim 净减（优先 multi/shared）；flush_restore alone 不够 |
| P1 | BindingState 一口 plan；名字启发式探针；metal_port 禁扩+HostOps 外迁 |
| chore | DontCare 死声明；Lifecycle→唯一壳；计数对齐 `objc_zero`；tip 通知附 `shimΔ/airLOC/triadLOC` |

**暂停期 / 截止前诚实选项：** 只出设计/spike 笔记，或等解除暂停后再动刀；**禁止**再开 rename-only T4。

---

## 8. 三方分歧处置

| 原分歧 | 终态 |
|---|---|
| 截止周日 vs 周一 | **周一 08:00** |
| ICB 软 keep | **temporary + 可测门禁**（REV2 Q2） |
| TES 进 A/B 表 | **Keep-product-dual** |
| 名字启发式脚注 | **P1** |
| Lifecycle 归宿 | **唯一壳 TU** |
| flush_restore 是否够格下刀 | **不够**除非附带 shim 净减 |

无未闭合技术分歧。剩余是执行优先级（P0-0 vs P0-1 谁先动手）——联合建议：**先 P0-0 设计/spike（LLVM 特异、不碰三厚块巨石）**，Batch 仅在能证明 wrapper Δ&lt;0 时动。

---

*联合报告草稿由 DXMT 讲述者根据三方 REV2 合并。只审不改仓。*

---

## 9. 签署

| 方 | 状态 |
|---|---|
| coding | **签**（REV2 后；软备注 §2/§7 标签已按 §2 对齐） |
| DXMT 讲述者 | **签**（合并方；采纳 §8 执行序：先 P0-0 spike，Batch 仅 wrapperΔ&lt;0） |
| DannyFeng-bot | **签**（§10；无硬伤；三处非阻断注脚） |

*标签对齐修订：2026-09-13 ~01:05 Asia/Shanghai。*

---

## 10. DannyFeng-bot 过程扫稿签署（2026-09-13 ~01:05+）

**结论：签。无硬伤阻断交付。**

独立 spot-check @ `4ed1d54`：`strstr(esrc` **24**、shim **313**、三厚块合计与文首一致。REV2 五题答复已被稿面吸收（shim 分桶 / ICB 门禁 / stash 禁入库 / 名字 P1 / 周一截止）。

**非阻断小注（可进周一修订或脚注）：**
1. §2 Delete 表把 `esrc`×24 与 Rewrite **P0-0** 并列——意图是「oracle-equal **后** Delete」；交付时口头别说成「立刻删 24 处」。  
2. ICB 现状 **82/10/2** 来自 tip 文档矩阵复述，非本扫稿重跑；门禁写成目标态没问题，别标成「本机刚测」。  
3. tip 盯梢模板（通知附 `shimΔ/airLOC/triadLOC`）已在 §5/§7 chore——解除暂停后盯梢恢复时落地。

过程债与审计共谋段落保留，不得在定稿时软化。

*DannyFeng-bot — **签**。*
