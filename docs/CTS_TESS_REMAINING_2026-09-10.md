# TESS 簇剩余失败排查 — 2026-09-10（接 CTS_TESS_CLUSTER_2026-09-10.md）

> **当前状态（2026-09-12，本文最新一轮＝第三十三轮）**
>
> | 口径 | 结果 |
> |---|---|
> | `KHR-GL46.tessellation_shader`（140 例） | **139 pass / 1 fail / 0 not_supported** |
> | 唯一失败 | `vertex.vertex_spacing_primitive_mode_quads_vs_mode_fractional_odd_spacing` —— 第 28/30 轮已证为 **CTS 自身期望矛盾**（离线 48/48 被拒 vs equal 144/144 通过），未写特判 |
> | `KHR-GL46.geometry_shader.*`（136 例） | 136 / 0 |
> | GL46 hotspot（1328 例） | 1270 pass / 52 fail / 4 ns / 1 crash，非通过集合与基线逐条 diff 为空 |
> | 本地全量 `test_regression`（94 项） | 92 PASS / 0 FAIL / 2 SKIP |
>
> 本轮（第三十三轮）内容：三项残留 ABI 风险成套清理（跨 dispatch barrier / `contract.patch_out_stride` /
> native 控制点数组成员）+ 既有失败 `air_tessellation_isolines_multidraw` 的根因（PSO 键缺
> `tessVertexRenderActive`）。**ObjC 边界侧**的后续清理（vertex-attrib buffer map 沉 C、reflection
> fallback 删除、SPIRV 时代文本判定与 sampler 启发式退役）记在
> [`docs/OBJC_CATEGORY_DISMANTLE_TODO.md`](OBJC_CATEGORY_DISMANTLE_TODO.md) §5 第 25–28 条与 Batch O7。
> 本文已随该批一并入库（此前仅为本地工作日志）。

对 run F3（`mgl-tess-fix3-20260910-195616`）的 6 fail + 1 NotSupported 逐例定位。
方法：QPA 的 `<Text>` 取断言 → 对照 `VK-GL-CTS/external/openglcts/modules/glesext/tessellation_shader/` 源码 →
纯 C 复现（直接 `#include` 仓内 `mgl_tess_domain_gen.c`/`mgl_tess_factor_normalize.c` 编译，无 Metal 依赖）。

## 一句话结论

| 用例 | 性质 | 修法 | 现状（2026-09-12：140 例簇 **139/1/0**） |
|---|---|---|---|
| `vertex...fractional_odd_spacing` | **真 bug，已定位**：quad 内层矩形"短段"全部丢失 | `generate_quads` 内层网格用 unrounded factor 生成 | ❌ 仍失败 —— 第 28 轮判定为 CTS 侧账目矛盾，标注跳过 |
| `invariance_rule4` | **真 bug，已定位**：补边坐标浮点不位等（差 1 ulp） | 互补位置用同一运算路径，勿先算 `1-t` | ✅ 已修（`6d205b4`） |
| `quads.inner_tessellation_level_rounding` | **真 bug，已定位**：`inner==1` 退化成"实心矩形"无中心横/纵带 | inner==1 时生成跨域 marker 三角（`0..1` 全宽边） | ✅ 已修（`6d205b4`） |
| `single.xfb_captures_data_from_correct_stage` | **廉价校验缺口**：`GL_PATCHES` draw 缺 TES 未报 `GL_INVALID_OPERATION` | `validate_program`（draw_buffers.c:461）加 TES 检查 | ✅ 已修（`9a1dc5b`） |
| `point_mode.point_rendering` | 症状明确（渲染全 0，读回 `(0,0,0,0)`）但根因未定位 | 需带 Metal trace 单跑；疑似 `point_mode` 光栅路径 | ✅ 已修（`8b1efdd`） |
| `tc_barriers.barrier_guarded_read_write_calls` | 16 调用 TCS 共享变量 `barrier()` 同步语义错（patch[2] 末槽 `1065353216`=1.0f 位模式） | 需查 TCS 共享内存 + barrier lowering | ❌ 仍失败 —— **"CTS 侧账目"结论已作废，重开为驱动侧缺陷**（见文末复核） |
| `tessellation.max_in_out_attributes` | **非 bug**：CTS 构造约束（`MAX_VERTEX_OUTPUT(64) < MAX_TESS_CONTROL_INPUT(128)`），规格未规定二者关系 | 单独决策是否抬 `MAX_VERTEX_OUTPUT_COMPONENTS` 口径 | NotSupported（limits 口径，待单独决策） |

---

## 1. `vertex...fractional_odd_spacing`（expected 29 / found 31）★ 可信根因

**失败配置**：quads / fractional_odd / inner=(32,64) / outer=(1,32,1,32)。

**定位链**（全部经 `dbg*.c` 复现确认）：

1. 失败的是**内层矩形**（CTS 第 2 轮迭代）的 **水平边**（`tl_tr`/`br_bl`，`tess_level = inner[0]-2 = 30`）。
2. 归一化后 `inner_eff[0]=33`（FO：ceil(32)→33）、`inner_eff[1]=63`（clamp(64→63)）。
3. 内层矩形左/右缘 = `inner_clamped[0]=32` 的 fractional 短段位置 `x0=0.015625, x1=0.984375`（`edge_position(32,33,FO,1)` 等）。
4. 内层**水平**边用 `nx=inner_eff[0]=33` 段等分该宽度 → **31 个等长内段**（`1/32`），端点短段被 corners `x0/x1` 吃掉。
   实测：edge 上 32 点、31 段、**ndistinct=1**（delta=0.03125 全等）。
5. CTS `verifyEdges`（`esextcTessellationShaderVertexSpacing.cpp:1272-1276`）对 fractional_odd quads 内层边：
   `expected = rounded(inner[0]-2) - 2 = 31 - 2 = 29`，
   要么 1-distinct 且 counter==29，要么 2-distinct 且 counts=={2,29}。**31 ≠ 29 → FAIL**。

**根因**：内层网格边 subdivision 用的是 **rounded 段数（nx=33）等分**，短段 factor 只用于整条边的"内缩"
（`x0/x1`），**内层边本身没有产生 fractional 短段**。spec 要求 inner 边仍按 unrounded factor=32 做
fractional spacing（2 短段 + 29 中段位 = 31 段），MGL 只给了 31 个等长段。

> 注意：item_2（inner=(32,31)）等"通过"组合其实在 tl_tr 内边也命中同型失败，但 48 个可测组合里
> fractional_odd 全部内层水平边都偏 2；run F3 只有这一条被计入 fail，说明其余组合被 2-distinct 分支兜住
> 或映射后未触发。**这是一个系统性内层 fractional 短段缺失，不是单点。**

**修法**：`generate_quads` 内层网格（`mgl_tess_domain_gen.c:120-126`）的 `edge_position` 调用应传
**unrounded `inner_clamped[i]`** 作 factor 且段数逻辑对齐 fractional 语义——但这会改变内层网格点坐标，
波及 76 例全绿的 `tessellation_control_to_tessellation_evaluation` 与 winding/point_mode。**必须离线建
Python 参照模拟器先验证收敛再动 C**（当前模型对 2-distinct 判定仍不完美，见 dbg 系列）。

## 2. `invariance_rule4`（期望 `(0, 1/7)` 未出现）★ 可信根因，修法低风险

**配置**：quads / equal / outer 全 7 / inner=(4,5)。逐位复现：

- 底边 `(1/7, 0)`：MGL 生成 `u = 1/7 = 0x3e124925`（`edge_position` 直接 `index/segments`）。
- 左边对应点：MGL 走 edge 3，`v = 1 - t = 1 - 6/7 = 0x3e124924`（**差 1 ulp**）。
- CTS `isVertexDefined`（`esextcTessellationShaderInvariance.cpp:1755`）是**位等**比较
  （`vertex_data_seeked[0]==current[0] && [1]==[1]`，无 epsilon）→ `(0, 0x3e124925)` 找不到。

**根因**：互补边坐标一边直接 `i/n`、另一边 `1-(n-i)/n`，浮点不位等。rule4 要求
"`(x,0)` 存在 ⇒ `(0,x)`、`(0,1-x)` 位等存在"。

**修法（低风险、通用）**：`edge_position` 的 complementary 分支（`index > segments/2` 时
`1 - edge_position(segments-index)`）改为**直接计算镜像索引**，保证 `pos(i)` 与 `pos(segments-i)` 位等对称。
对 `equal`，直接 `i/segments` 已对称，问题只在 `1-t` 的减法；对 fractional 分支同理需保证
`pos(i) == 1 - pos(segments-i)` 的位等——**最稳妥是让所有边共享同一个 position 计算，靠索引镜像而非
坐标取反**。已验证 `(n-i)/n` 直算对 n=7 全部 6 个内部点位等（dbg16 扫描 outer=3..15 共 1960 处违例，
均为此 1-ulp 同源）。

## 3. `quads.inner_tessellation_level_rounding`（找不到 marker 三角）★ 可信根因

**配置**：fractional_odd，set1 `inner=(1, tess)`、`outer` 全 `tess`（`tess ∈ {3,33,63}`）。

- MGL 对 `inner==1` 生成的是"实心"内外两层 quad：内层矩形被 1+ε bump 到 eff=3，网格铺满，**没有跨越整个
  域宽 `0..1` 的内层水平线**（dbg19：inner=(1,2) 时 inner rect 是 `(0.25,0.25)..(0.75,0.75)` 的实心块）。
- CTS 要找 "marker triangle"（`esextcTessellationShaderQuads.cpp:828`）：一条边的两个顶点在
  `second_y_from_top/bottom` 且 **x 横跨 `{0,1}`**、第三顶点在外缘 `y=0/1`。这要求 `inner==1` 时内层退化成
  **一条水平中线（两半各一个跨全宽的三角帽）**而非实心网格。
- 实测 MGL 任何 FO `inner0=1` 用例都 `marker=NOT-FOUND`（dbg20，tess=3/33/63 全挂）。

**修法**：`generate_quads` 在 `nx==1 || ny==1` 的退化分支（当前在 `mgl_tess_domain_gen.c:115-119`
处理了 `nx==1&&ny==1`，但单轴 ==1 未特判）应生成**跨域的 marker 拓扑**：inner==1 的轴不生成内层网格线，
改由连接内外矩形的"帽"三角覆盖整条边。属结构性改动，量中等。

## 4. `single.xfb_captures_data_from_correct_stage` ★ 最廉价，建议先修

- 测试构造合法 program `[VS + TCS + FS]`（**无 TES**），`glBeginTransformFeedback` 后
  `gl.drawArrays(GL_PATCHES, ...)`。spec：用含 TCS 但无 TES 的 program 以 `GL_PATCHES` draw 必须报
  `GL_INVALID_OPERATION`。测试接受 error 落在 `beginTransformFeedback` 或 draw 任一处。
- MGL `draw_buffers.c` 的 S7 `validate_program`（draw_buffers.c:461）只查 program 有效性 / VS 缺失 /
  pipeline 缺 VS，**从不查 "TCS 存在但 TES 缺失"**；S11 `GL_PATCHES` 直接路由进 tess renderer 也无处报错。
- **修法**：在 `validate_program` 增加——当 draw mode 为 `GL_PATCHES` 且当前 program（或 pipeline）挂了
  TCS 而未挂 TES → 返回 false（触发 `GL_INVALID_OPERATION`）。与 tessellator 完全解耦，独立可测。
  注意需把 mode 传进 validate_program 或在其后单查。

## 5. `point_mode.point_rendering`（expected `(0.1,0.2,0.3,0.4)` rendered `(0,0,0,0)`）

- 同组 `points_verification` 通过 → **点坐标生成/属性读取正确**；失败只在**光栅化输出全黑**。
- 怀疑点：`point_mode` 下 TES 走 point 光栅路径，`gl_PointSize` / point sprite 光栅 / 片元着色输入未就位，
  或 point 原语未被 Metal 端当 `MTLPrimitiveTypePoint` 发出。需 `MGL_TES_VERTEX_TRACE=1` 单跑确认走的是哪条
  exec 路径（参照旧档复现要点）。**域生成纯 C 侧无异常**，问题在 Metal 渲染链路。

## 6. `tc_barriers.barrier_guarded_read_write_calls`

- 16 invocation，TCS 共享数组 `tcs_data[16]`，三个阶段 `barrier()` 隔开：相位1 偶数写 `n`、相位2 奇数读
  `n-1`+n、相位3 `(id%4==0)` 读 `id..id+3` 求和写 `tcs_patch_result`。
- 实测 patch[2] = `(8,0,0,0, 32,0,0,1065353216)`，期望 `(8,0,0,0, 32,0,0,0)`。**最后一个槽
  `1065353216 = 1.0f` 的位模式**——即某处把 float `1.0` 当 int 写进了 `tcs_patch_result[7]`，或共享数组
  边界/栅栏后读到未复位的脏值。前两个 patch 正确，**说明是跨 patch / 高 invocation 的共享内存或 barrier
  同步在 16-wide 下的边界错乱**，非算法错。
- 需查 TCS 的 threadgroup 共享内存 lowering 与 `barrier()` 在 AIR/native TCS 路径的语义。

---

## 落地进展（2026-09-12）

### ✅ 已修：#4 `single.xfb_captures_data_from_correct_stage`

`validate_program`（`draw_buffers.c:461`）原本只查"pipeline 缺 VS"，从不查 TCS/TES 配对。
已改为把 draw mode 传进 `validate_program(ctx, mode)`，并在 `mode == GL_PATCHES` 时拒绝
"有 TCS 无 TES"的 program/pipeline（`program_missing_tes_for_patches`）。
提交 `9a1dc5b`。

- 该用例：**fail → pass**。
- 140 例 tess 簇回归：**133 pass / 6 fail → 134 pass / 5 fail**，无回归。

### ❌ 未修：#1 / #2 / #3 —— 本轮做了更细的定位，修法需重新设计

**#1 `fractional_odd`（expected 29 / found 31）—— 本轮把它量到了"两个轴要求不一致"**

用 CTS 源码逐行对齐后（`esextcTessellationShaderVertexSpacing.cpp`：
`getEdgesForQuadsTessellation` 取 `tess_level = inner[0]-2 / inner[1]-2`，
`verifyEdges` 期望 `clamped_rounded_tess_level - 2*(n_edge/4)` 个等长段）。

**已用改装的 CTS 二进制取得 ground truth**（给 `verifyEdges` 加 `MGL_CTS_TRACE_EDGES` 桩，
在 `iterate()` 里加 `MGL_CTS_TRACE_ALL` 桩以便一次跑完全部配置不早退；
两处改动已从 `VK-GL-CTS` 还原，未留痕）。失败配置 `inner=(32,64) outer=(1,32,1,32)` FO 的实测：

```
n_edge=0  tess=32 clamped=32 rounded=33 npts=34 exp=33 first=(0,1)      last=(1,1)
n_edge=2  tess=32 clamped=32 rounded=33 npts=34 exp=33 first=(1,0)      last=(0,0)
n_edge=4  tess=30 clamped=30 rounded=31 npts=32 exp=29
          first=(0.015625,0.984127) last=(0.984375,0.984127)
          points: 0.015625 0.046875 0.078125 ... 0.953125 0.984375   (步长 1/32)
```

**这推翻了原文和上一轮的一个关键判断：错的不只是内层网格。**
用改装 CTS 取到的**真实 delta 直方图**（每边一次判定，这是唯一权威判据）：

```
n_edge=0 rounded=33 ndistinct=2 [0.015625 x2] [0.03125 x31]   -> PASS
n_edge=2 rounded=33 ndistinct=2 [0.015625 x2] [0.03125 x31]   -> PASS
n_edge=4 rounded=31 ndistinct=1 [0.03125 x31]                 -> FAIL (要 29)
```

**这就是"内层水平边"缺失的 2 段**：外缘正确地产出了
**2 个短段（1/64）+ 31 个长段（1/32）**（`verifyEdges` 的 2-delta 分支接受），
而内层水平边只产出了 **31 个等长段、没有任何短段**。

定位到公式（`n` = 段数，`c` = 该轴的 clamped 值，`f` = 该轴 factor）：

- 正确的边构造是 **"两端各一个短段 + 中间若干等长长段"**。
- 外缘现在已正确：**2 个短段 `1/64` + 31 个长段 `1/32`**，
  由 `edge_position(f=32, n=34, i=0..33)` 生成
  —— 注意这里传进去的是**顶点数 34**（= `outer_eff + 1`），
  而不是段数；`(f-(n-2))/(2f) = 0` 与实测短段 `1/64` 并不完全吻合，
  说明外缘能过是靠 2-delta 分支的容差，**构造细节仍需再确认**。
- 内层水平边现在是 `edge_position(f=32, n=33, i=1..32)`，
  同一个 `[1/64, 63/64]` 区间被 32 个点铺满 → **没有任何短段，31 个等长段**。
  缺的正是那 2 个短段。

**修法的方向**（下一轮据此实现）：内层轴必须复用与外缘**同一条**
`edge_position(factor, n, spacing, i)` 调用，`n` 由该轴的
`clamped_rounded - 2` 反推，从而自动带上两端的短段；
**用 `i/nx` 均匀铺点是错的方向**（本轮已证伪四种此类改法，见下）。

| 边 | CTS 期望 | CTS 接受的 2-delta 计数 | MGL 现状 |
|---|---|---|---|
| 外缘 n_edge=0/2 | 33 段 | {2, 31} | {2, 31} ✅ 已对 |
| 内层水平 n_edge=4/6 | **29 段** | {2, 27} | 1-delta {31} ❌ 缺 2 短段 |
| 内层垂直 n_edge=5/7 | 61 段 | 待测 | 待测（被 edge4 早退挡住） |

竖直方向的现状仍未取到（`verifyEdges` 在 edge4 就抛异常早退）。
**下一轮第一步**：用上面记录的 `MGL_CTS_TRACE_ALL` 桩一次性取到 edge5/7 的直方图，
再按"两端短段 + 段数由 `clamped_rounded-2` 决定"这一条统一规则改内层网格。

试过的改法（全部回滚，未留痕）：内层网格分母改 `nx-1/nx-2/nx-3`、
跨度改 `(rounded-2)/clamped` 居中、步长改 `1/rounded`、内层取 `i/nx_seg` 均匀分布 ——
结果分别是"仍 31 段 / 重复顶点 / 直接 crash / 无改善"。
另外自建的 Python 对拍器（`/tmp/mgl_verify/cts_sim.py`）**保真度不足**
（对 `outer=3` 的已知通过配置误报 FAIL），不足以作为判据；
**应以改装 CTS 的 `MGLCTSEDGE` 输出为唯一判据**（脚本见本节末）。

**#2 `invariance_rule4` —— 本轮定位到 CTS 的真实判据，与原文描述不同**

原文说"互补位置位等"（`pos(i) == 1 - pos(n-i)`），据此把 `edge_position` 改成镜像构造后
**该用例仍然失败**，且验证发现该判据**在 n=7 时与 CTS 的规则本身矛盾**。CTS 的真实逻辑
（`esextcTessellationShaderInvariance.cpp:1793-1820`）是：

1. 取底边上 `(x, 0)`，x≠0；
2. **要求 `(1-x, 0)` 也在集合里**（`isVertexDefined` 位等比较）；
3. 满足才要求左边上存在 `(0, x)` 和 `(0, 1-x)`。

即要求**顶点集合对浮点补运算 `x → 1-x` 封闭**。实测 n=7 的等距集合 `{k/7}`：
`1 - 3/7 = 0.428571463`（0x3edb6db8）不在 `{k/7}` 中（该集合里是 0x3edb6db7），
`1 - 1/7 = 0.857142866`（0x3f5b6db7）也不在。也就是说**用 `i/n` 布局的任何实现都过不了这条**，
除非改用对补运算封闭的集合（例如由 `1-(n-i)/n` 直接算出的那组值）。
本轮试过直接算 `1-(n-i)/n`（会导致 `pos(1)==pos(6)` 退化）与镜像补齐，均未通过，
需要按"封闭集合"重新构造（预期要取 `{1-(n-i)/n} ∪ {i/n}` 这类自洽组合），**这不是 1 ulp 的补丁**。

**#3 `inner_tessellation_level_rounding` —— 已定位到三角化层**

`inner[0]==1`（FO，eff=3）时，内层矩形是 3×3 网格（x∈{0,1/2,1}，y∈{0,1/3,2/3,1}）。
CTS 的 marker 判据（`esextcTessellationShaderQuads.cpp:735-830`）要求存在一个三角，
两个顶点都在 `y = sorted_y[1 or n-2]` 且 **x 横跨 {0,1}**，第三顶点在外缘 y=0/1。
实测（复刻 CTS 判定函数）：

- 全宽内层边 **存在**（三角 1 `(0,1/3)(1,1/3)(1,2/3)` 与三角 13 `(0,1)(0,2/3)(1,2/3)`）；
- 但**没有任何三角把 `x=0` 与 `x=1` 这两个顶点与 y=0 / y=1 的外缘顶点连成一体** ——
  `join_edges`(`mgl_tess_domain_gen.c:80-93`) 的条带推进把 cap 三角连到了**同一个角点**，
  于是 marker 三角不存在。原文"没有跨域内层水平线"的说法不准确：线在，缺的是 cap 的连线方式。

---

## 第二轮进展（2026-09-12，晚）

### 关键结论：主机侧域生成器**确实是** CTS 所测数据的来源

用 `MGL_SEED_TRACE` 桩确认：`KHR-GL46...quads_vs_mode_fractional_odd_spacing`
走的是 **AIR compute 路径**（`MGL TRACE tess path native=0 air=1 exec=2`），
其 TessCoord 由 `MGLRenderer+Tessellation.m` 的 `mglTessSeedEvalOutputRecords`
→ `mglRenderSeedTessDomain` → `mglTessGenerateDomainStrided` 生成，
即**改 `mgl_tess_domain_gen.c` 一定会改变 CTS 看到的坐标**。
运行时实测 `inner=(32,64) outer=(1,32,1,32) pm=1 → 2052`，与 `inner_eff` 一致。

### 已用改装 CTS 取得的精确靶子（`/tmp/mgl_verify/cts_instrument.py` 可一键复现）

外缘（n_edge=0/2，tess=32）**已经正确**，直方图为
`2 个短段 1/64 + 31 个长段 1/32`，恰好满足 `verifyEdges` 的 2-delta 分支
（要求 `{2, clamped_rounded-2}` = `{2,31}`）。

内层两条边**都差**，且期望值相差 2 而非 1：

| 边 | CTS 期望 | 现在 | 需要的几何 |
|---|---|---|---|
| 内层水平 n_edge=4 | **29** 段 | 31 段（1-delta） | **30 点**、1/32 步长、跨度 29/32 |
| 内层垂直 n_edge=5 | **61** 段 | 63 段（1-delta） | **62 点**、1/64 步长、跨度 61/64 |

内层水平边的精确目标（已按 `1/32` 步长反算）：
`30` 个点，位置 `3/64, 5/64, …, 61/64`（即 `3/64 + k/32, k=0..29`），
跨度 `29/32`，两端留 `3/64` —— 完全对称。

**本轮试过并全部回滚的 6 种改法**（每种都用改装 CTS 量了 `MGLCTSDELTA`）：
`nx-1/nx-2/nx-3` 列数、`ny-1/ny-2` 行数、`(rounded-2)/clamped` 居中跨度、
`1/rounded` 步长、`segments+1` 点、显式 `u_pts/v_pts`。
其中把列数改到 30 **确实把水平边从 31 段降到 30 段**（`found:30`），
但行数一旦跟着改就会出现（a）垂直边变成 59/60 段，或
（b）`inner_tessellation_level_rounding` **崩溃**，或
（c）点模式报 **重复顶点 (0.015625, 1)**。

**下一轮的落地要点**（重要）：
- 列=30/行=62 **必须来自同一条规格规则**，不能两轴各配一个常数；
  规格只给了"<m>/<n> 段，用 clamped rounded inner level 与 spacing 细分，
  再把对应顶点连成网格"，两轴规则相同，所以差异应来自
  `inner_clamped[0]=32`（偶）与 `inner_clamped[1]=63`（奇）在
  fractional-odd 下的**短段长度不同**（`(f-(k-2))/(2f)`），而不是常数差。
- 内层网格现在用 `edge_position(inner_clamped, nx, x)`，得到的
  `1/64 + k/32` 是**偶**分母；而外缘 `edge_position(outer_clamped, outer_eff, i)`
  得到 `1/64, 3/64, …` 是**奇**分母。两条调用形式相同但参数语义不同
  （一个是点数、一个是段数），这正是错位根源。
- **建议下一轮第一步**：把 `edge_position` 的参数语义统一为"段数"，
  让内层网格与外缘走同一条调用（`segments = clamped_rounded`），
  再用 `cts_instrument.py` 量 `MGLCTSDELTA` 验证两条内层边同时变成 29/61。
- 改完必须跑满 140 例 tess 簇回归（当前基线 **134 pass / 5 fail / 1 ns**），
  尤其确认 `inner_tessellation_level_rounding` 不崩、点模式无重复顶点。

---

## 第三轮（2026-09-12 更晚）：两轴目标几何已完全确定

用 `cts_instrument.py` 取到**内层两条边的完整点列范围**，现在目标几何是**精确可复现**的：

| 边 | 点数 | 段数 | 步长 | 跨度 | 下界 | 上界 |
|---|---|---|---|---|---|---|
| 内层水平 (u) | **30** | **29** | 1/32 | 29/32 | 1/32 | 31/32 |
| 内层垂直 (v) | **62** | **61** | 1/63 | 61/63 | 1/63 | 62/63 |
| 外缘 (u,v) | 34 / 33 | 33 / 31 | — | — | — | — |

**外缘已经正确**（`2 短段 1/64 + 31 长段 1/32`，满足 2-delta 分支），**不要动**。

### 为什么之前 6 种改法都不行

内层网格当前调用 `edge_position(inner_clamped, nx, x)`，其步长由
`1/inner_clamped` 决定：u 轴给 **1/32**（因为 `inner_clamped[0]=32`，是"短段因子"），
v 轴给 **1/63**。但目标是 **u 用 1/32、v 用 1/63** —— 步长其实**已经对了**，
错的只是**点数**：u 需要 30 点（现在 31/32 点），v 需要 62 点。

一旦把点数改对，`edge_position` 的短段公式 `(f-(n-2))/(2f)` 也会跟着变，
于是**每改一次点数，步长就被一起改掉**（实测：点数从 32→31 时步长从 1/64+1/32 变成 1/31）。
这就是"改点数→段数乱跳（30/28/59/60）"的根因 ——
**`edge_position` 把"段数"和"端点"耦合在同一个参数里**。

### 下一轮的正确做法（已定位到具体函数）

把内层网格的定位从 `edge_position` 解耦成"给定点数 n 与区间 [lo, hi] 后
**等分**"：`u = lo + (hi-lo)*col/(n-1)`，其中

- u 轴：`n = 30`、`lo = 1/32`、`hi = 31/32`（步长自然为 `(31/32-1/32)/29 = 1/32`）
- v 轴：`n = 62`、`lo = 1/63`、`hi = 62/63`（步长自然为 `1/63`）

即**用等分而不是 `edge_position`**，这样点数和步长就不再互相牵制。
外缘继续用 `edge_position`（它已经对），只把内层网格换掉。

本轮实测过的候选（全部回滚）：`u_seg` 取 `nx-2/-3/-4`、
`v_seg` 取 `ny-1/-2/-3`、`u_lo=0.5/(seg+1)`、`u_lo=0.5/(seg+2)`、
`(hi-lo)/(n-1)` 等分、纯 `1/clamped` 步长 ——
分别得到 `found: 30 / 28 / 59 / 60`、"11~12 个 distinct delta"、重复顶点或崩溃。
**注意**：`(hi-lo)/(n-1)` 那一版之所以还是 30，是因为当时 `u_seg` 仍是 30（n=31），
按上表把 `n` 固定为 30 才是关键。

### 关键发现：两个轴通过"内层矩形角点"耦合，不能独立调

本轮实测（每次都只改一个轴、用改装 CTS 读 `found`）：

| 改动 | u 轴结果 | v 轴结果 |
|---|---|---|
| 原状 | 31（要 29） | **62（对，要 61 的"found 31/62"读法见下）** |
| 只把 u 点数改成 30（`u_pts=nx-3`，步长保持 1/32） | 29 附近 | **掉到 59** |

也就是说**缩窄 u 轴会同时改变 v 轴测到的段数** —— 因为
`getEdgesForQuadsTessellation` 的角点是"离 (0.5,0.5) 最远的点"，它由
**两轴的边界共同决定**；改动任一轴的边界位置，两个方向的"最远点"都会移动。

因此"分别把 u 调成 29、v 调成 61"这种做法在本模型下**不成立**：
必须在**一次**改动里同时给出两轴的精确坐标，使得

- 角点恰好落在 `(u_lo, v_lo)`、`(u_hi, v_lo)`、`(u_lo, v_hi)`、`(u_hi, v_hi)`
- 四条内层边分别是 `u` 方向 29 段、`v` 方向 61 段

本轮已确定的**目标坐标**：

| 边 | 段数 | 步长 | 下界 | 上界 |
|---|---|---|---|---|
| u 内层 | 29 | 1/32 | 1/32 | 31/32 |
| v 内层 | 61 | 1/63 | 1/63 | 62/63 |

u 轴的 `[1/32, 31/32]` 总宽 `30/32`，被 29 段 1/32 分成 **29 段 + 两端各 1/64 的余量**；
v 轴的 `[1/63, 62/63]` 总宽 `61/63`，被 61 段 1/63 精确分完。
**两轴的余量结构不同**（u 有 1/64 余量、v 没有），这正是"同一公式套两轴"一直失败的原因，
也说明 u 轴的几何应当由 **该轴自己的 short-segment 项**（`1/clamped[0] = 1/32` 的一半 = 1/64）
决定，而 v 轴的 `clamped[1] = 63` 是奇数、短段项不落到这些整数位置上。

**下一轮的正确切入点**（已推导到公式级）：不要按"点数"试，直接按
`edge_position` 的 short-segment 项构造两轴各自的余量：

```
margin = (f - (k-2)) / (2f)      // f = inner_clamped[axis], k = inner_eff[axis]
lo     = margin,  hi = 1 - margin
span   = 1 - 2*margin = (k-2)/f  // 恰好是 k-2 段 1/f 的宽度
```

代入本轮两个轴：

| 轴 | f | k | margin | span | 段数 |
|---|---|---|---|---|---|
| u | 32 | 33 | 1/32 | 30/32 | **29** ✅ |
| v | 63 | 63 | 1/126 | 61/63 | **61** ✅ |

**两轴的 margin 公式相同，但数值差 4 倍**（1/32 vs 1/126）——
这正是"套同一个余量常数"反复失败的原因，也是本轮的最终结论。
按此式构造后两轴都应在**一次**改动里同时命中 29 / 61；
**必须同时改两轴**，因为角点由两轴边界共同决定（见上表：只改 u 会让 v 掉到 59）。

### 本轮最终确定的两轴坐标（`inner=(32,64) outer=(1,32,1,32)` FO）

```
u 轴（内层水平边要 29 段）：30 个点，1/32 步长，u = 1/32 + k/32,  k = 0..29   -> [1/32, 31/32]
v 轴（内层垂直边要 61 段）：62 个点，1/63 步长，v = 1/63 + k/63,  k = 0..61   -> [1/63, 62/63]
```

- 两轴的**下界都等于自己的步长**（`1/32` 与 `1/63`），即 `lo = step`。
  这一条在 u 轴给出 `31/32 - 1/32 = 30/32`，被 `29` 段 `1/32` 分成
  **29 段 + 两端各 `1/64` 余量**；v 轴给出 `62/63 - 1/63 = 61/63`，
  被 `61` 段 `1/63` **精确分完**（无余量）。
- **两轴的步长不同源**：u 的 `1/32 = 1/clamped[0]`，v 的 `1/63` 不是 `1/clamped[1]`
  （`clamped[1]=63` → `1/63` 恰好相同，但 u 的 `clamped[0]=32` → `1/32` 也相同），
  即**步长 = `1/clamped[axis]`**，而下界 = 步长。本轮最后离目标只差 1
  （`found: 30` vs 期望 29），说明 `u_pts` 还需再减 1（`29` 点而非 `30`），
  但当时 `u_seg` 的推导尚未同步，故未继续。

**已实测的最接近配置**（可作下一轮起点）：
`u_seg = nx-3`、`v_seg = ny-2`、`step = 1/clamped`、`lo = step`、等分 →
得到 `found: 30`（差 1）；把 `u_pts` 再减 1 即命中 29。
**注意必须一次改两轴**：只改 u 会让 v 的 `found` 从 62 掉到 59（角点位移）。

---

## 第四轮（2026-09-12）：模型已能预测 u 轴，v 轴仍有 2 段缺口

### 已建立可离线预测的模型

把内层网格的构造写成 `margin = (f - (n-1)) / (2f)`、两轴各自等分，
`lo = margin`、`hi = 1 - margin`、点数 `n = k - bias`。用这个模型离线枚举
（`/tmp/mgl_verify/offline_mesh.py`）后与改装 CTS 的实测**逐项对齐**：

| u_bias | 模型预测 u 段数 | CTS 实测 |
|---|---|---|
| 2 | 30 | 30 ✅ |
| 3 | **29** | **29** ✅ |
| 4 | 28 | 28 ✅ |
| 5 | 27 | 27 ✅ |

**u 轴已完全命中**（`u_bias = 3`，即 `u_pts = nx - 3 = 30`）：
CTS 实测 `n_edge=4 npts=30 min=0.046875 max=0.953125 span=0.90625` = **29 段 1/32**。

### v 轴的缺口

同一次运行里 CTS 实测 `n_edge=5 npts=62 min=0.015873 max=0.984127 span=0.968254`
—— 几何上就是 **61 段 1/63**，与目标一致。**但测试报的是
`Invalid amount of segments (expected:61, found: 59)`**。

即：**几何对了（62 点 / 61 段），CTS 却仍报 59**。差 2 而不是 1，
且 `verifyEdges` 在报错时用的是**未修改的 CTS 源码**（我的桩只加打印、不改判定）。
这说明 v 轴还有第二个来源在影响：**`n_edge=5` 的角点不是内层网格的边界行**，
而是外缘或 1+ε 生成的顶点插进了该边的收集集合（收集规则是"在线上 + 距 start 更近"，
`1-s/2` 这类点位可能被一起收进）。

**下一轮的落地点**（已缩小到很具体）：
1. 先只改 v 轴（`v_bias = 1`，`v_pts = ny - 1 = 62`），保持 u_bias=3，
   用桩打印 `n_edge=5` 的**全部点**（不只 min/max），确认 59 是怎么数出来的；
2. 重点检查 `1/special` 顶点：`inner[1]=64` 时 `inner_clamped[1]=63`，
   但外缘 `outer[2]=1` 使 `1+ε` 路径可能产生额外顶点落在这条线上；
3. 定位后再决定是否需要在 v 轴单独扣除 2 个点（要给出规格依据，不能纯凑数）。

**当前基线**：140 例 tess 簇 **134 pass / 5 fail / 1 not_supported**，
工作树干净（实验全部回滚或存于 `git stash`）。

---

## 第五轮（2026-09-12）：拿到了 `n_edge=4` 的完整点列与 delta —— 结论出乎意料

按上一轮的落地点，用桩把 `n_edge=4` 的**全部点 + 全部 delta 直方图**打了出来
（`inner=(32,64) outer=(1,32,1,32)`，`u_bias=3`）：

```
MGLCTSE4 npts=29 ndelta=1 rounded=31 :
  0.0625 0.09375 0.125 0.15625 ... 0.90625 0.9375
MGLCTSE4 delta 0.03125 x28
```

即 **29 个点、28 个等长段、步长 1/32**，而 `verifyEdges` 的 1-delta 分支期望
`clamped_rounded - 2*(4/4) = 31 - 2 = 29`。

**所以 28 ≠ 29 —— 我又少了 1 段。** 而**未改代码前**同一条边是
`31 个点 / 31 段`（期望 29）。

关键推论：**这份日志里的 `n_edge=4` 不是报错的那条边。**
测试跑的 48 个配置里，报 `expected:29, found:31` 的那条边是**另一个配置**的
（很可能是 `outer` 全 32 或 `outer` 含 1 的变体），它的点数结构与
`outer=(1,32,1,32)` 不同 —— 这解释了为什么"照着这条边的几何去改"
（连续 4 轮、十余种改法）始终不能让整条用例通过。

**因此下一轮必须先做这一步**：把桩改成**打印所有 48 个配置的
`inner/outer/n_edge/npts/ndelta/rounded/exp`**（不只 `n_edge==4` 且不只
`inner==(32,64)`），先定位**到底哪一个配置的哪条边**在报 29/31，
再针对那个点位的真实结构改。**在那之前不要再改 `generate_quads`。**

### 本轮实测的两种网格在完整用例上的表现

| 网格 | `n_edge=4` 实测 | 140 例 tess 簇 |
|---|---|---|
| 现状（基线） | 31 点 / 31 段 | **134 pass / 5 fail** |
| `u_bias=3`（内层等分、u 取 `nx-3`） | 29 点 / 28 段 | 134 pass / 5 fail（**同样 5 个失败**） |

两种网格的**失败集合完全相同**，即这个改动是"净中性"的：它让
`n_edge=4` 的几何更接近规格，但对决定用例成败的那条边毫无影响 ——
进一步佐证"报错边不是这条"。

### 当前状态

- 工作树干净，只留已验证的 `9a1dc5b`；140 例 tess 簇 **134 pass / 5 fail / 1 ns**。
- CTS 源码树已还原干净；实验代码在 `git stash@{0}`。
- 复现工具：`/tmp/mgl_verify/cts_instrument.py`（CTS 加桩/还原）、
  `offline_mesh.py`（离线参数枚举）。

---

## 第六轮（2026-09-12）：全配置定位成功，靶子已完全确定

按上一轮的计划，把桩改成**全配置判定打印**（`MGL_CTS_VERDICT`，
对全部 48 个配置的每条边算出 found/exp 并只打印不符的），一次跑完拿到结果：

```
48 条失败边，全部是同一形态：
  edge=4  rounded=31  npts=32  ndelta=1  found=31  exp=29   (24 次, inner=(32,64) 全 24 种 outer)
  edge=5  rounded=31  npts=32  ndelta=1  found=31  exp=29   (24 次, inner=(64,32))
```

**结论（重要，纠正了前几轮的判断）**：
- 失败边**与 `outer` 无关**（24 种 outer 组合全挂），所以**不是**"盯错了边"，
  而是这条边本身就是错的 —— 上一轮"报错边是另一个配置"的推断被推翻。
- `exp` 恒为 `clamped_rounded - 2`，与规格一致。
- **`ndelta=1`**：内层边是**纯等长段**，没有短段 ——
  所以 `verifyEdges` 走 1-delta 分支，要求 `counter == rounded-2`。
  **内层边必须是 `rounded-2` 个等长段，步长 1/clamped。**

### 已把参数空间扫到只剩 1 个刻度

用同一套桩做参数扫描，每档都读 48 条边的 `npts/found/exp`：

| 配置 | u 轴 (edge=4) | v 轴 |
|---|---|---|
| 基线 | npts=32 found=31 exp=29 | npts=32 found=31 exp=29 |
| `u_seg=nx-3` | npts=31 found=30 exp=29 | — |
| `u_seg=nx-2` | npts=32 found=31 exp=29 | — |

**规律是 `found = npts - 1`**，而目标是 u 要 29、v 要 61，即
**u 需要 30 点、v 需要 62 点**。

### 本轮卡住的具体位置

设 `u_seg = nx - 3`（`nx = inner_clamped[0] = 32`）→ 30 点、跨度 `29/32`、步长 `1/32`，
**几何上正好是 29 段**（`first=(0.03125,·) last=(0.96875,·)`，与 29 点跨度吻合）。
但 CTS 实测 **`npts=31, found=30`** ——
**比我生成的点多 1**，即**外缘仍有一个点落在这条线上**被一起收进边集合
（`x=1/32` 或 `x=31/32` 处外缘与内层网格重合，而点模式下顶点可能重复计入）。

**下一轮第一步**（很具体）：把该边的**全部点坐标**打出来（本轮已加过
`MGLCTSPTS` 桩，直接复用），对照上面 `first/last` 找出那个多出来的点是谁、
来自外缘还是内层网格；确认后要么让内层网格避开该坐标，要么在生成 `1+ε`
顶点时去重 —— **必须给出规格依据，不能为了凑 29 而硬删点**。

**本轮实测汇总**：无论怎么调，140 例 tess 簇始终是 **134 pass / 5 fail**
（与基线同），即这些内层网格改动都是**净中性**、不产生回归。

---

## 第七轮（2026-09-12）：`fractional_even` 已全绿，`n==1` 退化是最后的卡点

用全配置判定桩把三个 spacing 模式分别扫了一遍，本轮**把失败边从 48 条压到 2 条**，
并且 **`fractional_even` 达到 0 条失败边**。演变过程（每次都读全部边）：

| 网格配置 | fractional_odd | equal | fractional_even |
|---|---|---|---|
| 基线 | 48 条失败边 | — | — |
| `u_seg=nx-4, v_seg=ny-4, lo=step` | **2** | 2 | **0** ✅ |
| `u_seg=nx-2, v_seg=ny-2, lo=step` | 2 | 2 | **0** ✅ |

**最终确定的内层网格规则**（已由实测收敛）：

```
step  = 1 / inner_clamped[axis]
seg   = inner_eff[axis] - 2          // = clamped_rounded - 2，规格的等长段数
pts   = seg + 1
lo    = step                          // 两端各留一个整步
u/v   = lo + i * step,   i = 0..seg
```

这条规则让 **fractional_even 三个用例全部通过**，并使 fractional_odd / equal
只剩 **2 条**失败边，全部集中在**同一个退化形态**：

```
n_edge=1  tess=1  rounded=1  npts=64  found=63 (FO) / 32 (equal)  exp=1
```

即 **`rounded == 1` 的退化边**：规格原文 "If `<n>` is one, the edge will not be
subdivided"，`verifyEdges` 对此设 `expected_counter = 1`。
现在这条边上被收集到 **64 个点**（来自外缘 + 内层网格），而它应当只有 **2 个点**。

**下一轮就做这一件事**：让 `rounded == 1` 的轴不要生成内层网格线
（`u_pts = 1` 只留中心那一列/行），使该退化边只剩两个端点。
本轮试过 `u_pts = nx > 1 ? seg+1 : 1` 但未生效（`npts` 仍是 64），
需检查 `nx`/`ny` 在 `rounded==1` 时的实际取值（很可能是 1，而 `u_seg` 走了 `>2u` 分支），
**把 `u_seg` 也一并归零**（`u_seg = 0` → `u_pts = 1`）再测。

**注意**：这条规则同时改了两轴，`equal_spacing` 的 2 条失败边也是同一退化形态，
所以修好退化就能同时收敛两个模式。

**本轮所有网格改动在 140 例 tess 簇上始终是 134 pass / 5 fail（净中性、零回归）**，
即这条路是安全的，可以放心在这个方向上继续收口。

---

## 第八轮（2026-09-12）：`equal_spacing` 也清零了，但退化修复会带回归

在上一轮"只剩 `rounded==1` 退化边"的基础上，本轮把网格改成**退化感知**版本
（`u_pts = nx>2 ? nx-1 : 1`、`u_seg = u_pts-1`，单点时放在 0.5）：

| spacing 模式 | 失败边数 | 说明 |
|---|---|---|
| `equal_spacing` | **0** ✅ | 本轮从 2 → 0 |
| `fractional_even_spacing` | **0** ✅ | 保持 |
| `fractional_odd_spacing` | 2 | 仍是 `n_edge=1 tess=1 rounded=1 npts=64 found=63 exp=1` |

**但 140 例 tess 簇整体退了**：

| 网格 | 140 例结果 |
|---|---|
| 基线（HEAD） | **134 pass / 5 fail** |
| 退化感知版 | **133 pass / 6 fail** ← `point_mode.points_verification` 新增失败 |

所以这个版本**不能提交**（净负），已回滚。**当前仓库保持在已验证的 134/5**。

### 关键教训（写下来避免重复）

- `point_mode.points_verification` 对**内层网格顶点数**敏感 ——
  改动"是否存在内层线 / 单点退化位置"会直接改变它期望的点集。
  本轮把单点放在 `0.5`、以及把 `u_pts` 在 `nx<=2` 时压成 1，都触发了它。
- 因此**退化处理必须与 `point_mode` 的顶点并集语义一起考虑**：
  `ARB_tessellation_shader` 明确写了
  "the tessellation primitive generator may produce 'interior' vertices that are
  positioned on the edge of the patch if an inner tessellation level is less than
  or equal to one. Such vertices are considered **distinct** from vertices produced
  by subdividing the outer edge of the patch, even if there are pairs of vertices
  with **identical coordinates**."
  → 1+ε 产生的"落在 patch 边上的内层顶点"是**允许与端点重合**的，
  不应被去重或压成单点；这正是本轮改动踩中的地方。

### 下一轮的正确做法

1. 保留已验证的 `seg = nx-2 / lo = step` 规则（它已让 **equal 与 fractional_even 全绿**）；
2. 退化（`rounded==1`）时**不要**把内层压成单点，而是按规格原文生成
   "落在 patch 边上的内层顶点"，并**保留其与端点的重复**；
3. 改完必须同时看两件事：`vertex_spacing` 三个模式 + `points_verification`，
   任一退化即回滚（本轮就是这么发现 133/6 的）。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、
140 例 **134 pass / 5 fail / 1 not_supported**；CTS 源码树干净。

---

## 第九轮（2026-09-12）：退化的"跳过内层线"方案被证伪

上一轮的退化感知版（单点放 0.5）净负（133/6）。本轮换一种退化策略：
`u_seg = nx>2 ? nx-2 : 0`、`u_pts = u_seg+1`，并在**两轴都退化时**把点放到中心
（`u = v = 0.5`），非退化轴照常。

结果**更差**：

| 网格 | 140 例 tess 簇 |
|---|---|
| 基线（HEAD） | **134 pass / 5 fail** ✅ |
| 跳过内层线版 | **129 pass / 10 fail** ❌ |

新增失败包括 `points_verification`、`tex_ordering`，以及
`vertex_spacing` 的 **equal / fractional_even 也一起挂了** ——
即这个退化分支会被**非退化配置**命中，把原本已经全绿的
`equal_spacing` / `fractional_even_spacing` 一起弄坏。

**已回滚**，仓库仍为已验证的 **134/5**。

### 结论：退化（`rounded==1`）不能用"压成单点/跳过线"处理

两轮两种退化策略（单点放中心、跳过内层线）都是净负，说明问题不在"怎么摆那个单点"，
而在**内层网格的存在与否会改变其它用例的顶点集合**。规格原文其实给了正确方向：

> "…may produce 'interior' vertices that are positioned on the edge of the patch if an
> inner tessellation level is less than or equal to one. Such vertices are considered
> **distinct** from vertices produced by subdividing the outer edge of the patch, even if
> there are pairs of vertices with **identical coordinates**."

→ 1+ε 情况下**应当照常生成内层顶点（允许与端点坐标重合），而不是把它们去掉或压成一个点**。
下一轮应该走"**保留内层顶点、但让该轴的段数/位置服从 `<n>=1` 不细分**"这条路，
即 `u_seg = 0` 时**仍然生成 `u_pts = 1` 的那条线上应有的顶点**（依规格是 1+ε 的
"on the edge" 顶点），而不是把整条网格塌掉。

### 已确定的可用规则（务必保留）

```
step = 1 / inner_clamped[axis]
seg  = inner_eff[axis] - 2          // = clamped_rounded - 2
pts  = seg + 1
lo   = step                          // 两端各留一个整步
pos  = lo + i*step,  i = 0..seg
```

这条规则下：**`equal_spacing` 与 `fractional_even_spacing` 三个用例全绿**，
`fractional_odd` 只剩 2 条 `rounded==1` 的退化边。相关版本存于 `git stash@{0}`。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**、
CTS 源码树干净。

---

## 第十轮（2026-09-12）：转向 #5/#6 —— 给 `point_mode.point_rendering` 找到一条明确线索

内层网格方向已连续两轮净负，本轮改为推进**尚未探查的 #5**，做了代码级追踪。

### #5 `point_mode.point_rendering`（渲染全 0）的调用链追踪

用例失败表现是 `expected (0.1,0.2,0.3,0.4) rendered (0,0,0,0)`，即**点没被画出来**上色。
追踪 TES 渲染链路（`exec=MGL_TESS_EXEC_TES_COMPUTE`）：

1. **域播种只写 position，不写 point size**
   `mglTessSeedEvalOutputRecords` → `mglRenderSeedTessDomain`
   → 只调用 `mglTessGenerateDomainStrided`，把 `u,v,w` 写进记录（position@0），
   **point_size@16 保持 0**（缓冲区是 `memset` 过的）。
2. **TES 计算核会写 point size**
   `mgl_air_backend.cpp:11737`：
   `gl_PointSize` 有写就用它，否则**默认 1.0**，然后 `storeGeometryPointSize(record, 0, pointSize)`
   写到记录的 `MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET`（=16）。
3. **passthrough 顶点函数按 vec4 步长读 point size**
   `MGLRenderer+RenderPass.m:1226-1234`：
   ```
   int mgl_base = gl_VertexID * vec4Stride;           // vec4Stride = out_stride/16
   gl_Position = mgl_tes_output.records[mgl_base];
   vec4 mgl_point_size = mgl_tes_output.records[mgl_base + 1];
   gl_PointSize = mgl_point_size.x;
   ```
   `records[mgl_base + 1]` 对应的字节偏移是 `out_stride + 16`，
   而 point size 写在**记录的字节 16**（即 `gl_PointSize` 槽）。

**待验证的怀疑点**：`out_stride`（`mgl_draw_tess.cpp:511` 由
`mglAIRPerVertexStrideForResources(TES stage-out)` 得出，下限 `MGL_AIR_PER_VERTEX_STRIDE=112`）
与记录里 point size 实际写入位置是否一致 —— 若 `out_stride` 不是 112
（例如该 TES 的输出 varying 撑大了 stride），则 `mgl_base + 1` 会指向错误槽位，
读到的不是 point size。这条链路里 **`storeGeometryPointSize` 的槽位
（固定按 112 步长计算）与 passthrough 的 vec4 索引（按 `out_stride/16` 计算）
是两套算法**，是最可疑的不一致处。

**下一轮验证方法**：在 passthrough 的 Metal 源码里把 `mgl_point_size.x` 直接写进
`gl_Position`（或把 `out_stride` 打进 `gl_Position.z`）用作可视化探针，
一次运行即可看出读到的到底是 point size 还是别的数据；
也可以在 `mgl_air_backend.cpp` 里确认 `storeGeometryPointSize` 用的步长。

### #6 `tc_barriers`（未动）

现象仍是 `patch[3]` 里出现 `1065353216 = 1.0f` 的位模式（应为整数 8/32）。
本轮未取得新证据；怀疑点仍是 TCS 共享内存 lowering 与 `barrier()` 语义
（`mgl_air_backend.cpp:5975` 把 GLSL `barrier()` 映射为
`air.wg.barrier(flags=3 /*device|threadgroup*/, scope=1)`），需单独排期。

### 本轮实测

- 内层网格又试了"退化时保留顶点、仅不细分"的思路，**仍净负**（已回滚，不再重复）。
- 仓库保持已验证状态：**134 pass / 5 fail / 1 ns**，工作树干净，CTS 源码树干净。

---

## 第十一轮（2026-09-12）：#6 `tc_barriers` 的失败数据已完整解码

### 失败的精确形态（来自 QPA 日志，patch[3]）

```
observed  = [1006648320, 1065353216, 0, 1065353216, 32, 0, 0, 0]
expected  = [8, 0, 0, 0, 32, 0, 0, 0]
```

**关键解码**：
- `1065353216 = 0x3F800000 = 1.0f`
- `1006648320 = 0x3C000000 ≈ 0.0078125`（**浮点位模式**，不是整数 8）

即 **float 位模式被当作 int 读了出来**，而且 `1.0f` 出现在**两个**位置
（index 1 也与 expected 0 不符）。这是 **TES 的 XFB 输出「float 载体 vs int 属性」
错位**的典型特征 —— 不是 TCS 算错。

### 已排除的方向

- **不是 `barrier()` lowering**：用 `MGL_DUMP_IR=1` 导出 IR 后确认，
  两个 `barrier()` 都正确变成 `call void @air.wg.barrier(i32 3, i32 1)`
  （flags=3 device|threadgroup，scope=1 threadgroup），且 TCS 的 patch 输出
  读取走的正是"从 patch-out 缓冲重载"的路径
  （`mgl_air_backend.cpp:4738`，注释明确写了要在 barrier 后重载以免被 SSA 缓存掩盖）。
- **不是 point size**（#5）：本轮给 passthrough 顶点函数加了
  "point size ≤ 0 时回退 1.0"（GL 4.6 §13.6.1 的默认值），
  `point_mode.point_rendering` **仍然失败**（140 例仍是 134/5，无回归）。
  该改动已回滚。

### 下一步（很具体的探针）

`patch 0/1/2` 正确、`patch 3` 起错误 → 怀疑 **patch 索引 / patch-out 缓冲偏移**
在高 patch 号下错位（`patchPos` × `patchOutStride`，或 XFB 记录里的 patch 步进）。
探针建议：
1. 在 `emitPatchVaryingStore`（`mgl_air_backend.cpp:2210`）里把
   `patch` 与 `off` 打到 stderr，跑该用例看 patch 3 的写入偏移是否连续正确；
2. 或把 TCS 写出的 `tcs_patch_result[0]` 直接写进 `gl_TessLevelOuter[0]` 之类
   可读出的通道，用 XFB 抓一次即可判断是"写入错位"还是"读出错位"。

### 当前状态

工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**；CTS 源码树干净。

---

## 第十二轮（2026-09-12）：#6 又排除一条假设，并把范围逼到"缓冲/步长错位"

### 已用探针确认的事实

1. **TCS 完全正确**。给 `mglTessPackXFBFieldFromCarrier` 加入口探针后，
   从 XFB 载体里读到的原始字节是**干净的整数值**：
   ```
   MGLXFBPACK gl_type=0x8b55 bytes=16 in: 00 00 00 41 00 00 00 00 ...   (= 8.0f 载体)
   MGLXFBPACK ... in: 00 00 00 42 ...   (= 32)
   MGLXFBPACK ... in: 00 00 60 42 ...   (= 56)
   MGLXFBPACK ... in: 00 00 a0 42 ...   (= 80)
   ```
   `8 / 32 / 56 / 80` 与 CTS 在 `verifyXFBBuffer` 里用 C++ 端口算出的期望值**完全一致**。
   所以 TCS 的 `tcs_patch_result[]`、`barrier()` 语义、patch 输出缓冲**都没问题**。
2. **packer 本身也正确**。在同一函数出口加探针，写入目标缓冲的字节是
   `08 00 00 00 ... / 20 00 00 00 ... / 38 00 00 00 ... / 50 00 00 00 ...`
   —— 正是整数 `8 / 32 / 56 / 80`，且 `v=0`、`v=1` 两个 isoline 顶点都正确。
3. **我按"位模式"猜测做的修复是错的**。曾把
   `mglTessPackXFBFieldFromCarrier` 的 `(int32_t)f` 值转换改成位拷贝，
   结果 `barrier_guarded_read_calls` / `write_calls` **新增失败**（3→2 pass），
   已回滚。该函数当前行为是对的，**不要再动它**。

### 结论：问题在"数据送达测试读取的那块缓冲"这一环

前两个顶点（patch 0 / 1）测试读到的是正确的 `8`，而 patch 3 读到
`[1006648320, 1065353216, 0, 1065353216, 32, 0, 0, 0]`。
packer 的出口字节是对的，说明**中间某处没有把正确的数据写进测试最终 map 到的那块缓冲**
（`mglRendererBufferSubData` → CPU shadow / live Metal allocation 的镜像，
见 `MGLRenderer+Tessellation.m:2020-2040`），或者**捕获的顶点数 / 目的偏移**
与测试的 stride 假设不一致。

**下一轮第一步（已缩到很具体）**：
在 `MGLRenderer+Tessellation.m` 的 interleaved 分支里，
把 `xfbWrittenBytes / xfbCopiedVertices / xfbCompactStride / xfbDestination`
以及 `mglRendererBufferSubData` 之后**从 `destBuf->data.buffer_data` 回读的头 32 字节**
打成 stderr，与上面 packer 出口的 `08 00 00 00 …` 对拍。
只要回读不再是 `08 00 00 00`，就能确定是"SubData / shadow 镜像"这条链路丢了数据。

### 当前状态

工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**；CTS 源码树干净。

---

## 第十三轮（2026-09-12）：#6 定位到"XFB 拷贝只覆盖 360/6400 顶点"

### 探针实测（`mglTessPlanXFBDestination`）

```
MGLXFBPLAN ipi=36 inst=10 cstride=64 vpp=1 session_off=0 slot_off=0
            visible=204800 capture_verts=360 prim_bytes=64
            capture_prims=360 copied_prims=360 copied_verts=360 written=23040
```

解读：
- `visible=204800` 与测试分配的 XFB 缓冲大小**完全一致**（`10 * 640 * 4 * 4 * 2`），
  说明 GL 侧缓冲大小没问题。
- **`ipi=36`（每实例 36 个 TES 输出项）** —— 该配置是 `layout(isolines, point_mode)`，
  point_mode 下 TES 每个 patch 只产出**去重后的点**，所以 2 patch × 18 项 = 36；
  于是 `copied_verts=360`、`written=23040`。
- 而测试按 **isoline 语义**假定每实例 640 个顶点（`2 点/段 × 2 patch × 16 invocation`），
  需要 6400 个顶点、204800 字节。

**所以缓冲区里只有前 360 个顶点（23040 字节）被填充，其余 88% 从未写入。**
测试读 patch 3 时取的是 `data_int[3 * 8 + n]`（拼写上属于 `patch_data_int` 的前 8 项），
这部分本应在已填充区内 —— 但仍读到 float 位模式，说明**写入区之外/边界处还有一层
对齐或偏移问题**（见下）。

### 已验证为正确的环节（不要再查）

- TCS：载体里就是 `8 / 32 / 56 / 80`，与 CTS 的 C++ 端口期望值一致。
- `mglTessPackXFBFieldFromCarrier`：出口字节 `08 00 00 00 / 20 00 …`（整数），
  且 v=0 / v=24 / v=48 三处都正确。
- 试图把该函数的 `(int32_t)f` 改成位拷贝 → **新增 2 个失败**，已回滚，**不要动**。

### 结论与下一步

`copied_verts=360` 远小于测试需要的 6400，是**确定的不一致**。
下一轮第一步：确认 `points_verification` 与 `point_rendering` 是否也依赖同一套
"point_mode 下 items 计数"逻辑（它们分别**通过**和**失败**，可作为对照）；
然后在 `mglTessPlanXFBDestination` 的 `visible_bytes` 与
`capture_primitives` 之间加对拍，判断应不应该按 `instance_count` 摊开
（现在 `instance_count=10` 只用于 `capture_verts` 的乘法，**没有为每个实例各写一段**，
`written=23040` 与 `instance_bytes` 的关系需要确认）。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十四轮（2026-09-12）：#6 锁定到"CPU 拷贝与测试读取的不是同一份内容"

### 关键探针（在 `mglRendererBufferSubData` 之后立刻回读目标缓冲）

```
MGLXFBCHK v24: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

即：**packer 出口是 `08 00 00 00 …`（正确的 8），但 CPU 拷贝完成、从
`mglTessBufferContents(xfbCopyDestination)` 回读时 v=24 处是 0**。
而测试最终读到的是 **float 位模式** `1006648320 / 1065353216`。

**三者互不相同** → 结论：**CPU 侧这条 packed 拷贝并没有成为测试读取的那份数据**，
测试读到的是另一条路径（GPU 写入 + 影子/镜像）产出的内容。

### 已完整排除的环节（都不必再查）

| 环节 | 证据 |
|---|---|
| TCS 计算 | 载体为 `8/32/56/80`，与 CTS 的 C++ 端口期望一致 |
| `barrier()` lowering | IR 里是 `air.wg.barrier(i32 3, i32 1)`，TCS patch 输出走 barrier 后重载路径 |
| `mglTessPackXFBFieldFromCarrier` | 出口字节 `08 00 00 00 / 20 00 …`（整数正确），v=0/24/48 均正确 |
| packer 的 `(int32_t)f` | 改成位拷贝会**新增 2 个失败**，已回滚，**不要动** |
| GL 侧缓冲大小 | `visible=204800` 与测试分配一致 |
| items 计数 | `ipi=36` 对 `(isolines, point_mode)` 是正确的（2 patch × 18 点） |

### 下一步（唯一剩下的分叉）

现在只剩下一条分叉需要判定：**测试读到的内容到底由哪条路径产生** ——
1. GPU 端 XFB 写入（`MGL_AIR_TESS_SLOT_XFB_OUT` / scatter）到 `xfbDestination`；
2. 还是 CPU 影子（`destBuf->data.buffer_data`）与 live Metal allocation
   （`mglTessBufferContents(xfbCopyDestination)`）之间的镜像。

**下一步做法**：在 `mglRendererBufferSubData` **之前**先回读一次
`xfbDestination` 与 `xfbCopyDestination` 的 v=24，跑完 draw 后再回读一次，
比较三个时间点的字节；同时确认 `mglRendererBufferSubData` 是否真的写进了
`xfbCopyDestination`（刚刚的探针显示**没有**，这一步最可疑）。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十五轮（2026-09-12）：#6 的机制已查明 —— SubData 写进了 snapshot，原缓冲不变

### 决定性探针（SubData 前后各回读一次同一块缓冲）

```
MGLXFB-PRE  v24: 00 00 00 00 ...  dstBuf=hasCPUShadow  destOffset=0 written=23040
MGLXFB-POST v24: 00 00 00 00 ...
```

**SubData 前后完全相同（都是 0）** —— 即 `mglRendererBufferSubData` 没有改动
`xfbCopyDestination` 这块缓冲。

### 代码级原因（`mgl_render.cpp:2156-2229`）

`mglRenderBufferSubDataStorage` 在"CPU 影子与 Metal base 不同"时走这条分支：

```c
if (cpuBase && cpuBase != metalBase) {
    memmove(cpuBase + offset, bytes, size);          // 1) 写 CPU 影子（正确）
    mglRenderSnapshotSharedDirtyBuffer(buffer, &snapshotBuffer, ...);  // 2) 建 snapshot
    metalBuffer = snapshotBuffer; metalBase = snapshotBuffer->contents();  // 3) 改指 snapshot
    ...
}
if (metalBuffer == bufferBeforeSnapshot) {           // 4) 已不等 → 跳过
    memcpy(metalBase + offset, bytes, size);
}
```

即：**packed 数据被写进 CPU 影子 + 一块新的 snapshot，而原本的
`buffer->data.mtl_data` 保持全 0**。所以：
- 我的探针（走 `mglTessBufferContents` → `mglRenderGetBufferContents`）
  读到 0 是**符合这条代码路径的**，不一定是 bug；
- 但这解释了一个确凿的现象：**CPU 侧 packed 数据并没有落到"原 Metal 缓冲"**，
  而测试通过 `glMapBufferRange` 读的是**影子**，所以两边本应对上 —— 却仍然对不上。

### 因此 #6 的剩余分叉（比上一轮更窄）

测试读到的 `1006648320 / 1065353216`（float 位模式）**既不是 packer 出口的
`08 00 00 00`，也不是影子里的值**。剩下两种可能：
1. 测试 map 到的影子不是这条 SubData 写的那份（两条不同的 shadow）；
2. GPU 端（XFB 写入 / snapshot 绑定到 draw）在 SubData 之后又覆盖了该区域。

**下一步做法**：在测试 map 之后、`verifyXFBBuffer` 之前，把
`xfb_data[24*64 .. +16]` 直接打出来（用例是 CTS 源码，可加桩），
再与 MGL 侧"影子里的值"对拍 —— 一次即可判定是 1 还是 2。
（注意：不要再改 SubData 的 snapshot 分支，它是**共享缓冲写回**的既有机制，
`barrier_guarded_read_calls` 依赖它。）

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十六轮（2026-09-12）：#6 定位到"XFB 数据没有按实例摊开"这一层

### 关键解码：测试读到的就是 TCS 的 tessellation level

把 patch[3] 的实测值按 float 位模式解码：

```
observed ints = [1006648320, 1065353216, 0, 1065353216, 32, 0, 0, 0]
as float bits = [~0.0078, 1.0, 0.0, 1.0, ~0.0, 0.0, 0.0, 0.0]
expected      = [8, 0, 0, 0, 32, 0, 0, 0]
```

`1.0f` 正好是 TCS 写的 `gl_TessLevelOuter[0]/[1] = 1.0`，
`~0.0078` 也落在 tessellation factor 的量级上。**即测试在 patch[3] 处读到的是
tessellation level 数据，而不是 TES 的 `tes_result1/2`。**

### 已确认的完整链路与唯一剩下的缺口

| 环节 | 实测 | 结论 |
|---|---|---|
| TCS 计算 | 载体 `8/32/56/80` | ✅ 正确 |
| packer | 出口 `08 00 00 00 …` | ✅ 正确 |
| SubData | 写 CPU 影子 + snapshot，原 Metal 缓冲不变 | ✅ 既有机制，勿动 |
| CTS 侧 map 到的缓冲 | `v=24 off=1536` 处是 `08 00 00 00 …` | ✅ 该处正确 |
| **测试比对的 `patch_data_int` 位置** | `data_int + 3*16*2 = byte 48` 处读到 tess level | ❌ **这里不对** |

### 结论（本轮最有价值的一条）

XFB 拷贝只写了 `copied_verts=360`、`written=23040` 字节（offset 0 起），
**而测试期望 10 个实例 × 640 顶点 = 6400 个顶点、204800 字节全部被填充**。
`instance_count=10` 目前只参与 `capture_verts` 的乘法，
**没有为每个实例各写一段**。测试按"每实例 20480 字节"的布局去 patch 3 的位置取数，
取到的自然是那些没被正确填充的区域（残留的 tessellation level 数据）。

**下一轮修法方向（已具体）**：让 XFB 拷贝按实例摊开 ——
在 `mglTessPlanXFBDestination` / `mglTessPlanEvalXfbCapture` 里，
把 `instance_bytes`（每实例 23040）与 `visible_bytes` 的关系理清，
确保每个实例都以 `instance_bytes` 为步长写入自己的那一段
（现在 `written=23040` 只覆盖了实例 0，且 `copied_verts` 的语义是"实例 0 的顶点数"）。
**改完必须同时验证 `points_verification`（当前通过）不被破坏。**

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十七轮（2026-09-12）：#6 定量结论 —— 每实例应写 640 顶点，实际只写 36

把两边的计数摆平后，缺口是**确定且可量化**的：

| 来源 | 每实例顶点数 | 依据 |
|---|---|---|
| **CTS 期望** | **640** | `m_n_result_vertices = 2(isoline 段点数) × 2(patch/invocation) × 16(invocation) × 10(instances) = 640`；缓冲区 = `10 × 640 × 4 × 4 × 2 = 204800` 字节（与实测 `visible=204800` **完全一致**） |
| **MGL 实际** | **36** | `ipi=36`（2 patch × 18 点）→ `copied_verts=360`、`written=23040` |

- 测试按"**每实例 16 个结果顶点**"的布局取数（`patch_data_int = data_int + patch × m_n_invocations × 2 = byte 48`）。
- MGL 每个 patch 只产出 18 个**去重后的点**（`point_mode`），其中出现在该 isoline 上的约 6~7 个，
  远少于测试要读的 16 个 → 读到的就是**未被 XFB 写入的区域**（残留的 tessellation level 数据，
  正好解释了 `1.0f` / tess-factor 量级的位模式）。
- `instance_count=10` 目前只用于 `capture_verts` 的乘法（`36×10=360`），
  缓冲里只有**前 23040 字节**被写；测试期望的是**每实例一整段 204800 字节**（或至少每实例 640 顶点）。

**这正是用例失败的根因**，且与前面各轮"TCS/packer/SubData 都正确"的结论一致。

### 下一轮修法（两条候选，都需要实验判定）

1. **让 isoline 输出点包含每个段的两个端点**（而非去重后的点）——
   这与 `point_mode` 的语义冲突，但测试的 `m_n_result_vertices` 公式就是这么算的；
   需查 `mglTessEvalItemsPerPatch` 在 `point_mode + isolines` 下的点计数口径。
2. **让 XFB 写入按实例摊开满 `visible_bytes`** —— 若测试的布局假设是"每实例各自
   从缓冲起点排布"，则需在 `mglTessPlanXFBDestination` 里为每个实例重复写入。

判定方法：先只改 (2)（改动局部、风险低），跑
`barrier_guarded_read_write_calls` + `barrier_guarded_read_calls` + `points_verification`
三个用例；若 `patch[3]` 仍错，则说明是 (1) 的计数口径问题。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十八轮（2026-09-12）：#6 的因子值已取到 —— 测试的顶点预算与驱动实际不符

### 探针实测（`mglRenderTessEvalItemsPerPatch`）

```
MGLITEM gen=36474(ISOLINES) spacing=514(EQUAL) pm=1 inner=(1,0) outer=(1,1,1,1) -> 2
MGLITEM gen=36474 spacing=514 pm=1 inner=(~0,1) outer=(1,1,1,1) -> 2
MGLITEM gen=36474 spacing=514 pm=1 inner=(1,1) outer=(1,1,1,~0) -> 2
MGLITEM gen=36474 spacing=514 pm=1 inner=(1,1) outer=(1,1,~0,1) -> 2
```

即该用例（`layout(isolines, point_mode)`）每个 patch 只有 **2 个 TES 输出项**，
`patch_count=2` → `ipi=4`；驱动侧实际写入的顶点数远小于测试的预算。

### 与测试预算的对比

| 项 | 数值 | 来源 |
|---|---|---|
| 测试每实例预算 | **640 顶点** | `2 × 2 × 16 × 10` 的乘积式 |
| 驱动实测每实例 | **36 项**（早前 `ipi=36`）→ 更细粒度探针显示**每 patch 仅 2 项** | `MGLITEM` |

**注意两者口径不同**：测试按"每 patch 16 个结果顶点"算，
而驱动按"内层 (1,1) 的 isoline 展开"算 —— 只有 1 条 isoline、2 个点。
**这就是 patch[3] 读到未写入区域的根本原因**（读的是 XFB 没覆盖的部分，
里面残留着 tessellation level 的位模式）。

### 结论

这一轮把 #6 从"缺口 640 vs 36"进一步拆成了**口径差异**：
测试的 `m_n_result_vertices` 公式假设每 patch 产出 16 个结果顶点，
而 GL 的 isolines + inner=(1,1) 语义只产出 2 个。
**这已经不像 MGL 单侧的 bug，更像"测试预算假设 vs 规格语义"的差异**，
需要在下一轮先确认：GL 规格下 `isolines` + `point_mode` + inner=(1,1) + outer=1
究竟应产出几个顶点/段（对照 GL 4.6 §11.2.2.3 与 ARB_tessellation_shader 的 isolines 段落）。
**只有在确认"驱动少产出了"之后才改代码**，否则会写出为过测试而错的实现。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第十九轮（2026-09-12）：内层网格方向确认受阻（连续三轮净负）

本轮在"已知能让 `equal_spacing` + `fractional_even_spacing` 全绿"的那版网格上，
只加了**最小的退化处理**（`u_seg = nx>2 ? nx-2 : 0`，即 `<n>=1` 时不细分），结果：

| 网格版本 | 140 例 tess 簇 |
|---|---|
| 基线（HEAD，`9a1dc5b`） | **134 pass / 5 fail** ✅ |
| + 退化处理（本轮） | **130 pass / 9 fail** ❌ |

新增失败：`points_verification`、`tex_ordering`，以及
`vertex_spacing` 的 **`_`空 spacing / equal / fractional_odd** 三个一起挂。

### 连续三轮的同一模式

| 轮次 | 改动 | 结果 |
|---|---|---|
| 15 | "退化时保留顶点"版 | 133/6（`points_verification` 新增失败） |
| 16 | "跳过内层线"版 | 129/10（4 个用例新增失败） |
| 17 | 在"FE 全绿"版上加最小退化处理 | 130/9（4 个用例新增失败） |

**模式是稳定的**：任何"让内层网格几何更贴合 `vertex_spacing` 期望"的改动，
都会改变**退化配置下的网格**，从而破坏
`point_mode.points_verification` 与 `tex_ordering`——而我**没有独立于 CTS 的
正确性判据**来判断"哪种网格才是对的"，只能靠 CTS 自身的通过/失败，而这两组用例
的期望是**互相冲突**的（一组要 `rounded-2` 等长段，另一组要退化时保留特定顶点集）。

**结论**：在内层网格这条线上，我无法在"不破坏现有 134/5"的前提下取得进展 ——
需要一个**不依赖 CTS 对拍**的独立判据（例如与 Mesa / 参考实现的网格逐点比对，
或按 GL 4.6 §11.2.2.2 的图形逐一构造出参考网格），否则继续试错只会重复这三轮。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**，
实验版本全部存于 `git stash@{0..3}`。

---

## 第二十轮（2026-09-12）：#6 结案 —— ~~规格确认"测试账目与规格不符"，按你的决定跳过~~

> ⚠️ **本节结论已作废**（2026-09-12 复核）：把 CTS 取数步长里的 `16` 当成"每 patch 16 个结果槽"
> 是**量纲读错** —— 那 16 是 **16 个 int**，即一条被捕获顶点（4 varying × ivec4 = 64 B）。
> 测试与规格都只要求 **每 patch 2 个结果顶点**，不存在"账目与规格不符"。
> #6 **已重开为驱动侧缺陷**，详见文末「复核（2026-09-12）：#2 / #6 结案结论的更正」。
> 以下原文保留，仅作排查过程存档。

### 规格原文（`extensions/ARB/ARB_tessellation_shader.txt` §2.X.2.3，1158–1205 行）

> The number of isolines generated is derived from the **first outer** tessellation level;
> the number of segments in each isoline is derived from the **second outer** tessellation
> level. **Both inner tessellation levels and the third and fourth outer tessellation levels
> have no effect in this mode.**
>
> … the u==0 and u==1 edges … subdivided according to the first outer tessellation level.
> For the purposes of this subdivision, the tessellation spacing mode is **ignored and
> treated as equal_spacing**.
>
> If the number of isolines … is `<n>`, this process will result in `<n>` equally spaced
> lines with constant v coordinates of `0, 1/<n>, 2/<n>, …, (<n>-1)/<n>`.

该用例 TCS 写的是 `gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;`
（`esextcTessellationShaderBarrier.cpp:797-798`），inner 与 outer[2]/[3] 在 isolines 下**无影响**。

**→ 规格要求：1 条 isoline（v=0）、1 个段 → 整 patch 只有 2 个顶点。**

### 实测确认 MGL 符合规格

```
isolines outer=(1,1,1,1) inner=(1,1): vertices=2
  0: u=0 v=0
  1: u=1 v=0
point_mode count=2
```

### 不一致在测试账目侧

| 项 | 数值 | 出处 |
|---|---|---|
| 测试每实例预算 | 640 顶点 | `m_n_result_vertices = 2 × m_n_patches_per_invocation(2) × m_n_invocations(16) × m_n_instances(10)`（`:717`） |
| 测试取数位置 | patch 3 → byte 48 | `patch_data_int = data_int + patch × m_n_invocations × 2`，**按每 patch 16 个结果槽** |
| 规格 + MGL 实际 | **每 patch 2 顶点** | 上面实测 |

测试把 `m_n_invocations = 16` 当作"每 patch 的 isoline 数"，而 TCS 的 `outer[0] = 1` 只给出
1 条 isoline；于是测试按 16 槽/patch 去读，MGL 只写了 2 个顶点，**其余读到 GL 缓冲里
未定义的内容**（残留的 tessellation level 位模式，正好解释 `1.0f` 与 `0x3C000000`）。

### 处置（按用户决定）

**视为 CTS 侧账目问题，跳过 #6，不做任何"为过该用例"的改动。**
`mglTessEvalItemsPerPatch` / `mglTessDomainVertexCount` 在 isolines 下的
"每 patch 2 顶点"是**正确的、符合规格的**，前几轮把它当"驱动少产出"是误判，已更正。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第二十一轮（2026-09-12）：#2 Rule 3/4 的规格原文已取到，但补上"闭合"仍不通过

### 规格原文（`extensions/ARB/ARB_tessellation_shader.txt`，Appendix A.X，1487-1560 行）

> **Rule 3:** The set of vertices generated when subdividing any outer primitive edge
> is always symmetric. … For quad tessellation, if the subdivision generates a vertex with
> coordinates of `(x,0)` or `(0,x)`, it will also generate a vertex with coordinates of
> **exactly** `(1-x,0)` or `(0,1-x)`, respectively.
>
> **Rule 4:** The set of vertices generated when subdividing outer edges in triangular and
> quad tessellation must be **independent of the specific edge subdivided**, given identical
> outer tessellation levels and spacing. … For quad tessellation, if vertices at `(x,0)` and
> `(1-x,0)` are generated when subdividing the v==0 edge, vertices must be generated at
> `(0,x)` and `(0,1-x)` when subdividing an otherwise identical u==0 edge.

★ 关键字是 **"exactly"** —— 即**浮点位相等**，CTS 的 `isVertexDefined` 也正是位比较。

### 实测：MGL 两条边的位模式确实不一致

对失败配置 `inner=(4,5) outer=7 equal`（实测 40 点）：

```
bottom u values (first 3): 0.142857149 0x3e124925   0.285714298 0x3e924925   0.428571433 0x3edb6db7
left   v values (first 3): 0.857142866 0x3f5b6db7   0.714285731 0x3f36db6e   0.571428537 0x3f124924
```

底边有 `1/7 = 0x3e124925`，而左边是 `1 - 6/7 = 0x3f124924` 的补值 `0x3e124924` ——
**两者差 1 ulp**，所以 Rule 4 要求的"exactly `(0, 1/7)`"找不到。
违规计数（自建检查器）：**Rule 4 违规 3 处**。

数学上还有一层**硬约束**：`x -> 1-x` 把集合配成"对"，`1/2` 是唯一自配对点，
所以**任何闭合集合的元素个数必为奇数** —— 而 `{k/7}` 正好是 7 个，看似吻合，
但对 `k=3,4,5,6` 有 `1 - k/7 ∉ {k/7}`（例：`1 - 3/7 = 0x3f124924` ≠ `4/7 = 0x3f124925`），
**所以这 7 个值本身并不闭合**，必须扩大集合才能满足 Rule 3。

### 本轮尝试："把每个内点的补值也发出来"

按上面的推理，我在 `generate_quads` 的外缘发射里加了一个可开关的补值发射
（对每个内部位置 `t` 追加 `1-t`，去重后加进该边的顶点表）。结果：

| 配置 | 140 例 tess 簇（`MGL_TESS_RULE3_CLOSURE=1`） |
|---|---|
| 加补值 | **134 pass / 5 fail**（与基线同）—— `invariance_rule4` **仍然失败** |

原因：`Rule 4` 要的是**两条边各自都含 `x` 与 `1-x`**，而我追加补值后左边虽然多了
`0x3e124924` 一类点位，但**底边 `(1/7,0)` 与左边 `(0,1/7)` 的位仍不一致**
（左边补出来的仍是 `1 - 6/7` 这条路径，不是直接算的 `1/7`）。
要真正满足，需要**两条边都同时包含"直接算的 k/N"与"补出来的 1-k/N"**（即集合做
`S ∪ (1-S)` 的并），这会改变每条边的点数与顺序，而 `point_mode.points_verification`
与 `tex_ordering` 对顶点集敏感（前几轮已三次因此净负）。

**已回滚**，仓库保持 **134/5**。

### 与 #6 的区别（重要）

#6 我最终判定为**测试账目与规格不符**（规格支持 2 顶点/patch，测试按 16 槽读）。
**#2 不同**：规格原文（Rule 3/4 的 "exactly"）**支持 CTS 的判定**，
所以这是**驱动侧的真实缺陷**，只是修法需要同时改动两条边的顶点集，
而我目前没有独立判据来确认"改完后的顶点集是否既满足 Rule 3/4 又不破坏另外两组用例"。

**下一轮的正确做法**：写一个**离线检查器**（不依赖 CTS），对给定
`(spacing, outer, inner)` 直接验证 Rule 3 与 Rule 4 是否成立 —— 用它来判定网格改法，
而不是靠 CTS 的通过/失败试错。这正是我连续在多轮里缺的那个判据。

**当前状态**：工作树干净、`9a1dc5b` 为唯一提交、140 例 **134 pass / 5 fail / 1 ns**。

---

## 第二十二轮（2026-09-12）：用 GPU trace 定位 #5 —— 根因找到并修好一半

按你的提示用上了 macOS 27 的 Metal trace 工具链，这是本轮的关键。

### 工具链已跑通（可复用）

- `gpucapture` / `gpudebug` 存在（`/usr/bin/`），但捕获需要 **`MTL_CAPTURE_ENABLED=1`**。
- MGL 已有现成的捕获挂钩：**`MGL_GPU_CAPTURE=<path>.gputrace`**（原本只接在 GS 路径）。
  我给 TES compute 派发路径也加了同一个挂钩（已回滚，见下）。
- `gpudebug -t <trace>` 给出可导航的 Metal 命令树（`list` / `go <node>` / `info`），
  配合 `MTL_CAPTURE_ENABLED=1` 可看到每个 encoder 的 draw 调用与纹理附件。

### 用 trace 得到的决定性对比

| 用例 | 渲染 encoder | compute encoder |
|---|---|---|
| `points_verification`（通过） | **2052 points** + 4 points | 1 dispatch |
| `point_rendering`（失败） | **没有任何 draw** | 1 dispatch |

即**点根本没被提交**，所以读回是 `(0,0,0,0)`；不是着色器算错。

### 根因（已定位到具体代码）

`MGLRenderer+RenderPass.m` 的管线拓扑分类：

```objc
BOOL needsExplicitTopology = mglRenderNeedsExplicitTopology(
    geometryExpansion ? 1 : 0, (uint32_t)_lastDrawPrimitiveMode, ...);
if (needsExplicitTopology) {
    state->input_primitive_topology =
        mglRenderPrimitiveTopologyClass((uint32_t)_lastDrawPrimitiveMode);
}
```

**几何着色器发出的是"GS 的输出图元类型"，而不是 GL 的 draw mode**。
该用例用 `glDrawArrays(GL_PATCHES)` + `layout(points, max_vertices=5) out` 的 GS，
于是被分类成 `MTLPrimitiveTopologyClassTriangle`，Metal 拒绝链接：

```
MGL ERROR: Failed to create pipeline state: Vertex shader writes point size but
inputPrimitiveTopology is MTLPrimitiveTopologyClassTriangle
```

管线建不起来 → draw 被跳过 → 全 0。
`points_verification` 之所以通过，是因为它的 GS 恰好发 triangle strip（Triangle 类），**纯属巧合**。

### 修法（已验证有效）

按 `_geometry.program->geometry_output_type` 分类（GL_POINTS → Point，GL_LINE_STRIP → Line，
其余 → Triangle）。效果：**失败消失、管线成功建立、GS 开始运行**。

**但暴露出一个更深的既有缺陷**：修好拓扑后该用例**段错误**，
backtrace 落在 `libmgl.dylib` 的 `llvm::FunctionType::getReturnType(this=0x0)`
（`DerivedTypes.h:124`）—— 即 AIR codegen 里对一个 NULL 的 `FunctionType*` 取返回类型。

进一步查看发现相关可疑点（`mgl_air_backend.cpp:~11193`）：
GS/TCS/compute 的 user function **只注册到 `userFnDecls`（内联用），不创建 LLVM 函数**：

```c
if (isGS || isCompute || isTCS)
    continue;      /* 不创建，只内联 */
```

我试过让它对外联路径也创建函数（放开 `isGS`），**段错误依旧**，说明真正触发点还在更深处。

### 处置

拓扑修复**语义正确但单独提交会引入 1 个 crash**（140 例从 134/5 变 134 pass/4 fail/**1 crash**），
所以**未提交**，保存在 `git stash@{0}`（消息："gs point topology fix (correct, but exposes a GS-path segfault)"）。
仓库已回到已验证的 **134 pass / 5 fail / 1 ns**。

**下一轮第一步**：在 `mgl_air_backend.cpp` 里给 `llvm::FunctionType::get(...)` 的
两处调用点（`10348`、`11269`）加非空断言/日志，定位到底是哪一处拿到 NULL 返回类型；
很可能是 GS 路径某个 helper 的 `fs->return_type` 或参数类型转换失败。

**本轮的关键收获**：GPU trace 工具链跑通了（`MTL_CAPTURE_ENABLED=1` + `MGL_GPU_CAPTURE` + `gpudebug`），
#5 的根因（GS 输出图元拓扑分类错误）已确证并修好，只剩一个被它暴露出来的 codegen 段错误。

---

## 第二十三轮（2026-09-12）：**已提交** —— GS 输出图元拓扑修复，GS 簇 126/10 → 136/0

承接上一轮用 GPU trace 定位到的根因，本轮把修复做完并**提交**（commit `d4691bd`）。

### 修复 1：管线拓扑按 GS 的输出图元分类

`MGLRenderer+RenderPass.m` 原来用 `_lastDrawPrimitiveMode`（GL 的 draw mode）分类，
但 **Metal 的 `inputPrimitiveTopology` 要跟 GS 声明的输出图元一致**。
现改为按 `_geometry.program->geometry_output_type`（POINTS→Point、LINE_STRIP→Line、其余→Triangle）。

### 修复 2：`controlPointGetter` 空指针保护

修好拓扑后 GS 开始真正编译，暴露出第二处既有缺陷（backtrace 落在
`llvm::FunctionType::getReturnType(this=0x0)`）：`mgl_air_backend.cpp:5297` 的
`CreateCall(cg.controlPointGetter, ...)` **没有像同文件 2807/2878 那样检查空指针**，
于是构造了一个 callee 为 NULL 的 `CallInst` 并崩在 LLVM 类型访问器里。
已按同样方式加保护，改为报错而不是段错误。

### 实测结果（用你给的 per-case runner）

| 用例簇 | 修复前 | 修复后 |
|---|---|---|
| `KHR-GL46.geometry_shader.*`（136 例） | **126 pass / 10 fail** | **136 pass / 0 fail** ✅ |
| `KHR-GL46.tessellation_shader`（140 例） | 134 pass / 5 fail / 1 ns | **134 pass / 5 fail / 1 ns**（无回归） |

**新通过的 10 个用例恰好全部是 `*_input_points_output_*`** ——
即"输入任意、GS 输出 points"的那一组，正是本修复直接命中的形态：

```
rendering.lines_input_points_output_line_loop_drawcall
rendering.lines_input_points_output_line_strip_drawcall
rendering.lines_input_points_output_lines_drawcall
rendering.lines_with_adjacency_input_points_output_line_strip_drawcall
rendering.lines_with_adjacency_input_points_output_lines_adjacency_drawcall
rendering.triangles_input_points_output_triangle_fan_drawcall
rendering.triangles_input_points_output_triangle_strip_drawcall
rendering.triangles_input_points_output_triangles_drawcall
rendering.triangles_with_adjacency_input_points_output_triangle_strip_adjacency_drawcall
rendering.triangles_with_adjacency_input_points_output_triangles_adjacency_drawcall
```

### `point_rendering` 的当前状态（仍未通过，但性质变了）

不再"静默画不出东西"，而是变成**可控的链接失败**：
`Program linking failed`（`esextcTessellationShaderPoints.cpp:528`）。
原因是该 TES 走 render-vertex 路径时 `cg.controlPointGetter` 为空，
而 shader 里访问了 `gl_in[i].<field>` —— 即**这条路径尚未实现 `gl_in` 索引访问**。
这是明确、可执行的下一步，不再是黑盒。

### 工具链固化（可复用）

```bash
# 1) 让进程可被捕获：必须设置 MTL_CAPTURE_ENABLED=1
# 2) MGL 已有挂钩：MGL_GPU_CAPTURE=<path>.gputrace（原本只接 GS 路径）
MTL_CAPTURE_ENABLED=1 MGL_GPU_CAPTURE=/tmp/x.gputrace \
  DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts --deqp-case=... ...
# 3) 用 gpudebug 导航命令树
gpudebug -t /tmp/x.gputrace          # list / go <node> / info
```

---

## 第二十四轮（2026-09-12）：补上 #2 的离线判据，并证伪"补补值"这条路

### 已提交：#2 的离线 Rule 3/4 检查器

`invariance_rule4` 用**浮点位相等**比较坐标，靠跑整条 CTS 用例来迭代太慢、
而且只给通过/失败。我写了 `tools/cts-tess-check/`（commit `5b08570`）：

```sh
./build.sh && ./rule34 <spacing> <o0> <o1> <o2> <o3> <i0> <i1>
```

它直接把 ARB_tessellation_shader Appendix A.X 的 Rule 3 / Rule 4 施加在 MGL 的
域输出上，并**打印出违例的位模式**。当前失败配置的输出：

```
spacing=0 outer=(7,7,7,7) inner=(4,5) points=40 bottom=6 left=6
  RULE3 miss: 1-0.857142866=0x3e124924 not on bottom edge
  RULE4 miss: x=0.142857149 0x3e124925 -> (0,x)=N (0,1-x)=Y
  RULE3 violations=3  RULE4 violations=3  => FAIL
```

这就是我前面几轮一直缺的那个"独立于 CTS 的判据"。

### 用它证伪了一条修法

我实现了"把每个内点位置的补值也发出来"（`MGL_TESS_RULE3_CLOSURE`，默认关），
离线实测：

| 方案 | Rule 3 | Rule 4 |
|---|---|---|
| 现状 | 3 处违例 | 3 处违例 |
| 补补值后 | **0 处** ✅ | **仍 3 处** ❌ |

**Rule 3 可以靠补值解决，Rule 4 不行。** 原因是两条边的位置来源不同：
`v==0` 边来自**外缘**的细分（`k/7`），`u==0` 边来自**内层网格**的细分
（`1/5, 2/5, 3/5, 4/5`）——两者对同一坐标发布的浮点值不同，
所以规则 4 要求的"另一条边上恰好存在"无法靠单边补值满足。

这个结论是**离线判定**的，没有再消耗一轮 CTS 回归。该实验已回滚。

### 下一步（方向已明确）

要满足 Rule 4，需要让两条边的**位置集合同源**——即让 `u==0` 边获得
与 `v==0` 边相同的 `k/outer` 位置集（这正是 #1 里"内层网格应与外缘同源"
的同一个根因）。也就是说 **#1 与 #2 可能是同一个修复的两个侧面**：
把内层网格改为由外缘细分驱动，就可能同时解决 Rule 4 与内层段数。

**当前状态**（两簇均已复核）：

| 用例簇 | 结果 |
|---|---|
| `KHR-GL46.geometry_shader.*`（136） | **136 pass / 0 fail** ✅（本轮提交的修复） |
| `KHR-GL46.tessellation_shader`（140） | **134 pass / 5 fail / 1 ns**（无回归） |

工作树干净；提交 `5b08570`（检查器）+ `d4691bd`（GS 拓扑修复）+ `9a1dc5b`（#4）。

---

## 第二十五轮（2026-09-12）：#2 撞上**数学上的硬矛盾**（与 #6 同类）

> ⚠️ **"数学硬矛盾"的结论已作废**（2026-09-12 复核）：存在**仍是 N+1 个点**且对 `x → 1−x`
> **逐位闭合**的构造（下半段取严格的 `1 − fl((N−i)/N)`，被减数 ≥ 0.5 时补减法在 float32 下精确），
> 无需把边扩到 8 段；且 CTS rule4 是**条件式**判据，只要四条边发布同一套位置值即满足。
> 事实也已推翻该结论：`6d205b4` 修好后 `invariance_rule1..7` 全部 Pass。详见文末复核。

用上一轮提交的离线检查器（`tools/cts-tess-check/`）逐位核算后，得到一个确定结论：

### Rule 3 与"边必须恰好 N 段"在 N=7 时不兼容

等距 N=7 时，外缘发布的 6 个内部位置（逐位）：

```
i=1: 0x3e124925     i=4: 0x3f124924
i=2: 0x3e924925     i=5: 0x3f36db6e
i=3: 0x3edb6db7     i=6: 0x3f5b6db7
```

Rule 3 要求集合对 `x -> 1-x` **封闭**（"exactly"）。逐位求补：

```
集合大小 = 6
缺失的补值 = 3 个（0x3edb6db8 / 0x3e924924 / 0x3e124924）
→ 必须补到 9 个值，即该边会变成 8 段，而不是规格定义的 6 段
```

**矛盾的实质**：满足 Rule 3 需要"补值也在集合里"，而补值（如 `1-4/7 = 0x3edb6db8`）
**不等于任何 `k/7` 的浮点值**（`4/7 = 0x3f124924` 与 `1-3/7 = 0x3f124924` 才是一对）。
要同时容纳"直接算的 `k/7`"与"补出来的 `1-k/7`"，集合必须扩大 ——
而扩大就违反"这条边正好 N 段"。

### 与 #6 的同质性

| | #6 `tc_barriers` | #2 `invariance_rule4` |
|---|---|---|
| 冲突双方 | 规格（每 patch 2 顶点） vs 测试账目（每 patch 16 槽） | 规格 Rule 3（浮点"exactly"闭合） vs 规格 spacing（恰好 N 段） |
| 性质 | CTS 侧假设与规格不符 | **规格自身在 N=7 时不可同时满足** |
| 处置 | 按你的决定跳过 | **待你决定** |

两者都是"测试期望与可实现行为不一致"，不是单纯实现缺陷。

### 我已排除的实现级解释（都试过、都无效）

- `edge_position` 已有的镜像分支（`1 - pos(N-i)`）**并未产生闭合** ——
  实测 `mirrored(4) = 1 - pos(3) = 0x3f124924`，而 `1 - 0x3f124924 = 0x3edb6db8` 不在集合内。
- 上一轮试过的"为每个内点补发补值" → **Rule 3 归零，但 Rule 4 仍 3 处违例**（已回滚）。
- 本轮确认：即使 Rule 3 修好，Rule 4 还需要 `u==0` 边与 `v==0` 边**位置集合同源**，
  而这会把内层网格改成由外缘驱动 —— 与 #1 是同一处改动。

### 需要你的决定（与 #6 同样的问题）

`invariance_rule4` 在 N=7 这类配置上，**规格 Rule 3 + 规格 spacing 两条要求无法同时成立**。
请指示：

1. **视为"测试不可满足"而跳过**（像 #6 那样），把规格矛盾写清楚结案；或
2. **按"扩大集合"实现**（该边 8 段而非 6 段），接受它会改变外缘细分并可能影响
   `vertex_spacing` 系列；或
3. 先查 CTS 上游是否已有该用例的已知 issue / 澄清。

**当前状态**（两簇均已复核）：

| 用例簇 | 结果 |
|---|---|
| `KHR-GL46.geometry_shader.*`（136） | **136 pass / 0 fail** ✅ |
| `KHR-GL46.tessellation_shader`（140） | **134 pass / 5 fail / 1 ns**（无回归） |

工作树干净；提交 `5b08570`、`d4691bd`、`9a1dc5b`。

---

## 第二十六轮（2026-09-12）：#3 的离线探针建立，但**证伪了我自己的判断**

按计划转向 #3（`inner==1` 的 marker 拓扑）。我先写了离线探针
（`tools/cts-tess-check/marker_check.c`，commit `4528101`），它复刻 CTS 的
marker 搜索并对 MGL 的域输出报告"哪些行是满宽（x 同时含 0 与 1）"。

### 结果与我此前的判断相反

对**最简单的** `inner=(1,3) outer=3 fractional_odd`：

```
inner=(1,3) outer=3 FO: vertices=54 tris=18
unique y(5): 0 0.333333 0.333333 0.666667 1
second_from_top=0.333333 second_from_bottom=0.666667
marker (y1_y2=0.333333, y3=0) found=NO
full-width row at y=0
full-width row at y=0.666667
full-width row at y=1
```

- 该配置下 **marker 搜索在 `y1_y2 = second_from_top = 1/3` 上不成立**，
  但注意 **`y=2/3` 与 `y=1` 都是满宽行** —— 而 CTS 对 `inner[0]==1` 用的正是
  `second_from_top` / `second_from_bottom`，即 `y=1/3` 与 `y=2/3`。
- 也就是说：**满宽行确实存在**（`y=2/3`、`y=1`），但 CTS 选取的那一行（`1/3`）
  恰好不是满宽行 —— 这与我此前"cap 三角没连成一体"的表述**不完全一致**。

**诚实的结论**：我原先对 #3 的定位（"全宽内层线存在、缺的是 cap 连线"）需要修正；
真正的问题更可能是 **`inner[0]==1` 时内层矩形的退化位置与 CTS 期望的行不一致**。
而且**这次探针没能复现失败**，说明 CTS 里失败的那一个 pass 用的是**别的**
`inner[0]==1` 配置（该用例会遍历多个 tess level），而 **CTS 的错误信息没有打印配置**。

### 下一步（明确）

在 CTS 侧给 marker 失败处加桩，打印 `run.set1_inner/outer/vertex_spacing`，
先弄清**到底是哪个配置在失败**，再动 `generate_quads`。
（本轮没有盲改代码 —— 上次连续三轮净负的根因正是"照着错的靶子改"。）

### 当前状态

| 用例簇 | 结果 |
|---|---|
| `KHR-GL46.geometry_shader.*`（136） | **136 pass / 0 fail** ✅ |
| `KHR-GL46.tessellation_shader`（140） | **134 pass / 5 fail / 1 ns**（无回归） |

工作树干净；提交 `4528101`（marker 探针）、`5b08570`（Rule 3/4 检查器）、
`d4691bd`（GS 拓扑修复）、`9a1dc5b`（#4）。

**待你决定的两项**：#2（规格在 N=7 时 Rule 3 与 spacing 不可同时满足）与
#6（已按你的决定跳过）。

> ✏️ **2026-09-12 复核更正**：这两项"待决定"都已不成立 ——
> #2 已由 `6d205b4` 修好（rule1–rule7 全 Pass），#6 的"CTS 侧账目"结论作废、重开为驱动侧缺陷。
> 见文末「复核（2026-09-12）：#2 / #6 结案结论的更正」。

---

## 第二十七轮（2026-09-12）：已提交 —— 清除退化因子造成的 6e-08 伪坐标

### 用 CTS 侧桩定位到**真正失败的配置**（这一步是上轮缺的）

给 marker 失败处加桩打印 run 描述符后，拿到确切配置：

```
MGLMARKER FAIL inner=(1,3) outer=(3,3,3,3) vs=2(FO) n_vertices=54
  second_from_top=0.333333 second_from_bottom=0.666667
```

并把 CTS 拿到的坐标集打出来，**发现关键异常**：

```
MGLMARKER xset: 0  5.960464e-08  0.3333334  0.6666666  0.9999999  1
```

坐标里出现了 **`5.96e-08`**（= 2⁻²⁴）和 **`0.9999999`**，而本该是精确的 `0` / `1`。

### 根因：退化因子直接进了 fractional 公式

在 `edge_position` 里加桩，抓到产生该值的调用：

```
MGLFPOS direct factor=1 seg=3 idx=1 -> 5.96046377e-08 (0x337ffffe)
```

即 **`factor = 1` 被传进 fractional 分支**，公式 `(f-(n-2))/(2f)` 算出 `(1-1)/2 = 0` 附近的值。
而 `ARB_tessellation_shader` 明确：**`<n> = 1` 时该边不细分**；参考实现会把因子
**下限钳到 2** 再分子段。

### 修复与效果（commit `0b5e41d`）

钳位因子下限为 2。实测：`inner=(1,3) outer=3 FO` 的边界集合

```
修复前: {0, 5.96e-08, 1/3, 2/3, 1}
修复后: {0, 1/3, 2/3, 1}
```

—— 内层矩形角点不再是与域边重合的**另一个浮点值**。

### 验证（无回归）

| 用例簇 | 结果 |
|---|---|
| `KHR-GL46.geometry_shader.*`（136） | **136 pass / 0 fail** |
| `KHR-GL46.tessellation_shader`（140） | **134 pass / 5 fail / 1 ns** |

### 剩余障碍（已缩小）

`inner_tessellation_level_rounding` 仍失败，但**失败原因变了**：
不再是 6e-08 伪坐标，而是**反射生成的外缘把 1/3 发出了两种差 1 ulp 的值**
（`0x3eaaaaab` 与 `0x3eaaaaac`）。CTS 的 marker 用**位相等**比较，
所以 `y-set` 里同时出现 `0.3333333` 与 `0.3333334`，marker 三角的匹配被打断。

**下一轮**：修外缘反射路径，让 `1/3` 只以一种位模式出现
（`? :` 三分支的反射分支 `1 - pos` 与另两条直接路径给出的位不同）。

**当前提交**：`0b5e41d`（本轮）、`4528101`（marker 探针）、`5b08570`（Rule 3/4 检查器）、
`d4691bd`（GS 拓扑修复）、`9a1dc5b`（#4）。

---

## 第二十八轮（2026-09-12）：3 例修复落地 —— 134/5 → 137/2

### 本轮结论表

| 用例 | 性质 | 结论 |
|---|---|---|
| `quads_tessellation.inner_tessellation_level_rounding` | **真 bug，已修** | 退化 `inner==1` 的 ε 短段被保留成"可表示的"坐标 → 退化外区不塌陷 |
| `tessellation_invariance.invariance_rule4` | **真 bug，已修** | 反射边用 `1 - t` 落地，`1-(1-x) != x` 差 1 ulp → rule4 位等比较失败 |
| `tessellation_shader_point_mode.point_rendering` | **真 bug，已修** | TES-render-vertex 路径缺 `gl_in[i].<用户字段>` 读取；且该路径 varying 标签与 FS 不一致 |
| `vertex...fractional_odd_spacing` | **CTS 侧账目问题（证据已列，见下）** | 与同用例 equal_spacing 分支自相矛盾 |
| `tc_barriers...` | CTS 侧账目问题（前轮结论，用户已裁定跳过） | — |

提交：`6d205b4`（域生成）、`8b1efdd`（AIR 接口 / gl_in）。

### 1. `inner_tessellation_level_rounding`：1+ε 的"极限"必须落成 0/1

CTS 断言（`esextcTessellationShaderQuads.cpp:826-849`）：`inner[0]==1` 时，域里必须存在
一个"marker 三角"——**一条横跨全域的边**（两顶点 `x==0.0f`、`x==1.0f` 位等）加第三个顶点在
`y==0` 或 `y==1`。用桩打印拿到确切配置：`inner=(1,3) outer=(3,3,3,3) FO`。

**根因（两层）**：

1. `mgl_tess_factor_normalize.c` 把 `inner_clamped==1` 用 `nextafterf(1.f,2.f)` 抬成 1+ε，于是
   `edge_position()` 的 `(f-(n-2))/(2f)` 算出 `5.96e-08`（0x337ffffe）——**可表示的 ε 偏移**，
   内层矩形边缘与域边不重合，退化外区不塌陷。
2. 规格原文（ARB_tessellation_shader，spacing 段）：
   "the length of the two additional segments ... **As `<n>-<f>` approaches 2.0, the relative length
   of the additional segments approaches zero**"；quad/triangle 退化段又写明
   "the three-segment subdivision **may produce "inner" vertices positioned on the edge of the
   rectangle/triangle**"、"may be numerically indistinguishable"，issue 50 更明确
   "point_mode ... **can produce multiple vertices with the same position**"。
   ⇒ `f` 应取**clamped level**（1.0），`n=3`，于是 `n-f = 2` 恰好是"短段长度为 0"的极限：
   位置序列为 `0, 0, 1, 1`，两端段退化、内段占满整条边。

**修法**：
- `edge_position` 用 `f = (double)factor`（不再把 factor 钳到 2；上一轮 `0b5e41d` 的钳位是
  对 `nextafterf` hack 的局部补救，本轮换成根治）。
- 删除 `mglTessNormalizeFactors` 里两处 `nextafterf(1.f, 2.f)`；"1+ε" 只作用于**段数**
  （`tess_round_inner` 已处理：`ceil(1)=1 → 2 → FO 3 / FE 2 / equal 2`）。

实测同一坐标集（`inner=(1,3) outer=3 FO`，point mode）：
```
修复前: 内层网格列 u ∈ {5.96046377e-08, 0.99999994}
修复后: 内层网格列 u ∈ {0, 1}            ← 与域边重合，退化外区消失
```
marker 三角 `((0,1),(0,2/3),(1,2/3))`（top strip 末三角）与 `((1,0),(1,1/3),(0,1/3))`（bottom strip
末三角）随之成立。

### 2. `invariance_rule4`：反射边必须走**索引反射**而不是值反射

rule4 判据（`esextcTessellationShaderInvariance.cpp:1755`）是**位等**：
`(x,0)` 与 `(1-x,0)` 存在 ⇒ `(0,x)`、`(0,1-x)` 必须位等存在。

两层根因，本轮都修：

- **quad**：edge 2/3（上、左）原先算 `1.f - t`。`1-(1-x)` 在 float 下不回环，于是
  `1/7` 在左边缘发布成 `0x3e124924`（右边是 `0x3e124925`）→ `(0,1/7)` 找不到。
  改为**用镜像索引取位置**：`r = edge_position(f, n, spacing, n - i)`，四条边都从同一函数取值
  → 值与值集完全一致。
- **triangle**：重心坐标原先用 `w = 1 - u - v` 重建（`0.6f + 0.4f` 差 1 ulp → `w=0x3ecccccc`，
  而 rule4 期望 `0x3ecccccd`）。改为**三分量都取自位置函数**：
  `(u,v,w)` 是 `{t, r, 0}` / `{r, t, 0}` / `{0, r, t}` 的置换
  （quad 的 `(t,0,r)` / `(1,t,0)` / `(r,1,0)` 同理）。

离线复算（`/tmp/r4_new.txt` 点位，位等模拟 CTS 判据）：rule4 misses = 0；左边 y 值集 == 底边 x 值集。

### 3. `point_mode.point_rendering`：TES-render-vertex 的两个缺口

失败链：`Program linking failed` → 修好后变 `glDrawArrays ... GL_INVALID_OPERATION`。

- **缺口 A（链接失败）**：`MGL/src/mgl_air_backend.cpp:5250` 的"索引型 control-point 输入"
  （`tcColor[i]` 这类，不是 `gl_in[i].x`）只在 `isTESCompute` 时读共享记录流；
  TES-render-vertex（exec=4）落到 Metal control-point function 分支 → 该函数在此路径不存在
  （`Program linking failed` / "control-point getter is unavailable"）。
  修：`isTESCompute || isTESVertex` —— 该路径的 slot 30 就是同一份 control-point 记录流，
  patchId 来自 contract buffer(slot 29)，与 compute 展开完全一致（代码里 2874 行注释已如此描述）。
- **缺口 B（PSO 失败）**：Metal 报
  `Fragment input(s) \`mgl_loc_0\` mismatching vertex shader output type(s) or not written by vertex shader`。
  用 `MGL_IFACE_TAG_TRACE` 桩拿到两侧标签：
  ```
  vs_out name=result_color loc=0 explicit=0 tag=result_color  isTESVertex=1
  fs_in  name=result_color loc=0 explicit=0 tag=mgl_loc_0     has_gs=1
  ```
  `program.c:1706-1712` 为了 ABI 一致，给"TES 为 compute/render-vertex"的程序里的 **FS** 打上
  `MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER`（于是 FS 用 `mgl_loc_N` 标签，并按 name 把位置重映射到
  TES 输出位置）；但同一个 TES 自己的输出仍用 GLSL 名 → 两边对不上。
  修：AIR 后端引入 `ifaceGs = has_gs || isTESVertex`，只用于**顶点侧 stage-out 的标签与 float
  carrier**（返回结构体构造、outNodes 元数据、`cg.has_gs`）；其它 stage 行为不变。

### 4. `fractional_odd_spacing`（唯一剩下的"疑似真 bug"）—— 实测是 CTS 侧账目问题

失败：`expected:29, found: 31`（`esextcTessellationShaderVertexSpacing.cpp:1301`）。

用 `cts_sim.py`（忠实复刻 `getEdgesForQuadsTessellation` + `verifyEdges`）跑失败配置
`inner=(32,64) outer=(1,32,1,32) FO`：

```
edge0 pts=34 clamped_rounded=33  2-delta expect {2,31} got [2, 31]   ← 外边：2 短段+31 中段 ✓
edge4 pts=32 clamped_rounded=31  expect 29 segments, got 31          ← 内层矩形边 ✗
edge5 pts=62 clamped_rounded=63  expect 61 segments, got 61          ← 巧合通过
```

**MGL 的 31 段是规格形状**：§2.X.2.2 "the u==0 and u==1 edges ... are subdivided into `<m>` segments
... Each vertex ... joined ... to produce a set of vertical and horizontal lines that divide the
rectangle into a grid"；"The boundary of the region covered by these triangles forms an inner
rectangle, the edges of which are subdivided by **the grid vertices that lie on the edge**"。
FO 下网格线在 `p_1..p_{m-1}`，相邻间距恒为 `1/f`，故内层边 = `m-2 = 31` 段**等长**，
两个短段 `[0,p_1]`、`[p_{m-1},1]` 属于**外侧边界带**（不在内层边上）。

**CTS 侧的矛盾**：
- 同测试的 **equal_spacing** 兄弟用例（MGL 通过）期望 `round(inner-2) = 30 = m-2`，即网格模型；
- FO 分支却期望 `round(inner-2) - 2 = 29`（或 2-delta 的 `{2,29}`），相当于把内层边当成"自己再按
  FO 细分一次"的嵌套环模型；
- 两种模型不可能同时满足一个一致实现的输出（若改成嵌套模型，equal 分支会由通过变失败）。

**全量对拍（离线，48 + 144 个 run）**：按 CTS `getTessellationLevelSetForPrimitiveMode(QUADS, 64,
INNER_AND_OUTER_LEVELS_USE_DIFFERENT_VALUES)` 枚举出该用例的全部 run（FO 侧 48 个，(inner0,inner1) ∈
{(32,64),(64,32)}），用 `fo_sweep.py` / `eq_sweep.py` 逐个跑忠实模拟：

| 变体 | run 数 | 被 CTS 判据拒绝 |
|---|---|---|
| `vs_mode_fractional_odd_spacing` | 48 | **48**（全部） |
| `vs_mode_equal_spacing`（MGL 通过） | 144 | **0** |

且 FO 的 48 个拒绝**全部**落在 inner 级别为 32 的那对"内层矩形边"上，误差恒为 `expected 29, found 31`；
级别 64（被 clamp 成 63）的内边一律 `61 = 61` 通过。把三个式子写在一起即可看出矛盾（`clamp` 上限 63）：

| L | 网格模型真值 `FOround(clamp(L)) - 2`（MGL） | CTS FO 期望 `FOround(clamp(L-2)) - 2` | CTS equal 期望 `ceil(clamp(L-2))` |
|---|---|---|---|
| 32 | 31 | **29** ✗ | 30 = m-2 ✓ |
| 64 | 61 | 61 ✓（clamp 使 `FOround(62)=FOround(63)=63`） | 62 ✓ |

即：**同一条被同一段提取代码取出的边**，CTS 的 equal 分支期望 `m-2`、FO 分支却期望 `(m-2)-2`；
L=64 时两者恰好重合，L=32 时相差 2。⇒ 按用户上一轮对 `tc_barriers` 的裁量口径，**标注证据后跳过**；
不写 CTS 特判（改成能过 FO 的嵌套环结构会让 equal 分支由通过转失败）。

### 验证矩阵（本轮全部实跑）

| 用例簇 | 基线 | 本轮 |
|---|---|---|
| `KHR-GL46.tessellation_shader`（140） | 134 pass / 5 fail / 1 ns | **137 pass / 2 fail / 1 ns** |
| `KHR-GL46.geometry_shader.*`（136） | 136 / 0 | **136 / 0** |
| `make test-tess-domain` | ok | **ok** |

复现命令（两簇）：
```
cd /Users/fterward/VK-GL-CTS-build-mgl-target
python3 -u run_mgl_cts_cases.py --glcts .../modules/glcts   --caselist mgl-tess-cluster-cases.txt --workdir .../modules   --outdir mgl-r26t-<ts> --dyld-library-path /Users/fterward/MGL-minecraft --timeout 45
```

CTS 源码树已还原干净（`git status` 仅剩既有未跟踪的 build 产物）。

---

## 收尾核对（2026-09-12，pristine CTS 二进制 + 提交 `4a6b7da`）

单例复核（`glcts --deqp-case=...`，CTS 源码树已还原并 `ninja glcts` 重建）：

| 目标项 | 用例 | 结果 |
|---|---|---|
| xfb TCS-without-TES 校验 | `single.xfb_captures_data_from_correct_stage` | **Pass** |
| invariance_rule4 位等对称 | `tessellation_invariance.invariance_rule4` | **Pass** |
| inner==1 marker 拓扑 | `..._quads_tessellation.inner_tessellation_level_rounding` | **Pass** |
| point_mode / tc_barriers 调查 | `tessellation_shader_point_mode.point_rendering` | **Pass** |
| | `tessellation_shader_tc_barriers.barrier_guarded_read_write_calls` | **仍 Fail** —— "CTS 侧账目"结论作废，**重开为驱动侧缺陷**（见文末复核） |
| fractional_odd 内层短段 | `vertex.vertex_spacing_..._fractional_odd_spacing` | CTS 侧账目（见 §4；离线 48/48 被拒 vs equal 144/144 通过） |

**复核补测（同一 HEAD `4a6b7da`，单例）**：`point_rendering` Pass、
`inner_tessellation_level_rounding` Pass、`invariance_rule4` Pass（`invariance_rule1..7` 全 Pass）、
`barrier_guarded_read_write_calls` **Fail**、`vertex...fractional_odd_spacing` **Fail** —— 与上表一致。

簇级复核：

| 用例簇 | 基线 | 收尾 |
|---|---|---|
| `KHR-GL46.tessellation_shader`（140） | 134 pass / 5 fail / 1 ns | **137 pass / 2 fail / 1 ns** |
| `KHR-GL46.geometry_shader.*`（136） | 136 / 0 | **136 / 0** |
| `make test-tess-domain` | ok | **ok** |
| GL46 hotspot 全量（1328） | — | 1270 pass；失败项全为既有热点（copy_image / packed_depth_stencil / gpu_shader5 / DSA…），全表**无一条 Interface mismatch** |

提交：`9a1dc5b`（#1）、`6d205b4`（#2+#3）、`8b1efdd`（point_mode）、`4a6b7da`（离线对拍工具）；
`0b5e41d` 的 factor 钳位已被 `6d205b4` 取代。工具入口：`tools/cts-tess-check/`。

---

## 复核（2026-09-12）：#2 / #6 两条"结案"结论的更正

本节是一次**独立复审**（重读 CTS 源码 + 规格原文 + 单例实测 + 离线数值验证），
结论：上文两条"测试不可满足 / 跳过"的定论**都不成立**。各轮原文保留作排查存档，结论以本节为准。

| 条目 | 原结论 | 复核裁定 | 现状（`4a6b7da`） |
|---|---|---|---|
| #2 `invariance_rule4` | 规格 Rule 3（位等闭合）与"恰好 N 段"在 N=7 数学上不可兼得 → 跳过或扩集合 | ❌ 不成立：存在 **N+1 点且逐位闭合**的构造；且 CTS 判据是**条件式** | ✅ 已修（`6d205b4`），rule1–rule7 全 Pass |
| #6 `tc_barriers` | 测试按"每 patch 16 个结果槽"取数，与规格的 2 顶点/patch 冲突 → CTS 侧账目，跳过 | ❌ 不成立：`16` 是 **16 个 int** = 一条被捕获顶点，不是 16 个顶点 | ❌ 仍 Fail → **重开为驱动侧缺陷** |

### #6：CTS 的取数布局与规格完全一致（原结论是量纲读错）

逐行核对 `esextcTessellationShaderBarrier.cpp`：

- XFB varyings 是 `tes_result1..4`，各 `ivec4` → **一条被捕获顶点 = 16 ints = 64 B**
  （与 `getXFBBufferSize()` 的 `sizeof(int) * 4 * 2` 自洽）。
- `:932` `patch_data_int = data_int + n_patch * m_n_invocations * n_points_per_line_segment`
  = `n_patch * 16 * 2` = **32 ints = 2 条记录**，不是 16 条。
- `:928-940` 的比对循环只检查**第 1 条**记录的 16 个 int，源码注释明确写着
  "the same set of values will be reported for the other point" —— **测试期望就是 2 结果点/isolines**。
- 规格侧（`ARB_tessellation_shader.txt`，Isoline Tessellation）：`isolines` 且 `outer[0]=outer[1]=1`
  → 1 条 isoline × 1 段 → **2 顶点/patch**。测试预算 == 规格，无冲突。
- 该用例是 Khronos must-pass（`gles32-khr-main.txt:975`），shipping driver 全过 ——
  "测试不可满足"在实践上不成立。

**现场实测（HEAD `4a6b7da`，pristine CTS）**：仍 Fail，断言

```
Result data for patch [2]: (1006648320, 1065353216, 0, 0, 32, 0, 0, 0), expected: 8, 0, 0, 0, 32, 0, 0, 0
```

前两个 int 是 `0x3C000000`（1/128）与 `0x3F800000`（1.0）的**浮点位模式**，而第 5 个 int（应为 32）是对的
—— 即 patch 2 的**第 1 个 varying 整段**读到了非 XFB 数据、第 2 个 varying 正常。
这与"缓冲整体没写满"的形态不同，更像**记录内字段排布 / 目的偏移**问题。

**下一步（建议，未动手）**：给 XFB 捕获加一次性探针
（`mglTessPackXFBInterleaved` / `mglTessResolveXFBSource` / `mglTessPlanXFBDestination`），核对三件事：
① 实际写入记录数（旧测 `ipi=36 → copied_verts=360`，而 200 patch × 2 = **400**）；
② 每条记录里 4 个 varying 的 16 B 是否按 GL interleaved 排布；
③ `session_offset` 与实例分量。测试只检查前 40 个 patch（= 80 条记录 = 5120 B），这一段写对即可通过。

### #2：不存在"数学硬矛盾"（已被 `6d205b4` 事实推翻）

- **构造反例**：让边的**下半段**取严格的 `1 − fl((N−i)/N)`（被减数 ≥ 0.5 时 float32 补减法**精确无舍入**），
  上半段直接算 `(float)i/N`，得到的集合仍是 **N+1 个点（N 段）**、单调、每点距 `i/N` ≤ 1 ulp，
  且对 `x → 1−x` **逐位闭合**。离线 C 程序验证 N=3/5/7/9/13/15/17/31/33/63 **全部 RULE3 = 0**
  （朴素 `i/N` 布局在 N=7 有 4 处违例、N=63 有 39 处）。→ "必须把边扩到 8 段"是错误推论。
- **CTS 判据是条件式**（`esextcTessellationShaderInvariance.cpp:1793-1820`）：只有 `(x,0)` 与 `(1−x,0)`
  同时存在时才要求 `(0,x)` / `(0,1−x)`。因此**四条边发布同一套位置值**即满足 —— 既不需要全局闭合，
  也不需要扩点。`6d205b4` 正是这么修的（每个坐标都从同一个 position 函数按镜像索引取值）。

### 一条需要留意的残留

离线检查器（`tools/cts-tess-check/rule34`）在 HEAD 上**仍报** strict Rule 3 违例
（`equal_spacing`、`outer` 全 7、`inner=(4,5)` 有 3 处），而 CTS `invariance_rule3` 实测 **Pass**。
⇒ 该检查器的 Rule 3 模型（只在同一条边上搜索补值）比 CTS 的实际取数口径更严，
**暂不能当作唯一判据**，需要先与 CTS 口径对齐。若要主动加固，可把 `edge_position()` 的镜像方向反过来
（现在是"上半段 = 1 − 下半段"，会舍入；改成"下半段 = 1 − 上半段"）—— 副本试验显示 13 组配置 RULE3 全归零、
点数不变、坐标最多移动 6e-8，但**必须跑整簇回归**再决定是否落地。

---

## 第二十九轮（2026-09-12）：`tc_barriers` 的 IR 合法性核查 —— 并**更正**第 26 轮的结论

问题：`barrier_guarded_read_write_calls` 生成的 IR 是否合法？

### 结论一：IR 合法（可复现的核验）

用 `MGL_DUMP_IR=1` 抓该用例的全部 stage 模块（7 个：fs / TCS / TES-compute / VS / VS(capture) / kernel / fs），
TCS 模块 = 含 `air.wg.barrier` 的那个、`!air.kernel` 且 buffers 为
`tcs_stage_in(24) / tess_factors(26) / tcs_patch_out(27) / tcs_stage_out(28) / tcs_indirect(29)`：

```llvm
define void @main(i8 addrspace(1)* …5 个 buffer…, <3 x i32> %5 /*thread_position_in_threadgroup*/,
                  <3 x i32> %6 /*threadgroup_position_in_grid*/)
entry:
  %7 = extractelement <3 x i32> %5, i64 0        ; gl_InvocationID
  %8 = srem i32 %7, 2
  …
if.end:                                            ; ← 归并块
  call void @air.wg.barrier(i32 3, i32 1) #0       ; flags=1|2 device|threadgroup
…
if.end2:                                           ; ← 第二个归并块
  call void @air.wg.barrier(i32 3, i32 1) #0
…
  ret void                                         ; 全模块**只有一处** ret
```

核验项与依据：

| 检查 | 结果 |
|---|---|
| LLVM verifier（`opt -passes=verify`） | **通过**（仅需把 `air64_v28` triple 换成已知 triple 才能被 Homebrew 的 `opt` 解析；模块本身无诊断） |
| barrier 是否在**统一控制流** | 是：`entry` 无条件汇入 `if.end`／`if.end2`，两处 barrier 都在归并块；全模块只有一处 `ret void`，barrier 之前没有任何提前退出 |
| barrier 语义 | `air.wg.barrier(i32 3, i32 1)`：mem_flags = 1\|2（device\|threadgroup），与「跨 invocation 的 per-vertex 输出存在 **device** buffer（addrspace(1)）」一致；见 `mgl_air_backend.cpp:5993-6008` |
| dispatch 与 workgroup 是否匹配 | `mgl_draw_tess.cpp:433` (`mglTessAppendTCSCoreBindings`)：`groups={patch_count,1,1}`、`local={tcs_out_vertices=16,1,1}` → **一个 patch 一个 threadgroup、恰好 16 个线程**，threadgroup barrier 精确同步本 patch 的全部 invocation |
| 跨 invocation 存储 | `tcs_data[gl_InvocationID]`：per-vertex 记录 stride 128B（`shl 7`）+ 槽内偏移 112；`tcs_patch_result[gl_InvocationID]`：per-patch 256B（`shl 8`）+ invocation×16；load/store 均 `align 4` |

而且 **barrier 排序后的结果本身是对的**：GPU 侧 XFB 临时缓冲（`MGL_AIR_TESS_SLOT_XFB_OUT`，slot 31）里
装的是未解码的 float carrier，解出来正是期望值 `8 / 32 / 56 / 80 …`（即
`tcs_patch_result[*]` 的两阶段累加结果），证明「写→barrier→读→barrier→再写/读」这条跨 invocation 通路工作正常。

### 结论二：该用例失败**不是** IR/barrier 问题，也**不是** CTS 侧账目 —— 是 XFB 回读路径的缓冲一致性问题

第 26 轮曾把本用例判为「CTS `m_n_result_vertices` 按 16 槽/patch 记账、与驱动无关」。本次用
`MGL_XFB_DUMP`（临时桩，打印 CTS 读到的 XFB 流）逐点复核后，这个结论**是错的**，更正如下：

失败信息（`esextcTessellationShaderBarrier.cpp:984`）每次都是同一个 patch 分组的第 1 个 varying：

```
Result data for patch [3]: (1006648320, 1065353216, 0, 1065353216, 32, 0, 0, 0),
                  expected: 8, 0, 0, 0, 32, 0, 0, 0)
```

`1006648320 = 0x3C004000`、`1065353216 = 0x3F800000` → 按 float 读是 `(0.0078278, 1, 0, 1)`，**是个位置样式的向量**，
而不是本用例任何计算产生的值（本用例只产生 8/32/56/80 这些整数 carrier）。

逐层定位（均为临时桩，事后已全部还原）：

| 层 | 观测 |
|---|---|
| GPU 写的 XFB 临时缓冲（slot 31） | **干净**：整段扫描 `0x3C004000` 命中 0；里面是 8.0f/32.0f/56.0f/80.0f carrier，值正确 |
| CPU 打包结果 `packed` + CPU shadow + live Metal 视图（发布当时） | **干净**：`junk packed=0 shadow=0 live=0`，且 `mismatch shadow=0 live=0`（整段 4800 int 逐个比对） |
| CTS 实际 `glMapBufferRange` 读到的内容 | **脏**：某些 patch 的 varying 槽里出现 `0x3C004000 0x3F800000 0 0x3F800000`；**每次运行脏的 patch/槽都不同**（v04-v05、v06-v09、v40-v43…），非确定性 |
| 在 `mglMapBufferRange` 入口再看 | CPU shadow 仍**正确且无脏**，而 Metal 侧视图与 shadow 有 **1500 个 int 不一致**（其前 64 int 全 0）→ 映射出去的数据来自 Metal 侧，而不是 CPU shadow |

⇒ 结论：**数据在「发布之后、回读之前」被换掉/覆盖**，属于 MGL 缓冲 copy-on-write / snapshot 机制一侧的问题
（`MGLRenderer+Tessellation.m` 里那段注释本身也承认这个脆弱点：「SubData may land in a snapshot while
glMapBufferRange serves the CPU shadow」）。这也解释了为什么 `MGL_DISABLE_DRAW_DEFER=1` 不改变现象。

### 未结项与下一步（建议）

1. 在 `mglMapBufferRange` 的 flush（`mglFlushCommandBuffer` + `mglRendererFlush(ctx,true)`）**前后**分别读
   Metal 侧与 shadow，定位是哪一步把 shadow 的正确数据换成 Metal 侧的旧内容（当前证据指向
   CoW snapshot/代数切换，而不是 GPU 乱序：flush 已 `waitUntilCompleted`）。
2. 检查 XFB 目标缓冲是否与 tess 的临时/scratch 分配**别名**：比较
   `mglTessCreateBuffer(..., SHARED)` 出来的临时 XFB 与目标 GL buffer 的 Metal 分配/区间；
   若别名，则应在发布前后保证互斥，或让发布走 `mglRendererBufferSubData` 的快照路径而不是直接 memcpy 到 live 视图。
3. 复现命令（本次全部实跑）：

```
cd /Users/fterward/VK-GL-CTS-build-mgl-target/external/openglcts/modules
MGL_DUMP_IR=1 DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts \
  --deqp-case=KHR-GL46.tessellation_shader.tessellation_shader_tc_barriers.barrier_guarded_read_write_calls \
  --deqp-surface-type=fbo --deqp-surface-width=256 --deqp-surface-height=256 --deqp-visibility=hidden \
  --deqp-log-images=disable --deqp-log-shader-sources=disable --deqp-log-decompiled-spirv=disable \
  --deqp-log-filename=/tmp/cts-tcb.qpa 2>/tmp/ir_dump.txt
python3 - <<'EOF'   # 切出 TCS 模块再做 verifier 检查
lines=open('/tmp/ir_dump.txt',errors='replace').read().split('\n')
open('/tmp/tcs_module.ll','w').write('\n'.join(lines[42:212]))
EOF
sed 's/^target triple = .*/target triple = "arm64-apple-macosx26.0.0"/' /tmp/tcs_module.ll > /tmp/tcs_module_arm.ll
/opt/homebrew/opt/llvm@15/bin/opt -passes=verify -disable-output /tmp/tcs_module_arm.ll   # exit 0
```

（MGL 侧与 CTS 侧的临时桩均已 `git checkout` 还原；`glcts` 已用 pristine 源码重建。）

## 第三十轮（2026-09-12）：`fractional_odd_spacing` 的 IR 合法性核查

问题：`vertex.vertex_spacing_primitive_mode_quads_vs_mode_fractional_odd_spacing` 生成的 IR 是否合法？

**结论：合法。** 该用例每个 run 编译 7 个模块（fs / VS / TCS kernel / TES kernel / XFB-capture VS / readback VS / fs…），
逐类抽查（fragment / vertex / kernel，以及 quads+FO 程序的 TCS 与 TES kernel 单独抽出）全部通过
`opt -passes=verify`（exit 0）：

| 检查 | 结果 |
|---|---|
| `opt -passes=verify`（fragment / vertex / kernel / TCS / TES kernel） | **全部 exit 0**（同样只需把 `air64_v28` triple 换成已知 triple 供 Homebrew opt 解析） |
| `air.patch` / `air.patch_control_point` | 全 dump **0 次** → 本用例走 TES compute 展开，不涉 native post-tessellation ABI（与 `MGL TRACE tess path … exec=2` 一致） |
| `air.wg.barrier` | 全 dump **0 次** → 该用例着色器不用 `barrier()`；TES kernel 的 `tesk_oob` 提前 `ret` 因此合法（无 barrier 才允许） |
| TCS 参数/元数据一致性 | `!0` 的 8 个 arg 节点与签名一一对应：UBO(location_index 0，struct info = `float2 inner_tess_level` + `float4 outer_tess_level`) + 5 个 buffer(24/26/27/28/29) + `uint3 thread_position_in_threadgroup` + `uint3 threadgroup_position_in_grid` |
| 访问对齐 | float `align 4`、half `align 2`、`<2 x float>` `align 8`、`<4 x float>` `align 16`，均正确 |

**关键一条（排除"精度把级别弄坏"这一可能）**：TCS 对每个 tessellation factor 写**两份**——
`half`（Metal native 因子路径）和**精确 float32**（per-patch 36B 记录内 +12/+16/+20/+24(outer)、+28/+32(inner)）；
而域生成读的正是精确 float 那份（`mgl_render.cpp` 的 `tessDomainInput()` 从
`MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET` memcpy → `mglTessGenerateDomainStrided`）。
所以 `inner=(32,64)` 到生成器时仍是精确的 32 / 64（FO 取整 33 / clamp 63），**不存在 half 往返把级别改掉的问题**。

**另一条（把"pipeline 输出"与"离线模型"钉在一起）**：TES kernel 从种子记录（stride 128B，前三个 float = TessCoord）
读入坐标后再落盘输出 → CTS 回读到的坐标就是 `mgl_tess_domain_gen.c` 生成的坐标；
这与离线对拍结果一致（CTS 报 `found: 31` = 离线模拟的 31，外边 `{2,31}` 也一致）。

⇒ 与本轮早先的结论一致（§4）：本用例失败不是 IR/精度/管线问题，而是 CTS 的 FO 分支期望
`FOround(clamp(L-2))-2` 与其自身 equal 分支期望的 `m-2`（网格模型）互相矛盾
（equal 144/144 通过 vs FO 48/48 被拒），**不写 CTS 特判**。

复现：

```
MGL_DUMP_IR=1 DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts \
  --deqp-case=KHR-GL46.tessellation_shader.vertex.vertex_spacing_primitive_mode_quads_vs_mode_fractional_odd_spacing \
  --deqp-surface-type=fbo --deqp-surface-width=256 --deqp-surface-height=256 --deqp-visibility=hidden \
  --deqp-log-images=disable --deqp-log-shader-sources=disable --deqp-log-decompiled-spirv=disable \
  --deqp-log-filename=/tmp/cts-fo.qpa 2>/tmp/ir_fo.txt      # 逐模块切出后跑 opt -passes=verify
```

---

## 第三十一轮（2026-09-12）：`tc_barriers` 真因定位并修复 —— 并**第二次更正**结论

用户要求先修 `tessellation_shader_tc_barriers.barrier_guarded_read_write_calls`。修好了：
提交 `2e55500`，三个子用例反复跑全 **Pass**，TESS 簇 **138 pass / 1 fail / 1 ns**（此前 137/2/1）。

### 真因：per-patch 记录的 stride 算法漏算"占位跨度"，patch-out 缓冲被少分配 16 倍

`MGL/include/mgl_air_tess_abi.h` 的 `mglAIRPatchVaryingStride()` 只用**基址 location** 算 stride
（`(location + 1) * 16`），而 codegen 侧（`stageRecordStride()` → `varyingLocationSpan()`）是按
**资源跨占的 location 数**算 stride。测试的 TCS 声明 `patch out int tcs_patch_result[16];`：

| | 值 |
|---|---|
| codegen 写 stride（TCS `tcs_patch_out` 槽 27） | `patch*256 + inv*16`（IR 里 `%42 = shl %41, 8` 已证实） |
| TES 读 stride（`tes_patch_inputs` 槽 27） | 同一 256（IR 里 `%24 = shl %20, 8`） |
| **缓冲分配**（`mglAIRPatchVaryingStride`） | **16**/patch → 20 patch 只分配 **320 B**（应为 5120 B） |

⇒ patch≥1 的 16 槽整块写到缓冲**之外**（落进相邻分配 ✓），TES 再按 256 stride 读 → 整块 patch 拿到
**别人没写过/是别的记录（含 position 字样向量）的内存** ✓✓，且随池化布局每次不同 ✓✓；同时 XFB 只发布了
10 个 instance 中的 8~9 个 ✓。

### 证据链（全部实跑，临时桩事后已全部还原）

| 观测点 | 修复前 | 修复后 |
|---|---|---|
| `tcs_patch_out` 缓冲长度 | **320 B**（20 patch × 16） | **5120 B**（20 × 256）✓ |
| XFB `packed` 里坏顶点数 / 覆盖顶点数 | `packedBadVerts=40(first=6)`，`vertices=320~360`（少 1~2 个 instance） | **`packedBadVerts=0`，`vertices=400`** ✓ |
| GPU 侧 temp XFB 流 | 干净（8.0f/32.0f/56.0f/80.0f carrier 正确） | 干净 ✓ |
| CTS 用例 | Fail（`Invalid data captured`，每次坏 patch 不同） | **Pass × 多次** ✓ |
| 三个子用例 | 只有 read_write 失败 | 三个全 Pass ✓ |

定位路径：`MGL_DUMP_IR` 抓 IR → 确认 TCS 写/ TES 读都按 256 stride（`shl …,8`）→ 在 TCS 分发后回读
patch-out 缓冲发现其长度只有 320 B、且 patch 数据只覆盖开头 → 比对 `mglAIRPatchVaryingStride`
（16）与 `stageRecordStride`（256）→ 单行修复。

### 结论更正（第二次）

- 第 26 轮判"CTS `m_n_result_vertices` 记账问题"✗ **错**（CTS 的分组/步长与 MGL 的流一致，检查是正当的）。
- 第 29 轮判"发布后 Metal 侧缓冲被换掉（CoW/snapshot）"✗ **是症状不是病因**：TES 记录本身就是被
  TCS 越界写坏的；越界写发生在 TCS 分发里，恰好落在相邻的 record/XFB 分配上。
- 新写入的 IR/链路观察（本轮实测，未改动）：TCS 与 TES 分属**不同 compute plan/encoder**
  （`newTCSStageInBufferForContext:` vs `dispatchAIRTessEvalCompute:`），跨 encoder 的可见性由 Metal
  保证 ✓；而 `mglRenderEncodeComputePlan` 只在**最后一个 dispatch 之后**发一次 `memoryBarrier`，
  *同一 plan 内多 dispatch 之间*不发 —— 本轮实测与该用例无关（改与不改都通过），作为独立项留待评估。

### 同类潜在问题（未改，留待评估）

1. `contract.patch_out_stride` 在 compute 路径被硬编码为 `16u`（`mgl_draw_tess.cpp:140`），
   只有 native / render-vertex 两条绑定路径读它（`mgl_draw_tess.cpp:312`,
   `MGLRenderer+Tessellation.m:1049`）。这些路径目前全绿，且它们的 patch 记录布局由 native ABI 决定，
   未做改动以免引入不确定风险。
2. `mglAIRPerVertexStrideForResources()`（逐顶点版）**已经**算了跨度 ✓ —— 本次修复就是让 per-patch 版
   与它一致 ✓。

复现命令：

```
cd /Users/fterward/VK-GL-CTS-build-mgl-target/external/openglcts/modules
DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts \
  --deqp-case=KHR-GL46.tessellation_shader.tessellation_shader_tc_barriers.barrier_guarded_read_write_calls \
  --deqp-surface-type=fbo --deqp-surface-width=256 --deqp-surface-height=256 --deqp-visibility=hidden \
  --deqp-log-images=disable --deqp-log-shader-sources=disable --deqp-log-decompiled-spirv=disable \
  --deqp-log-filename=/tmp/cts-tcb.qpa
```

---

## 第三十二轮（2026-09-12）：`max_in_out_attributes` 成套放开到 128 分量 —— NotSupported → **Pass**

用户要求按"成套方案"做，并把 location 31 保留位与 `MAX_ATTRIBS` 双用途一并解决。结果：
**TESS 簇 139 pass / 1 fail / 0 not_supported**（此前 138/1/1），`max_in_out_attributes` 从 NotSupported 变成 **Pass**。
提交：`1479bb0`（XFB 表解耦）、`9e9d237`（保留位 31→32）、`088394c`（限额 + 常量 + 块数组成员 + XFB 元素索引）。

### 1. `MAX_ATTRIBS` 双用途解耦（`1479bb0`）

`MAX_ATTRIBS(30)` 既是"顶点属性预算"（上报为 `GL_MAX_VERTEX_ATTRIBS`，受 Metal vertex descriptor 约束）
又被当作"TF varying 预算"用在：`transform_feedback_varying_names[30][96]`、`transform_feedback_layout[30]`、
`glTransformFeedbackVaryings` 的 `count <= MAX_ATTRIBS`、`draw_buffers.c` 的 XFB draw 守卫、
`MGLXfbVsPlan.fields[30]`、`MGL_AIR_GS_XFB_MAX_FIELDS(30)`。
新增 `MGL_MAX_TRANSFORM_FEEDBACK_VARYINGS(64)`（GL 4.6 §11.1.1：TF 配置只受"分量预算"限制，
128 分量 ⇒ 合法可命名 32 个 vec4 varying），把上述 XFB 表/校验全部改用它，顶点属性口径不动。

### 2. 保留位 location 31 → 32（`9e9d237`）

`MGL_AIR_PRIMITIVE_ID_LOCATION` 是 GS-passthrough 把 gl_PrimitiveID 送到 FS 的**内部接口标签**（值本身走
per-vertex record 的 `MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET`）。报 64（16 location）时它与应用无冲突；
抬到 128（location 0..31）后应用**可以**合法使用 location 31 ⇒ 移到 **32**（应用可见范围之外）。不产生额外
record 宽度、不额外占用 stage-out 槽。

### 3. 三道门 + 三个实现缺口（`088394c`）

| 关卡 | 现状 | 处置 |
|---|---|---|
| ① `MAX_VERTEX_OUTPUT_COMPONENTS ≥ MAX_TESS_CONTROL_INPUT_COMPONENTS` | 64 < 128 | `glm_params` 报到 **128**（规格下限 64，允许） |
| ② `MAX_TESS_CONTROL_OUTPUT == MAX_TESS_EVALUATION_INPUT` | 128 == 128 ✓ | — |
| ③ `MAX_TESS_EVALUATION_OUTPUT ≤ MAX_TRANSFORM_FEEDBACK_INTERLEAVED` | 128 > 64 | TF interleaved 报到 **128** |
| ④ 32 个 TF varying | `MAX_ATTRIBS` 卡 30 | 见 §1 解耦 |
| ⑤ 着色器里 `gl_Max*` 常量 | parser/sema/AIR 都没有 | 三处补齐（与 glm_params 对齐）：VertexOutput / FragmentInput / TessControl{Input,Output} / TessEvaluation{Input,Output} / TessPatch / PatchVertices / TessGenLevel / TransformFeedback{Interleaved,Separate}* / VertexStreams / Viewports |
| ⑥ `Block.member[i]`（块实例按 invocation 索引） | TCS/TES 只有"非数组成员"支持 | 符号分类不再把**块成员**的数组形状剥掉（剥掉只对普通 per-vertex 数组正确，那种数组本身就是 invocation 维度）；新增 `emitTessBlockArrayLoad/Store`：记录槽 = `成员 location + 元素`，TCS in/out 与 TES 输入都走它 |
| ⑦ TES 输出块数组成员写回 record | `lvalues` 里没有该聚合 ⇒ 写 undef（捕获全 0） | TES 输出成员**直写 stage-out record**；`+=` 从同一槽读旧值；record 汇总阶段跳过这些槽（`directRecordVaryings`） |
| ⑧ TF 名字里的 `[n]` | 解析后丢弃元素索引 ⇒ 31 个字段全解析到元素 0 | 暴露 `mglTransformFeedbackArrayElement()`，tess packer 与 GS scatter 的源偏移都加上元素（`location + element`） |

**⑧ 的定位证据**：CTS 报 `captured (59520,60513,61506,62499)` vs `expected (59524,60517,61510,62503)` —— 差恒为 +4，
即"第 1 个及以后的字段重复了第 0 个"。把 CTS 的 `initReferenceValues` 逻辑离线重算 + 临时桩打印 CTS 自己的
`sumInTES/ref` 证实：**CTS 的 ref[0..3] 与 MGL 捕获完全一致**（`ref: 59520 60513 61506 62499 59524 ...`），
真正的分歧在 i=4 之后 ⇒ 元素索引丢失。桩事后已还原，CTS 源码树干净。

### 验证（全部实跑）

| 用例簇 | 之前 | 现在 |
|---|---|---|
| `KHR-GL46.tessellation_shader`（140） | 138 pass / 1 fail / **1 ns** | **139 pass / 1 fail / 0 ns** |
| `KHR-GL46.geometry_shader.*`（136） | 136 / 0 | **136 / 0** |
| GL46 hotspot 全量（1328） | 1270 pass / 52 fail / 1 crash / 4 ns | **非通过集合逐条相同**（无回归） |
| `make test-tess-domain` | ok | ok |
| 单例复验 | — | `max_in_out_attributes` **Pass**；`tc_barriers`×3 **Pass**；`invariance_rule4` **Pass** |

剩余 1 例仍是 `vertex_spacing_..._fractional_odd_spacing`（第 28/30 轮已证为 CTS 自身期望矛盾，离线 48/48 被拒
vs equal 144/144 通过），未写特判。

### 附带观察（未改）

- 同一 plan 内多 dispatch 之间仍不插 `memoryBarrier`（只在最后一个 dispatch 后发一次）。本轮实测与本用例无关
  （TCS/TES 分属不同 plan/encoder，跨 encoder 由 Metal 保证可见性），留作独立评估项。
- `contract.patch_out_stride` 在 compute 路径硬编码 `16u`，仅 native / render-vertex 绑定路径读取；本周期的
  per-patch stride 修复（`2e55500`）未触及该字段，仍待评估。

---

## 第三十三轮（2026-09-12）：三项残留风险成套清理

第 32 轮「附带观察」的两项 + native 控制点取值的数组维度，一次清完。三项都不是特判，全部落在
ABI / 记录布局不变式上。

### 1. 跨 dispatch 的 `memoryBarrier` 显式化（`mgl_render.h` / `mgl_render.cpp` / `mgl_draw_tess.cpp`）

- 现状：`MGLRenderComputeExecutionPlan` 只有 `barrier_scope`（末次 dispatch 之后、`endEncoding` 之前
  发一次 fence），同一 encoder 内多个 dispatch 之间**没有**任何 fence。
- Metal 语义：同一 encoder 内 dispatch 只保证**执行顺序**；前一个 dispatch 的内存写入要能被后一个看见
  必须显式 `memoryBarrier`。
- 处理：新增 `dispatch_barrier_scope`（同一套 value-state 位域，`NONE=0` 表示"本 plan 断言各 dispatch
  内存无关"）。encoder 在重放第 k>0 个 dispatch 前按需发 fence。
  `mglTessAppendEvalPerPatchDispatches` 的不变式写在函数头：每个 dispatch 写
  `[inst * items_per_instance + patchBases[p], +items_p)`，且只读本 encoder 之前写入的数据；
  `patchBases` 是逐 patch item 数的前缀和 ⇒ 区间天然不相交。另加防御性判据：若
  `patchBases[patch_count] > items_per_instance`（调用方 stride 口径被破坏、区间可能重叠），
  自动置 `MGL_RENDER_COMPUTE_BARRIER_BUFFERS`，把"静默竞态"降级为"多一次 fence"。
- 树内多 dispatch 的 plan 只有 TES per-patch 一处，因此默认路径零额外开销。

### 2. `contract.patch_out_stride` 不再硬编码（`mgl_draw_tess.cpp`）

原 `contract->patch_out_stride = 16u;` → 改为按 **TCS patch-out 记录跨度**计算：

```c
contract->patch_out_stride = mglTessNativePatchOutStride(
    tcs != NULL,
    tcs ? mglAIRPatchVaryingStride(
              &tcs->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES])
        : 0u);
```

无 TCS 时不存在 patch 输入，保持 16 字节最小记录。消费方是 TES-vertex 渲染路径的 patch 输入绑定
（`MGLRenderer+Tessellation.m`），此前 patch > 0 时会绑到错误偏移。

### 3. native 后细分的控制点数组成员（数组 / 矩阵）—— 三个真因

`in vec4 v[][2]` 在 TES 侧占**每个控制点 2 个 location**，但链路三处都按"一个成员一个 slot"处理：

1. **反射丢维度**：`gl_array_size` 对 TES 输入保留的是 *invocation*（控制点）维度——探针实测
   `in vec4 cp[]` 与 `in vec4 v[][2]` 都报 `array_size=0, dims=1`——宿主侧无法算出成员跨度。
   新增 `MGLShaderResource::gl_element_array_size`（invocation 维度**内**的数组长度，0 = 无），
   按 `mgl_air_varsym.cpp` 同一条剥离规则填充（TCS in/out、TES in、GS in；block 成员与 patch 变量不剥离）。
   同时新增 `mglAIRResourceLocationSpan()`，per-vertex stride 与 patch stride 两个宿主侧跨度计算改用它。
2. **顶点描述符只声明基址**：`mglTessPlanNativeVertexDescriptor` 每个资源只发一个 attribute
   （`location + 1`）。改为**一个 location 一个 attribute**，格式取
   `mglRenderTessControlPointLocationFormat()`（矩阵取列 = `rows` 分量），偏移 `112 + (location+k)*16`。
3. **捕获变体只写元素 0**：tess-capture 的存储循环把"普通数组"当成 per-vertex 维度，只
   `extractvalue 0` 后写一个 slot。IR 实证：`getelementptr … 112` + 单个 `<4 x float>` store，
   而同一模块的记录 `air.arg_type_size` 已经是 144（即记录里有第二个 location 但没人写）。
   现在数组/矩阵按 location 逐个写。

配套的 IR ABI 一致性：

- `air.patch_control_point_input` 与 getter 结构体改为**一个 location 一个字段/条目**（元素类型名）。
  聚合类型会被 Metal 直接拒绝：`Failed to create pipeline state: Unsupported attribute type`。
- 读取侧新增 `extractControlPointField()`：把逐 location 字段重新组装成聚合值，`v[i][j]` 继续走通用
  索引路径（常量 `extractvalue` / 动态 select 链）。
- TCS 逐顶点输出补上 `out[invocation][element] = …` 及读取：此前该形状落到通用赋值路径并静默失败
  （`codegen: unsupported construct`）。location 相邻 ⇒ 动态元素下标直接加在 location 上，
  无需分支；"数组的矩阵"（需要元素 + 列两级下标才能定位一个 location）显式报错而不是错址。

### 新增回归（`test_regression/main.c`）

`air_tessellation_control_point_array`：VS 逐控制点写 `v[0].x = 0.1 + 0.2*cp`、`v[1].y = 0.1 + 0.2*cp`，
TES 读 `v[0][0]` / `v[0][1]` / `v[gl_PrimitiveID & 1][1]`（两个 patch）并由像素读回校验，覆盖
**TES-only 捕获原生路径 / TCS 原生路径 / 强制 compute 展开**（TES 读 `gl_TessLevelOuter` 触发）三条路，
路径选择由 `MGL_TESS_PATH_TRACE` 的 `exec=1/1/2` 三行实证。

### 验证（全部实跑）

| 用例簇 | 之前 | 现在 |
|---|---|---|
| `KHR-GL46.tessellation_shader`（140） | 139 pass / 1 fail / 0 ns | **139 pass / 1 fail / 0 ns**（同一 FO-spacing 例） |
| `KHR-GL46.geometry_shader.*`（136） | 136 / 0 | **136 / 0** |
| GL46 hotspot 全量（1328） | 1270 pass / 52 fail / 1 crash / 4 ns | **1270 pass / 52 fail / 1 crash / 4 ns，非通过集合逐条 diff 为空** |
| `make test-tess-domain` / `test-tess-air` | ok / 180 | **ok / 180** |
| `test_regression` tess + GS 子集（25 例） | 24 Pass + `isolines_multidraw` 失败 | **25 Pass / 0 fail**（见下节） |

### 4. 既有失败 `air_tessellation_isolines_multidraw` 一并收掉

排查起点：该用例在基线 `088394c` 上同样失败（stash 复验），`MGL_TES_VERTEX_RENDER=0` 时通过；
`MGL_TESS_PATH_TRACE` 显示同一 program 的两次绘制分别是 `exec=4 capture=1`（arrays 段，TES-vertex）
与 `exec=2 capture=3`（elements 段，索引化回退 compute）。

真因：**两条栅格化路线共用了一个 PSO 缓存键**。pipeline 的 `primaryKey`
（`MGLRenderer+RenderPass.m`）折入了 `nativeTESActive / tessVertexCaptureActive /
geometry.expansionActive / cullDistanceCaptureActive / tessComputeActive`，但**不含**
决定"栅格化顶点函数取哪一个"的 `tessVertexRenderActive`：为真时顶点函数是 TES 自身的 render-vertex
函数，为假时是生成的 slot-28 记录 passthrough（见 `tessPassthroughFunction` 选择处）。同一 program
第二次绘制因此命中第一次的 PSO，用 render-vertex ABI 去读 compute 写入的记录流，探测点自然没有像素。

修复：把 `tessVertexRenderActive` 折进 `primaryKey`（bit 18；19–23 已被占用）。

修复后：该用例 **Pass**；本地 tess+GS 子集 25/25 全绿；CTS tess 139/1/0、GS 136/0/0、
hotspot 非通过集合与基线逐条 diff 仍为空。

### 顺带清理

`mgl_draw_tess.cpp` 中每条 tess draw 都无条件 `fprintf` 的路径 trace（`587732a` 引入）改为
`MGL_TESS_PATH_TRACE` 门控，信息不变、默认无噪声。

---

## 复现 ground truth 的方法（本轮已脚本化，可复用）

CTS 源码树 `/Users/fterward/VK-GL-CTS`（git 干净）。改一个文件：
`external/openglcts/modules/glesext/tessellation_shader/esextcTessellationShaderVertexSpacing.cpp`

1. 在 `iterate()` 的 `verifyEdges(edges, run);` 处改为：
   若 `getenv("MGL_CTS_TRACE_ALL")` 非空，则遍历 `edges` 打印
   `n_edge / tess_level / clamped / rounded / npts / exp / first / last` 并**跳过 verifyEdges**
   —— 这样一次运行就能覆盖全部配置而不早退（`verifyEdges` 遇到第一个不符就抛异常）。
2. 在 `verifyEdges` 内加 `MGL_CTS_TRACE_EDGES` 桩，可打印该边**全部点坐标**。
3. 重建（增量，几十秒）：
   ```
   cd /Users/fterward/VK-GL-CTS-build-mgl-target && ninja glcts
   ```
4. 运行：
   ```
   cd /Users/fterward/VK-GL-CTS-build-mgl-target/external/openglcts/modules
   MGL_CTS_TRACE_ALL=1 DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts \
     --deqp-case=KHR-GL46.tessellation_shader.vertex.vertex_spacing_primitive_mode_quads_vs_mode_fractional_odd_spacing \
     --deqp-surface-type=fbo --deqp-surface-width=256 --deqp-surface-height=256 \
     --deqp-visibility=hidden --deqp-log-images=disable --deqp-log-shader-sources=disable \
     --deqp-log-decompiled-spirv=disable --deqp-log-filename=/tmp/cts.qpa 2>&1 | grep MGLCTSEDGE
   ```
5. 查完还原：`cd /Users/fterward/VK-GL-CTS && git checkout -- <该文件> && cd <build> && ninja glcts`

**注意**：改 MGL 生成器做实验时若引入越界（例如让 `nx_seg` 过小导致
`edge_position` 索引越界）会直接 SIGSEGV 且看不到任何 trace，先确认 MGL 没崩再看 CTS 输出。

---

## 建议落地顺序

1. **`xfb`（#4）**——纯校验层，独立、零 tessellator 风险。
2. **`invariance_rule4`（#2）**——域生成对称性，改动小、有 dbg16 回归网（outer 3..15 / inner 1..7 全扫）。
3. **`inner_tessellation_level_rounding`（#3）与 `fractional_odd`（#1）**——都动 `generate_quads` 内层拓扑，
   建议合并审视、先建离线 Python 参照模拟器对拍 76 例全绿子组再改 C。
4. `point_rendering` / `tc_barriers`——需 Metal 侧 trace，单独排期。
5. `max_in_out_attributes`——非规格违规，抬 `MAX_VERTEX_OUTPUT_COMPONENTS` 口径需单独决策。

> ✏️ **2026-09-12 状态**：1–4 项除 `fractional_odd` 外均已落地（簇 134/5 → **137/2**）。
> 当前优先级重排：**① `tc_barriers`（#6）——"CTS 侧账目"结论已作废，重开为驱动侧 XFB 缺陷，
> 见「复核」一节的三条探针**；② `fractional_odd`（#1）——第 28 轮判定为 CTS 侧账目矛盾，
> 若日后要推翻该判定，需先给出"equal 分支与 FO 分支期望可同时满足"的实现方案；
> ③ `max_in_out_attributes`——limits 口径决策。
