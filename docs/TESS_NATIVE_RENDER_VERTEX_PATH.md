# TESS isolines / point_mode Metal 原生化设计(render vertex 路径)

日期:2026-09-09(2026-09-10 更新验证结果)
状态:**已实施 —— §8 步骤 1–5 已落地,回退开关全绿;render PSO `materializeAll` 阻塞已定位并修复(见 §10.1);§10.3 的 3 个 TES-vertex 功能正确性 bug 已全部修复,`test_regression` 全量 93 项 = 91 PASS / 0 FAIL / 2 SKIP**
范围:第一版 = isolines + point_mode(非 XFB、非 GS);XFB / GS-after-TES / uses_tess_level-fill 保留 compute(见 §9 路线图)

## 1. 背景与动机

GL_PATCHES draw 当前有两条 tessellation 路径:

| 路径 | 覆盖 | TES 形态 |
|------|------|----------|
| Metal 原生 | triangles / quads fill(无 XFB/GS) | post-tessellation vertex function,Metal tessellator 喂顶点 |
| AIR TES compute expansion | isolines / point_mode / XFB / GS / uses_tess_level | compute kernel,自算域坐标,输出记录 + passthrough VS |

Metal tessellator 只产 triangle/quad patch 的三角形填充,没有 isolines patch 类型和 point 输出拓扑
([mgl_air_backend.cpp:9572-9583](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_air_backend.cpp#L9572)),
因此 isolines/point_mode 无法走 post-tessellation 原生路径,一直由 compute expansion 承载。

compute expansion 的结构性开销(P0 修复后 93 回归全绿,功能正确,但性能与路径复杂度不佳):

1. **每 patch 每 instance 一次 compute dispatch**
   ([mgl_draw_tess.cpp:1339-1361](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_draw_tess.cpp#L1339)
   `mglTessAppendEvalPerPatchDispatches` 在 per-patch 循环内 append dispatch;patch 数线性放大
   compute command 数量);
2. **compute/render encoder 切换**
   ([MGLRenderer+Tessellation.m:1208-1211](file:///Users/fterward/MGL-minecraft/MGL/src/MGLRenderer+Tessellation.m#L1208)
   dispatch 前 endRenderEncoding,之后重建 render encoder);
3. **输出记录中转**:TES kernel 每项写一条记录(slot 28),passthrough VS
   ([MGLRenderer+RenderPass.m:1151](file:///Users/fterward/MGL-minecraft/MGL/src/MGLRenderer+RenderPass.m#L1151))
   再从记录读回光栅化 —— 求值结果在 GPU 内存兜一圈。

## 2. 核心设计

把 isolines/point_mode 的 TES 编译成 **render pipeline 的普通 vertex function**(下称 TES-vertex),
用 CPU 预 seed 的域坐标记录驱动,直接光栅化点/线。

### 2.1 可行性依据(codegen 复用度)

TES compute kernel 的关键 lowering 已与"kernel 入口"解耦,可直接搬到 vertex 入口:

| kernel 机制 | 代码位置 | vertex 形态复用 |
|---|---|---|
| TessCoord 从 seed 记录 `position.xyz` 读 | [mgl_air_backend.cpp:11537-11553](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_air_backend.cpp#L11537) | 完全复用;记录 index 从 `gl_VertexID` 取(Metal vertex_id 含 drawPrimitives 的 vertexStart) |
| gl_TessLevel 从 factor 记录 float32 精确区读(CTS 1e-5 精度) | 11503-11536 | 完全复用;patch_id 从 slot 29 setBytes 读 |
| gl_in / patch_out 从 slot 30/27 指针读 | 2792 段 | 完全复用;dispatch 端按 patch 偏移绑定 |
| 每 thread 写一条记录 | 11563 段(`storeTessComputeVaryings`) | **替换**为 vertex 输出(§4.3) |

CPU 域展开(seed)通路已存在:
[mglTessSeedEvalOutputRecords](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_draw_tess.cpp#L803)
基于规范环算法域生成
([mgl_tess_domain_gen.c](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_tess_domain_gen.c),
golden `{1,6,13,24,37}` 由 test_tess_domain 锁定)。

### 2.2 新路径链路

```
TCS compute dispatch(不变)→ 控制点记录(slot 28)+ factor buffer(slot 26)
CPU 域展开(不变)→ seed TessCoord 全量记录(slot 28,只读)
render encoder(单编码器,全程无切换):
  for instance:
    for patch:
      setRenderBuffer  slot 26 ← factor + p*36
      setRenderBuffer  slot 27 ← patch_out + p*stride
      setRenderBuffer  slot 30 ← gl_in + patch 偏移(indexed: gather 流)
      setBytes         slot 29 ← {patch_id, gl_in_vertices, items, 0}
      drawPrimitives(topology, vertexStart=patchBases[p], count=items,
                     instanceCount=1, baseInstance=instance)
  topology = point_mode ? MTLPrimitiveTypePoint : MTLPrimitiveTypeLine(isolines)
```

消除:TES compute dispatch(每 patch 一次)、encoder 切换、输出记录写回、passthrough VS。

### 2.3 顶点流语义

- **point_mode**:每记录一点;items 数 = 域顶点数;
- **isolines**:记录两两成对为线段 —— 域生成 emit 顺序
  ([mgl_tess_domain_gen.c:196-206](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_tess_domain_gen.c#L196))
  恰为 line-list 语义,无需重排;
- **indexed / indirect / multi-draw / instanced / rasterizer_discard**:contract 字段
  (patch_vertices/items/instance)与 `mglTessFillEvalPatchItemBases` 已覆盖;rasterizer_discard
  走 query-only 分支(query 计数来自同一 CPU 域计数,不变)。

## 3. Buffer slot 布局(render 形态)

沿用 compute ABI slot 编号([mgl_air_tess_abi.h:130-160](file:///Users/fterward/MGL-minecraft/MGL/include/mgl_air_tess_abi.h#L130)),
但绑定阶段从 compute 改为 `MGL_RENDER_BINDING_STAGE_VERTEX`:

| slot | 内容 | 绑定方式 |
|---|---|---|
| 26 | factor 记录(half + float32 精确区) | setRenderBuffer,per-patch offset = p×36 |
| 27 | patch_out(per-patch varying) | setRenderBuffer,per-patch offset |
| 28 | seed TessCoord 记录(只读)/ TCS stage-out 布局 | setRenderBuffer,instance offset |
| 29 | contract {patch_id, gl_in_vertices, items, 0} | setBytes |
| 30 | gl_in(TCS 输出控制点,per-patch 偏移)/ indexed gather 流 | setRenderBuffer |

注意:render 侧 slot 25-30 有 per-stage 保留语义(见 mgl_buffer_slots.h 注册表);
TES-vertex 形态不得占用 `MGL_AIR_TESS_SLOT_TCS_STAGE_INRepl(24)`(仅 compute TCS 用)。
绑定必须经 `applyMSLResourceBindings` 冲突检测路径验证(项目已有 path-aware 冲突检测基建)。

## 4. 组件改动

### 4.1 ABI / 路径枚举

- [mgl_draw_tess.h](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_draw_tess.h):
  `MGLTessDrawExec` 增加 `MGL_TESS_EXEC_TES_VERTEX`;
- [mgl_draw_tess.cpp:166-217](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_draw_tess.cpp#L166)
  `mglTessPlanDrawPath`:
  `exec = EXEC_TES_VERTEX` 当 `tess_eval_render_vertex && env(MGL_TES_VERTEX_RENDER)`;
  否则维持 EXEC_TES_COMPUTE / UNSUPPORTED 判定。

### 4.2 program.c 判定拆分

[program.c:1877-1883](file:///Users/fterward/MGL-minecraft/MGL/src/program.c#L1877) 拆两个字段:

- `tess_eval_render_vertex` = `(tess_gen_mode == GL_ISOLINES || tess_gen_point_mode) &&
  transform_feedback_varying_count == 0 && !shader_slots[_GEOMETRY_SHADER]`;
- `tess_eval_compute` = XFB / GS / uses_tess_level(fill 场景)其余情形(收缩)。

[program.c:1669-1687](file:///Users/fterward/MGL-minecraft/MGL/src/program.c#L1669) FS 编译的
`MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER`(mgl_loc_N tag)与 iface remap(TES outs)条件
改为 `tess_eval_compute || tess_eval_render_vertex` —— FS 侧机制零改动,条件覆盖新字段。
save/restore 快照(program.c:92-209)同步新字段。

### 4.3 codegen:isTESVertex 目标

[mgl_air_backend.cpp](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_air_backend.cpp) 四个改动区:

1. **判定**(9579-9583):
   `isTESVertex = isTES && (isolines || point_mode) && !force_tes_compute && !has_gs && !xfb`
   (xfb 从 program 传入新编译 flag `MGL_AIR_COMPILE_TES_VERTEX`);
   `isKernel` 排除 isTESVertex(走 vertex 入口);
2. **入口参数**(10232-10312):isTESVertex 用 vertex 入口形状(参照 isVS 的 paramTys:
   返回结构 + vertex/instance/base 3×i32)+ 4 个 buffer 指针参数(slot 26/27/28/30)+
   contract 指针(slot 29);
3. **读取 lowering**(11464-11554):contract patch_id 读取与 factor GEP 保留;
   `threadItem` 语义变化:记录 index = `gl_VertexID`(不再 `outputBase + innerId` —— Metal
   vertex_id 已含 vertexStart=patchBases[p]);OOB 早退保留(无害,draw count 已保证);
4. **输出路径(最大新增)**:11563 段"写记录"替换为 vertex 输出 —— 参照普通 VS 输出
   codegen(10310 段 isVS 形状)+ TES sym 表(VARYING / mgl_loc_N):
   gl_Position 返回、varying store、gl_PointSize(point_mode 且 TES 写)、
   gl_CullDistance(共享 12507 分支语义:isolines 两端点 partner 规则由 passthrough
   的 (gl_VertexID ^ 1) 逻辑迁入)、gl_ClipDistance。

pipeline 侧:isTESVertex 不建 compute pipeline,函数注入 render PSO
(§4.5 缓存类别)。

### 4.4 dispatch:render vertex 派发

[MGLRenderer+Tessellation.m](file:////Users/fterward/MGL-minecraft/MGL/src/MGLRenderer+Tessellation.m)
新函数 `dispatchAIRTessEvalVertexRender`(约 250 行),复用 dispatchAIRTessEvalCompute 前半:

- evalPlan(`mglTessPlanEvalCompute`,items 计算)+ seed(`mglTessSeedEvalOutputRecords`,已有);
- 省去:compute pipeline 创建 / XFB 分支 / outBuffer 分配 / encoder 切换
  (若 TCS 刚跑完 compute,render encoder 已重建,直接复用);
- 绑定(§3 表)+ per-instance × per-patch
  `mglRenderEncodeDrawForRenderEncoderOwner`(`MGL_RENDER_DRAW_ARRAY`);
- 收尾:`mglRecordActivePrimitiveQueryDraw`(已有)+ `_currentCBHasWork` + 状态开关;
- indexed gather:沿用 slot 30 gather 流 + slot 25 params 绑定(仅 indexed 场景);
- 失败路径:glBindFramebuffer 一致性检查沿用 `mglTessPassthroughRasterReady` 判定。

### 4.5 PSO / 状态接线

- 复用 `_tessellation.tessComputeActive` 机制(passthrough 的管线选择路径不变):
  render 场景下 vertex function = TES-vertex(替换 passthrough);
  [MGLRenderer+RenderPass.m:1151](file:///Users/fterward/MGL-minecraft/MGL/src/MGLRenderer+RenderPass.m#L1151)
  `ensureAIRTessEvalPassthroughFunctionForProgram` 按形态返回 TES-vertex function
  (编译一次,按 program 缓存,新 passthrough 缓存类别 `TES_EVAL_VERTEX`);
- [mgl_draw_metal_port.m](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_draw_metal_port.m)
  ops 表:新增 `dispatch_air_tes_vertex` port;
- [mgl_draw_tess.cpp:3175](file:////Users/fterward/MGL-minecraft/MGL/src/mgl_draw_tess.cpp#L3175)
  airTES 分支按 exec 路由:EXEC_TES_VERTEX → 新 dispatch;EXEC_TES_COMPUTE → 旧路径;
- [mgl_buffer_slots.c:158/204](file:///Users/fterward/MGL-minecraft/MGL/src/mgl_buffer_slots.c#L158)
  stage binding 判定:TES vertex 形态按 render 域处理(排除 compute-TES 分支)。

### 4.6 回退开关

`MGL_TES_VERTEX_RENDER=0`(默认开,mgl_env_flag 基建)在
mglTessPlanDrawPath / program 判定处短路 → 全链路退回 compute。
用途:风险兜底 + A/B 性能对比。

## 5. 关键设计决策记录

1. **per-patch draw(而非单 draw)**:与 native 路径 `mglTessEncodeNativePatches`
   的 per-patch setBuffer+draw 模式对齐;codegen 只需最小分叉(factor GEP 保留
   patch_id 乘法)。单 draw + per-item patch-id 查找表留作后续优化(收益:draw call 数)。
2. **instance 用循环**:沿用 passthrough draw 的 per-instance 循环
   (MGLRenderer+Tessellation.m:1723-1733 模式),风险最低;顶点流 per-instance 重复 +
   单 draw 需要记录内嵌 instance index,留作后续。
3. **FS 零改动**:mgl_loc_N remap 机制已覆盖(program.c:1683 iface_peers = TES outs)。
4. **XFB/GS 不迁移(第一版)**:XFB 依赖 kernel 写 slot 31 记录 + CPU gather;
   GS 交接依赖记录流。render 化需要把捕获逻辑搬到 render 端(阶段 3)。
5. **uses_tess_level(fill)不迁移(第一版)**:虽然 factor float32 读取同构可搬,
   但保守起见 triangles-fill 场景维持已验证的 compute;阶段 2 再迁。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| render 侧 slot 26/27/28/30 与 per-stage 保留槽冲突 | `applyMSLResourceBindings` 冲突检测;实现时先跑 MSL 编译验证 slot 表 |
| codegen vertex 输出路径(4.3.4)为全新代码 | 参照 isVS 输出形状 + TES sym 表;单测 test-tess-air 对比 CPU/AIR 流 |
| 顶点流 gl_VertexID 语义(vertexStart 含入)与 kernel innerId 不一致 | 4.3.3 显式语义注释 + isolines query 计数测试(16/32/8)验证 |
| golden 图像变化(线段顺序/插值) | 域生成 emit 顺序即 line-list 语义,理论不变;若 golden diff 人工核对覆盖一致性 |
| 回归面大(6 个 isolines 非 XFB 测试 + point_mode 测试) | env 回退开关;先单测后套件 |

## 7. 验证方案

1. `make build-test-regression && make test-regression` — 93 全绿
   (isoline probe 像素、query 计数 16/32/8、tripoint 8 prims、cull_distance 8/16、
   rasterdiscard 8;isolines_xfb 保持 compute 不变);
2. `MGL_TES_VERTEX_RENDER=0 make test-regression` — 回退路径同样全绿;
3. `make test-tess-domain && make test-tess-air` — 单元/流比对全绿;
4. trace 抽查:`MGL TRACE tess path` 显示新 exec 枚举;日志无 per-patch compute
   dispatch 与 encoder 切换;
5. golden TGA 比对(不变或人工确认后更新)。

## 8. 实施顺序

1. ABI 枚举 + mglTessPlanDrawPath 路由(可编译,行为不变);
2. program.c 判定拆分(新字段,FS 条件覆盖,行为不变);
3. codegen isTESVertex(新编译目标,含单测);
4. dispatchAIRTessEvalVertexRender + ops port + PSO 缓存;
5. 状态接线 + env 回退;
6. 全套件验证 + 文档更新 —— **已完成**:步骤 1–5 已落地;`materializeAll` 阻塞已
   定位并修复(§10.1);§10.3 的 3 个 TES-vertex 功能正确性 bug 已全部修复,
   `make test-regression` 默认开与 `MGL_TES_VERTEX_RENDER=0` 均 91 PASS / 0 FAIL / 2 SKIP。

## 9. 路线图(后续阶段)

## 10. 验证结果(2026-09-10 更新)

| §7 项 | 内容 | 结果 |
|---|---|---|
| 3 | `make test-tess-domain` / `make test-tess-air` | ✅ `test_tess_domain: ok`;`test_tess_air: 180 CPU/AIR stream comparisons passed` |
| 2 | `MGL_TES_VERTEX_RENDER=0 make test-regression` | ✅ **PASS 91 / FAIL 0 / SKIP 2**(回退到 compute 全绿) |
| 1 | `make test-regression`(默认开) | ✅ **PASS 91 / FAIL 0 / SKIP 2** —— `materializeAll` 阻塞已修复;§10.3 的 3 个 TES-vertex 功能正确性 bug 已全部修复 |

原失败 3 例(默认开,isolines 场景)均已转绿并保留在回归集中:
`air_tessellation_isolines_indexed`、`air_tessellation_isolines_multidraw`、
`air_tessellation_cull_distance`。其中 indexed / multidraw 经双发 kernel +
per-draw 绑定 flag 退回 compute;cull_distance 修正为 per-vertex 剔除判据。

(注:`air_atomic_counter_validation` 的 `shader compile FAIL` 是预期的负向编译测试,
实际 PASS,不计入失败。)

### 10.1 原阻塞点:render PSO `materializeAll` —— 已定位并修复

现象(已消除):TES-vertex 函数编译成功、参数元数据正确,但 render pipeline 创建报
`Failed to materializeAll. (domain=AGXMetalG16G_B0 code=3)`,随后
`MGL TESS ERROR: TES-vertex raster skip` → draw 被跳过 → query 计数 0。
独立 harness(`test_legacy_compat/scratch_tes_vertex.mm`,`MGL_DUMP_IR=1` +
`MGL_DUMP_METALLIB=1`)确认失败隔离在**顶点函数本身**(即便
`rasterizationEnabled=NO` 的 vertex-only PSO 也失败),与 fragment 链接 / 光栅化无关;
且最 minimal 的 `isolines_minimal`(无 varyings,仅 `gl_Position`)同样失败,
排除用户 varying 问题。

根因(3 处,均在 `MGL/src/mgl_air_backend.cpp`):TES-vertex 编译目标被错误地套用了
**后曲面细分原生域着色器**的基础设施,使 metallib 含 Metal 无法解析的符号:

1. **悬挂的 control-point 函数声明**(L10353):原 `if (isTES && !isTESCompute)`
   对 TES-vertex 也为真(TES-vertex 满足 `isTES && !isTESCompute`),于是 emit 了
   `declare ... @_Z12ControlPoint.MTL_CONTROL_POINT_FN(i32, %struct._patch_control_point_t*) section "air.externally_defined"`。
   该符号只有同一 metallib 内存在 TCS kernel 时才可被链接;独立编译的普通
   `air.vertex` 函数无人提供 → 链接失败 → `materializeAll`。
   改为 `if (isTES && !isTESCompute && !isTESVertex)`,TES-vertex 不再 emit 它。
2. **无用的 opaque 类型**(L10091):`patchControlTy = isTES ? ...` 在 TES-vertex 下
   也建了 `struct._patch_control_point_t`(仅被上述 gated 分支引用)。改为
   `isTES && !isTESVertex`,消除悬挂类型。
3. **重复的 `ret` 终止符**(L11730):TES-vertex 收尾提前 `b.CreateRet(assembleReturn(cg))`,
   之后又落入通用收尾(文件末尾的 `else { b.CreateRet(assembleReturn(cg)); }`)再 emit 一次,
   致 entry block 含两个 terminator,IR 非法。删除提前 ret,让 TES-vertex 落入通用单次返回。

验证:harness 确认 IR 不再含 `ControlPoint` / `_patch_control_point_t` 声明,
且 `isolines` / `point_mode` 完整 PSO(含 fragment)构建成功;
`make test-regression` 主路径 6 个曲面细分用例(`air_tessellation_accumulation`、
`air_tessellation_isolines_point_mode`、`air_tessellation_isolines_variants`、
`air_tessellation_isolines_rasterdiscard`、`air_tessellation_isolines_tripoint_instanced`、
`air_tessellation_factors_spacing`)由 FAIL 转 PASS。

### 10.2 与本文档的设计偏差(实现已定,文档以此为准)

- §4.5 的独立 PSO 缓存类别 `TES_EVAL_VERTEX` **未新增**:直接复用
  `tessComputeProgram->modules[_TESS_EVALUATION_SHADER].mtl_function`,靠
  vertexProgram 差异区分缓存键。
- 枚举:删除 `MGL_TESS_EXEC_TES_FALLBACK`,`MGL_TESS_EXEC_UNSUPPORTED` 由 4 改 3,
  新增 `MGL_TESS_EXEC_TES_VERTEX = 4`。
- per-patch draw 记录由新增的 `mglTessBuildEvalVertexPatches` 生成
  (per-patch `{base, items}` + contract words),不是文档提到的
  `mglTessFillEvalPatchItemBases`。
- 新增 point-size 参数绑定 `kMGLPointSizeParamBufferIndex`(文档未列)。
- 多 instance + TCS 场景会复用 instance-0 控制点,已加
  `mglTessMultiInstanceTCSReuseWarn`(可配为错误)。
- **indexed draw 的结构性缺口**:TES-vertex ABI 只有 5 个 buffer 参数,没有
  compute 路径的 gather 流 + gather params 参数,而 indexed 场景的 gl_in 是 sparse
  capture 流,必须经 gather 才能定位。§2.3 声称 indexed 已被覆盖**不成立**。
  **已按"把 indexed 排除出 TES-vertex 路由(退回 compute)"解决**,并补齐 program
  层双发 kernel + per-draw 绑定 flag,见 §10.3 第 1 项。

### 10.3 3 个 TES-vertex 功能正确性 bug(已全部修复)

均为 draw 真正执行后的像素比对失败,**不再**是 `materializeAll`。修复后
`test_regression` 全量 93 项 = 91 PASS / 0 FAIL / 2 SKIP;回退开关
`MGL_TES_VERTEX_RENDER=0`(纯 compute 路径)同样全绿。

1. **`air_tessellation_isolines_indexed`** —— **已修复**。
   根因确认为 §10.2 记录的 **indexed 结构性缺口**:TES-vertex ABI 只有 5 个
   buffer 参数,没有 compute 路径的 gather 流 + gather params,indexed 场景的
   `gl_in` 是 sparse capture 流,必须经 gather 才能定位。
   采用"把 indexed 排除出 TES-vertex 路由(退回 compute)"方案,并补上其前置
   依赖(退回 compute 需要 program 内存在可用的 compute kernel):

   - **program 层双发**:`program.c` 在 isolines / point_mode 且无 XFB / GS、
     `MGL_TES_VERTEX_RENDER` 开启时,除主编译产出 render-vertex 函数
     (`modules[_TES].metallib_bytes`)外,再用去掉 `MGL_AIR_COMPILE_TES_VERTEX`
     的 `air_flags` 二次编译出 compute 展开 kernel,存入新增字段
     `modules[_TES].metallib_bytes_tes_compute`;`tess_eval_compute` 不再与
     `tess_eval_render_vertex` 互斥,两者可同时置位。kernel 发出失败时只告警,
     保底退回纯 compute(主编译行为)。
   - **kernel function 惰性加载**:`mglGetOrCreateProgramComputePipeline`
     (`mgl_render.cpp`)在 `metallib_bytes_tes_compute` 存在时优先使用
     `modules[_TES].mtl_function_compute`;该 function 在首次取 compute pipeline
     时经 `mglRenderLoadAIRMainFunction` 惰性加载并缓存。**不能依赖 stage 绑定期
     加载**:program 可能在 kernel blob 发布之前就已完成 stage 加载(绑定期早于
     链接期 emit),绑定期无法看到该字段。
   - **路由**:`mglTessPlanDrawPath` 与 draw dispatch 的 TES-vertex 分支增加
     `!indexed`,indexed draw(`glDrawElements` / `glMultiDrawElements`)落到
     `MGL_TESS_EXEC_TES_COMPUTE`(trace `exec=2`)。
   - **per-draw 绑定路径(必要配套)**:program 同时置位两个 field 后,
     `tess_eval_render_vertex` 这类 program 级标志不再能表达"本次 draw 走哪条
     路"。新增 per-draw `_tessellation.tessVertexRenderActive`
     (`MGLRenderer_State.h`),在 `dispatchAIRTessEvalVertexRender` 置 YES、
     `dispatchAIRTessEvalCompute` 置 NO;`prepareTessStageBufferBindings`
     (isolated 绑定 / copy-back 决策)与 `processGLState`(TES passthrough 函数
     选择、PSO `input_primitive_topology`)中的 5 处判断改读该 flag。否则
     compute 路径会按 render-vertex 语义绑定(错误地取消 isolated 绑定、
     选用 render-vertex 函数而非 slot-28 记录读取函数),导致扩流不落地。

2. **`air_tessellation_isolines_multidraw`** —— **已修复**,与 bug 1 同源。
   `glMultiDrawArrays` 段(非 indexed)仍走 TES-vertex;`glMultiDrawElements`
   段是 indexed,与 bug 1 走同一退回 compute 路径。同一 program 因此必须同时
   携带 render-vertex 函数与 compute kernel —— 这正是双发设计的动机。

### 10.4 再回归与根因修正(2026-09-12):PSO 键漏了 `tessVertexRenderActive`

§10.3 第 1/2 项当时确实转绿,但**后续某个提交把它regression 掉了**:在基线 `088394c`
上 `air_tessellation_isolines_multidraw` 再次失败(报 `elements probe 0 not drawn at
(52,54)`),而 `MGL_TES_VERTEX_RENDER=0`(完全没有 vertex 路线)时通过。用 stash 重建
基线库复验确认:该失败**不是本轮改动引入**。

- **真因**:同一 isolines/point-mode program 的两条栅格化路线**共用一个 PSO 缓存键**。
  pipeline 的 `primaryKey`(`MGLRenderer+RenderPass.m`)折入了
  `nativeTESActive / tessVertexCaptureActive / geometry.expansionActive /
  cullDistanceCaptureActive / tessComputeActive`,但**不含**决定"栅格化顶点函数取哪一个"的
  `tessVertexRenderActive`——为真时取 TES 自身的 render-vertex 函数,为假时取生成的 slot-28
  记录 passthrough。非 indexed 走前者、indexed 回退后者,于是第二次绘制命中第一次的 PSO,
  用 render-vertex ABI 去读 compute 写入的记录流,探测点自然没有像素。
- **修复**(`06ab844`):把 `tessVertexRenderActive` 折进 `primaryKey`(bit 18;19–23 已占用)。
- **验证**:该用例 Pass;本地 tess+GS 子集 25/25;CTS `KHR-GL46.tessellation_shader`
  139 pass / 1 fail / 0 ns(仍是已归档的 FO-spacing CTS 期望矛盾例)、
  `KHR-GL46.geometry_shader.*` 136/0、GL46 hotspot(1328)非通过集合与基线逐条 diff 为空。

> 教训:§10.2/§10.3 这类"同一 program 走两条栅格化路线"的设计,凡是**影响 PSO 内容**的 per-draw
> flag 都必须进 PSO 键;只在前 5 处判断里读 flag 而漏掉缓存键,会表现为"第一条 draw 正常、第二条
> 静默画不出东西"。

3. **`air_tessellation_cull_distance`** —— **已修复**。
   根因不是 partner 索引口径,而是**剔除判据本身用错**:原实现用
   `criterion = own * partner` 的**符号跨立**判定,当整条 isoline 行的顶点同号
   时(如 `d = 0.5 - v` 把 v=3/4 整行判负)永不触发,本应剔除的端点反而可见。
   按 GL 4.6 §13.6.1,顶点当且仅当其**自身** `gl_CullDistance` 分量为负时被剔除,
   光栅化再对跨立图元做裁剪:故判据改为 per-vertex `criterion = own`,
   `shouldCull = 任一分量 < 0`(`mgl_air_backend.cpp` 的 TES-vertex cull 块),
   并删除原 partner / straddle 记录索引逻辑(该逻辑对
   `vertexStart == 0` 的用例本就是 no-op,是误导性的表层原因)。

- **阶段 2**:GL_TRIANGLES fill + uses_tess_level 迁移(消除该强制 compute 项);
- **阶段 3**:XFB render 化(gather/捕获逻辑迁移)+ TES→GS 顶点流交接;
- **阶段 4(可选)**:单 draw + per-item patch-id 表;顶点流 per-instance 合并。
