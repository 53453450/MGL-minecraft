# GS XFB GL4 终态 + KHR-GL46 geometry_shader CTS 推进记录（2026-08-21）

本文档记录 2026-08-20/21 两天的 GS 工作切片：XFB GL4 终态、passthrough 闭环、
反射 API 闭环、KHR-GL46 geometry_shader 组首轮 CTS 批跑与失败分类。
承接 `docs/AIR_M3_CPP_TODO.md` 的 M3 标准 1。

## 一、已完成（全矩阵绿：regression 78/80 PASS 0 FAIL 2 SKIP[环境门控]，
## test-mglair / test-mglair-gtest / test-metalcpp 全 0，git diff --check 干净）

### 1. GL4 multi-stream XFB 终态（mgl_air_gs_abi.h §5b）

原型（2026-08-12）的四个缺口全部补齐，架构为 ordered 2-pass：

- **保序**：pass 1 kernel 不再原子 cursor 乱序写；stream 0 记录升序写在
  stage-out run 前段，stream>0 记录降序写后段并盖 stream 戳
  （`MGL_AIR_PER_VERTEX_STREAM_OFFSET`=48）；per-(work-item, buffer) 可见
  字节写 slot 26 visibility buffer；CPU 求 exclusive prefix；pass 2
  `gs_xfb_scatter`（P3 预编译 aux asset，`MGL/aux_shaders/gs_xfb_scatter.metal`）
  按序重排写 slot-31 临时 buffer，blit 回拷。
- **整图元跨 buffer 原子截断**：pass 2 以图元为单位检查"所有目标 buffer
  都装得下才写"（有序 offset 上天然原子）。
- **varying→buffer 绑定**：消费 link plan 的
  `transform_feedback_layout[].buffer_index/component_offset/stream`；
  `MGLAIRGSXFBMeta.buffer_stream[4]`（buffer→stream，link 验证保证唯一）传入
  kernel 做可见字节归属。stream 0 可经 gl_NextBuffer 喂多 buffer。
- **SEPARATE_ATTRIBS 放开**：program.c computeRoute gate 移除，与
  INTERLEAVED 共用同一 per-buffer 紧凑打包路径。

关键实现事实（后来者勿再踩）：

- `MGLAIRGSXFBStreamMeta` 16B/stream（stride/capacity/capture_base/generated），
  meta 共 80B（4×16 + buffer_stream[4]）；旧 cursor/written 原子字段已删。
- kernel 参数表：locs[8] = {24,28,29,30,25,31,27,26}，vis 必须独占 slot 26
  （与 gather 共用 30 会在 indexed+XFB draw 时互相覆盖）；threadPos 由
  isKernel 分支统一提供，isGS 切勿重复添加（参数/metadata 不等即 Metal
  编译服务 segfault）。
- stream>0 record 与 stream0 record 共用 work item 的 expanded region
  （全局 emit guard 保证总量 ≤ max_vertices，两区永不重叠）；
  **emitGeometryStreamVertex 不得碰 counts word0**（stream0-only 语义，
  光栅化间接绘制与 query 都依赖）。
- counts word0 在 `requireCPUVisibility=0` 时 CPU 读是竞态（读到的是
  memset 0）——诊断时别被误导；kernel 真实计数在 meta.generated。
- XFB record 布局是 GL 规范的紧凑 per-buffer 打包（只有声明的 varying），
  **不再是 80B full stage-out record**；回归断言全部按紧凑布局。

回归：`air_geometry_xfb`（紧凑+严格保序）、`air_geometry_multi_stream_xfb`
（双 buffer 严格保序）、`air_geometry_xfb_truncate`（跨 buffer 原子截断 +
1.5 图元容量无 torn record）、`air_geometry_separate_xfb`（SEPARATE 执行）。

### 2. passthrough GS + XFB 捕获闭环

passthrough 判定（`gl_in[n_vertex_index]` 直转发启发式）原有 4 处内联重复，
全部不收 XFB：bypass 后 draw 走普通 VS→FS，CPU feedback 又显式拒绝 GS 程序
（draw_buffers.c:936），捕获静默丢失。

收口：`mglProgramHasPassthroughGeometryShader` 移到
`mgl_program_reflection.h` 为 `static inline`（smoke 二进制不链接 .c 也能
共享），加 `transform_feedback_varying_count > 0 → 非 passthrough` 排除；
DrawSupport.m / RenderPass.m / mgl_render.cpp 三处内联副本全部删除改调
helper。回归 `air_geometry_passthrough_xfb`（旁路实验双向验证：无 gate 时
generated=0/written=0 复现丢失）。

### 3. TF 反射 API 面闭环

- `glGetTransformFeedbackVarying` 从 `mgl_unimplemented` 补全
  （ARB_transform_feedback3：特殊名计入索引；gl_NextBuffer→size=0/type=NONE；
  gl_SkipComponentsN→size=N/type=NONE；错误时输出不被修改）。
- `glGetProgramiv` 补 GL_TRANSFORM_FEEDBACK_VARYINGS / BUFFER_MODE（未指定
  时默认 INTERLEAVED）/ VARYING_MAX_LENGTH。
- `glGetProgramResourceiv(GL_TRANSFORM_FEEDBACK_VARYING)` 的
  REFERENCED_BY_*_SHADER 从 attached-mask 粗放值改为按名精确匹配
  （产出/消费该 varying 的 stage 才 TRUE，GL 4.6 §7.3.1）。
- 已核对无需改：AIR 反射默认 stream=0；link plan 默认 0；GL API 无 TF
  varying 的 stream 属性可暴露。

回归 `air_xfb_reflection`（4 用例：特殊名逐索引、SEPARATE 回读、无 TF
默认值 + 未链接错误、GS default-stream + referenced-by FALSE/TRUE/TRUE）。

### 4. GS 表达力补齐（CTS 驱动）

- **gl_Layer / gl_ViewportIndex 读回**：VAR_REF 新增分支（out 变量读回=
  本次 invocation 最近写入值，未写=0）。此前读取直接 codegen 报
  "unsupported construct"。
- **passthrough 整数 varying**：`flat out int/uint` 及其向量经
  `floatBitsToInt/Uint` 从 stage-out record 位回读（kernel 原始位存储），
  声明自动加 `flat`（GLSL 强制）。此前 passthrough 只支持 float 系，
  遇到 int 直接 GL_OUT_OF_MEMORY 级失败。
- **PRIMITIVES_GENERATED 精确计数**：kernel 侧 meta.stream[0].generated
  原子累积发射图元数（points 每次 EmitVertex +1；line/triangle strip 在
  图元发射点 +1；**含 culled**——culling 在 generation 之后，GL 4.6）。
  替代旧的 max-expansion 估计（发射不满 max_vertices 即错）与
  visible-record 和（排除 culled 即错）。renderer 在任一 primitive query
  （新增 `mglHasActivePrimitiveQuery`，非索引 index 0）活跃时强制 CPU
  可见并读 meta。stream>0 generated 移到 culled 检查之前（同样含 culled）。

回归 `air_geometry_layered_repro`（CTS layered_rendering 形态复刻：
uniform 门控 gl_Layer 写 + 读回 + flat int + 6 迭代 24 顶点，断言
generated=12 精确值）。

## 二、KHR-GL46 geometry_shader CTS 首轮批跑（mgl-gs-run3）

136 case，逐 case 单进程（run_mgl_cts_cases.py，45s 超时，fbo 256×256 hidden）：

**总：32 pass / 99 fail / 4 not_supported / 1 crash**（run1 基线 29/99/4/4
→ run2 29/96/4/7 → run3 32/99/4/1；crash 从 7 降到 1，6 个原 crash 转
pass/fail）

| 组 | pass/total | 备注 |
|---|---|---|
| rendering | 0/33 | 全灭：多为 "Could not build shader program"（编译构式未覆盖，待逐样本二分） |
| primitive_counter | 0/19 | 全灭：18 个 "Program could not have been created"（编译/链接，待查） |
| api | 6/16 | 4 ns 合理；6 fail 待查 |
| adjacency | 0/8 | GL_INVALID_OPERATION at esextcGeometryShaderAdjacency.cpp:422（渲染路径） |
| limits | 0/9 | 部分是 Metal 固有能力上限（GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS_EXT=0 等），1 crash |
| linking | 7/13 | |
| layered_*（合计） | 10/18 | layered_rendering 主 case 编译已过，像素不匹配（渲染内容偏差，待查） |
| input / output / 其余小组 | 5/13 | |
| blitting / clipping | 4/4 | 全过 |

三论 run 对比工具：
`mgl-gs-run{,2,3}/summary.tsv` + `qpa/`（逐 case qpa 日志）+
`crash-stderr/`（进程级 crash 尾部）。

### 已知在途问题

- layered_rendering 主 case：编译（gl_Layer 读回）已修，运行时渲染内容
  与参考不符（像素 diff），未定根因。
- limits 组 1 个 crash 未查。
- rendering/primitive_counter 的编译失败需要用 gscc 探针逐 case 二分
  （见下节）。

## 三、复现/迭代工作流（已验证可用）

```bash
# 编译
cd /Users/fterward/MGL-minecraft && make -j4 lib

# 单 case（workdir 必须是 modules 目录，否则 CTS 找不到资源文件）
cd /Users/fterward/VK-GL-CTS-build-mgl-target/external/openglcts/modules
DYLD_LIBRARY_PATH=/Users/fterward/MGL-minecraft ./glcts \
  --deqp-case=KHR-GL46.geometry_shader.xxx --deqp-surface-type=fbo \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-visibility=hidden --deqp-log-images=disable \
  --deqp-log-shader-sources=disable --deqp-log-decompiled-spirv=disable \
  --deqp-log-filename=/tmp/cts-one.qpa

# 批跑（GS 清单 /tmp/gs-cases.txt，136 case ≈ 27s）
python3 -u /Users/fterward/VK-GL-CTS-build-mgl-target/run_mgl_cts_cases.py \
  --glcts .../glcts --caselist /tmp/gs-cases.txt --workdir .../modules \
  --outdir .../mgl-gs-runN --dyld-library-path /Users/fterward/MGL-minecraft \
  --timeout 45 --progress-every 30

# shader 编译探针（不经 GL 运行时，直接报 codegen 错误）：/tmp/gscc
# 构建方式见会话记录（clang++ + MGL 后端源集 + mgl_uniform_reflection.o）
# 用法：/tmp/gscc gs shader.glsl ；MGL_DUMP_IR=1 可 dump LLVM IR
```

注：/tmp 下的 gs-cases.txt 与 gscc 探针是易失文件，重启后需重建（清单可由
khr-gl46-all-cases.clean.txt 用 `grep '^KHR-GL46.geometry_shader'` 重新导出）。

## 四、下一步（按收益排序）
1. rendering 组 33 个编译失败：取若干 case 的 qpa shader，用 gscc 二分
   不支持的构式（怀疑集中在循环内 if+EmitVertex 组合、数组 varying、
   或更多内建读回）。
2. primitive_counter 组 19 个链接失败：同上二分；修完正好用新增的内核侧
   generated 计数验证计数语义。
3. adjacency 组 GL_INVALID_OPERATION：draw 验证/拓扑映射问题，单独查。
4. layered_rendering 像素偏差：kernel 已发射（repro 证明），查分层 FBO
   与 passthrough 内容链路。
5. 全绿后跑全量 GL46（`khr-gl46-all-cases.clean.txt`，12080 case）。

## 五、2026-08-21 下午接力（rendering 组根因排查）

### 已修两个真实 bug（regression 80/82 全绿，test-mglair/gtest/metalcpp 全 0）

1. **VS capture variant 不注册 VARYING lvalue**（mgl_air_backend.cpp）：
   `isVS && isCapture` 时数组 varying（`out vec4 c[N]`）索引写入报
   "codegen: unknown lvalue"。补注册 UndefValue 聚合（同非 capture 路径），
   capture 返回组装也按元素扁平展开（对齐 retElems 构造）。
2. **tess capture（GS/TES 输入捕获）数组 varying 写溢出**：capture kernel
   把整个数组（N×16B）写进 per-vertex 记录的 64+loc*16 槽，元素 1 溢出到
   stride 80 记录之外、覆盖下一顶点的 gl_Position。GLSL 语义上 GS 输入数组
   按图元顶点索引（每顶点记录只存元素 0），修复为仅写元素 0（16B）。

新增回归：`air_geometry_points_grid`（CTS points_input_points_output 形态：
数组 varying + ivec2 uniform + 无 EndPrimitive 的 points 发射 + 无 GS 对照点
同像素）；`air_geometry_lines_expand`（CTS lines/line_strip 形态：数组 varying
按顶点索引读取 + 非对称 y + readback 底部原点断言；VS 含同名 uniform）。

诊断要点：`MGL_GS_DIAG=1` 现在还打印 uniform slot0 内容与 rasterize-check
（empty/culled）；gscc/gscc_tess 探针 + `MGL_DUMP_IR=1` 看 kernel IR；
glReadPixels 行原点在左下。

### 在途更新（条件消融完成，两个新修复）

**修复 3（commit b1876e0）：AIR argument 元数据未按参数索引排序**。
消融矩阵（普通 VS+FS，无 GS，gl_PointSize 补齐后干净复现）：
FS plain uniform + varying 同时存在 → varying 读 0（与 SSBO/GS 无关，
**普通渲染路径同样坏**；此前"GS 特有"是错觉——对照点绘制缺
gl_PointSize 写入导致不画，干扰了判断）。根因：emitter 把隐式
uniform-buffer 参数节点 push 在 fragment_input 值参数节点之前，
Metal 按列表顺序解析参数表导致 varying 链接错位。stable_sort 修复 +
回归 `fs_varying_with_plain_uniform`。GS kernel/compute（mglair 全套）
无回归。

同 commit 附带：GS compute dispatch 重建 render encoder 后补绑
fragment-stage buffers（dirty-domain 同步可能已按旧 encoder 标记完成）。

**GS rendering 组仍未通**：排序修复后普通路径 varying+uniform 正常，
但 GS expansion（passthrough VS 管线）+ FS plain uniform 组合仍 0 像素
（排序前是黑点=FS 执行 varying 0；排序后完全无像素）。run9 = 37/136
持平。剩余嫌疑：无 vertexDescriptor 的 GS 管线（4430 跳过
generateVertexDescriptorState）下 Metal 对排序后参数表的处理差异。
下一步：对比 GS 管线与普通管线在 Metal 端的 PSO 差异（GPU capture），
或试验给 GS 管线也生成 vertex descriptor（passthrough VS 无 attribute，
给空 descriptor）。

**验证基线**：regression 81/83（新增 1 用例），mglair/gtest/metalcpp 全 0，
非 GS CTS 抽样 157/299 持平无回归。

### 六、2026-08-21 傍晚接力（rendering 组 indexed 分支修复）

**修复 4：GS/TES indexed capture 的 GL_UNSIGNED_BYTE 索引未扩展**
（MGLRenderer+DrawSupport.m，captureAIRVertexPositionsForGeometryIndexed）。

消融定位链（air_geometry_points_grid 复制品 + CTS 单 case）：
1. points_grid 只画第 0 个 work item → 消融排除 indirect counts/绑定/偏移
   （ONLY_PRIM/DIRECT_DRAW/DRAW_VCOUNT/COPY_DRAW 全部同样症状）。
2. 色码探针（passthrough VS 把记录值编码进颜色）证明 draw 执行、SSBO 数据
   正确、psize/z/w 全对——唯独 varying 颜色槽 GPU 读到零 → 网格画了但纯黑，
   测试计数器看不见。
3. 追溯到 VS capture：非 indexed 时 capture 记录 1..N-1 的 varying 为零
   （复制品 color VBO 只有 12B、stride 0 画 4 点 → 属性越界读零，测试自身
   bug，已修 colors[3]→colors[12]）；indexed 时位置也错（记录 1+ 全零）。
4. indexed 复现：GL_UNSIGNED_INT 索引完全正常，GL_UNSIGNED_BYTE 全灭。
   根因：capture 直接把原始 byte EBO 配 MTLIndexTypeUInt16 下发，Metal 把
   字节对当 u16 索引读（{0,2}→512），属性越界取零、记录写飞。Metal 无
   UInt8 索引类型，正常 draw 路径走 mglPreparedElementIndexBuffer 扩展，
   capture 路径漏了。

修复：capture 函数改为接收 GL 索引类型（两个调用点同步），内部先做既有
restart 消毒（此前因收到 MTL 编码类型而恒跳过——顺带修复），再经
mglPreparedElementIndexBuffer 扩展/取正确 MTL 类型后下发。native-TES 路径
同款 bug 一并修复。

**附带发现（未修）**：
- glClear 写 alpha=255 而非 clear 色 alpha=0（CTS 未触碰像素校验会挂；
  rendering 组后续 case 可能受益于修这个）。
- 渲染中途 glReadPixels（clear 与 draw 之间）SIGSEGV。

**验证基线**：regression 81/83 全绿 EXIT=0；test-mglair / gtest(42) /
metalcpp 全过；CTS run11/run12 = **40/136**（+3：points_input_{points,
line_strip,triangles}_output 三个渲染 case 解锁），零回归。
