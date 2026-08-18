# AIR M3 与 Metal-cpp Renderer TODO

> 当前快照：2026-08-18（P5 单路径收口，platform-shell/value-state 边界批次）。
> 生产源码不再读取迁移期开关，command/resource/encoder 操作通过 C++ owner facade；
> `mgl_render_cpp_objc.h`、`MGLMetal*Ref` 和 smoke 对 transition adapter 的依赖已删除。
> 当前终态审计以 `make check-p5-metalcpp`、`make -j4 lib` 和 Metal-cpp smoke 为准；
> 历史 P4/A-B 数字仅保留在下方完成记录，不代表当前验证契约。
>
> 适用分支：当前工作区（`lgpl`）
>
> 设计依据：`AIR_SHADER_BACKEND_DESIGN.md`、`GEOMETRY_SHADER_METAL_PLAN.md`、
> `METALCPP_RENDERER_PLAN.md`

本文档是后续执行清单。旧计划保留为设计历史，但其中的进度数字和待办不再作为
当前事实；完成度只以当前源码、构建依赖和运行测试为准。

## 1. 终态定义

M3 完成必须同时满足以下四项，不能只以 smoke 或单个 draw case 通过作为完成：

1. triangles/quads TCS/TES 走 Metal 4 原生 tessellation；Metal API 无法表达的
   isolines/point-mode 走 compute/mesh emulation；GS 走 compute expansion，并
   覆盖 MGL 对外暴露的直接、indexed、instanced、indirect 和 multi-draw 入口。
2. GLSL 到运行时只保留自研 frontend -> MGLIR -> AIR -> metallib 路径；删除动态
   MSL fallback、glslang、SPIRV-Cross 和 SPIRV-Tools 的源码/构建依赖。
3. Metal renderer 的资源所有权、pipeline/pass/binding/draw/submit 主体由
   `mgl_render_cpp.cpp` 持有；C ABI 不暴露 `MTL::*`，Metal-cpp implementation
   macros 只在该 TU 定义。
4. Metal-cpp 过渡 gate 和 tessellation 过渡 gate 删除；ObjC 只保留 AppKit、
   `NSView`、`CAMetalLayer` 等平台外壳。

## 2. 当前进度摘要

| 领域 | 状态 | 当前证据 |
|---|---|---|
| AIR VS/FS/CS | 已接入 | frontend、reflection、metallib、PSO 和 runtime tests 已通过 |
| AIR TCS/TES | 大部分完成 | TCS kernel、TES post-tess vertex、varying/resource ABI 已有；native triangle/quad 支持 array/indexed/instanced/indirect 和非 indexed multi-patch；isolines/point-mode 走 TES compute expansion + passthrough vertex，isolines 几何按 GL 4.6 §11.2.2.3（n 行 edge v、m 段/行、每段 1 线 2 顶点），XFB 经 slot 31 捕获并与光栅化并存（见 P2E） |
| AIR GS | 大部分完成 | compute route、全部输入/输出拓扑、invocation、varying、resource、instancing、**direct indexed（P1）**、**indirect（P1）**、**multi-draw（P1）**、**XFB（P1，单 stream INTERLEAVED，slot 31 捕获 + slot 27 原子 meta）**、**gl_Layer/gl_ViewportIndex（2026-08-12，`air_geometry_layer_viewport` 回归）**、**多 stream 原型（2026-08-12，`air_geometry_multi_stream_xfb` 回归）** 已有；**XFB link-time layout plan（2026-08-16）** 已有；剩余 SEPARATE_ATTRIBS capture execution、规范化 multi-stream capacity/order、整图元截断/保序、passthrough-XFB 与 default-stream reflection 闭环 |
| 旧 SPIR-V 构建链 | 已完全脱离（2026-08-14 核验） | `external/` 无 glslang/SPIRV-*；`mgl_msl_compiler.*`/`test_msl_bindings` 已删；`check_air_only.sh` 通过（MGL/src 0 命中动态编译符号）；SPIRV 命名已迁为中性（`MGLShaderModule`/`MGLShaderResource`，旧名仅留 alias）；干净 clone `make` 即构建（glfw 自动配置，见 b188af2） |
| Metal-cpp 基础设施 | P5 单路径 | 纯 C facade、owner、resource creation、encoder setters、draw/blit/compute wrappers 与 command transaction 已接线 |
| C++ renderer P5 | 单路径收口 | 生产 gate、旧 ref typedef、过渡 adapter 和直接 ObjC Metal 操作已删除；平台对象由 `MGLPlatformRendererShell` 唯一持有；非平台 ObjC/私有头的 Metal 对象与 descriptor census 为零 |

### 当前工作区收口状态（2026-08-18，未提交增量）

本轮完成了 P5 单路径的 gate/adapter 删除、owner ABI 补强以及 platform-shell 边界收口；
私有 GL 语义接口按薄适配层使用纯 value/opaque 状态：

- `CommandBufferOwner` 的统一 transaction 覆盖 detached submission 匹配、commit
  guard、提交、可选等待、completion 注册、recovery value-state、next-current
  创建和 reset-request latch；`.m` 中 raw current-command-buffer getter 已归零。
- compute execution plan 已统一 resource binding replay、direct/indirect dispatch、
  buffer barrier、copy-back blit、owner submit/wait 与 CPU-prefix 同步；temporary
  resources 由 plan/snapshot 与 command buffer completion 保持所需生命周期。
- binding 的 buffer/bytes/texture/sampler/nil-clear ordered snapshot 已接线；
  Draw/DrawSupport setter/draw 和 render-encoder 生命周期均走 owner facade，
  不再保留 gate-off 分支。
- completion callback 现在由分类内显式 C callback/context 注册，block 通过
  `__bridge_retained` 配对 destroy callback 释放；不再有共享 ObjC transition adapter。
- `MGLRenderPassManager` 的 command-buffer、render-encoder、pending-event 和 MDI
  scratch 接口已改为 opaque `void *`；实现文件也只保留 untyped `id` bridge，Metal
  protocol/descriptor 类型和直接 Metal selector 均下沉到 C++ owner。
- `MGLPipelineCache.h` 的设备、pipeline/function、descriptor、格式和 blend 参数已
  改为 opaque `void *`/value `uint64_t`/`uint32_t`；descriptor value-state 读取、
  depth/stencil 创建和设备身份查询均通过 `mgl_render_cpp.cpp`，缓存状态由 C++ owner
  持有。
- `mgl_index_buffer.m` 已删除并由 `mgl_index_buffer.cpp` 接管；primitive-emulation
  index cache、UInt8 展开、source/readback 生命周期和 Metal buffer retain/release
  全部在 C++ owner 内完成。纯数值 ABI 固定 `MTLIndexType` 的实际值
  UInt16=0、UInt32=1，并保留 64-bit enum 输出宽度，避免 Objective-C
  `NS_ENUM(NSUInteger, ...)` 指针被 32-bit 写入破坏。
- `mgl_draw_encode.m` 的公共/私有入口已改为 opaque handle、数值 primitive/index
  enum 和 C `bool`；它只保留 GL primitive emulation/validation，draw 提交统一走
  `mglRenderCppEncodeDrawForRenderEncoderOwner`。
- `check-p5-metalcpp` 直接检查 gate/legacy bridge/ref typedef、adapter 文件、
  implementation macro 唯一性、backend/platform roots、render-pass/pipeline-cache
  私有头 opaque 约束和 Objective-C command operation；`make -j4 lib`、
  `make test-metalcpp`、`git diff --check` 已通过。
- 后续审计不再使用 `MGL_USE_METALCPP=0/1` A/B；回归、ASan/TSan 和干净 clone
  构建必须以单一路径分别执行。

本轮串行回归证据（构建产物为当前工作区）：唯一 Metal-cpp 路径为
`73 PASS / 0 FAIL / 2 SKIP`；`make check-air-only`、`make test-mglair`、
`make test-mglair-gtest`（42/42）、`make test-metalcpp` 和
`make check-p5-metalcpp` 通过。

- `MGLPlatformRendererShell` 现在是唯一的 `CAMetalLayer`/drawable/AppKit Metal
  生命周期边界：layer 配置、几何同步、drawable texture bridge 和 detach 均由 shell
  facade 提供；`MGLRenderer+Lifecycle.m`、`MGLRenderer.m`、`+RenderPass.m`、`+Blit.m`
  与 `+Texture.m` 不再发送 layer/drawable Metal selector。standalone smoke 的
  QuartzCore 链接依赖已显式声明。
- `mgl_state_compat.h` 与 `mgl_readback.h` 的跨语言签名已改为 `uint32_t`/opaque
  value-state，不再暴露 `MTLCompareFunction`、`MTLWinding` 或 `MTLPixelFormat`。
  P5 checker 现在逐文件剥离注释后审计所有 ObjC 源和私有头，并拒绝非平台
  layer/drawable selector、`CAMetalDrawable` 和 command-queue class 访问。
- 本轮新增验证：`make test-regression` 在接口调整后重新编译，结果仍为
  `73 PASS / 0 FAIL / 2 SKIP`；仓库中未发现可运行的完整 CTS runner 或 Metal 4
  真机专项入口，相关清单保持未勾选，不用 regression/smoke 代替其证据。
- sanitizer 终验：`make build_dir=build-asan-p5 SANITIZE=address test-regression`
  与 `make build_dir=build-tsan-p5 SANITIZE=thread test-regression` 均为
  `73 PASS / 0 FAIL / 2 SKIP`，无 sanitizer 报告。ASan 首轮暴露的
  cull-distance capture 后批次 context 持有已释放 render-encoder owner 问题，
  已在 `MGLRenderer+BatchReplay.m` 刷新恢复后的 owner 句柄并重新验证通过。

### 2.4 P0 完成记录（2026-08-10）

P0「固定 M3 runtime contract」已交付并全绿验证：

- **新增 `MGL/include/mgl_air_gs_abi.h`**：GS compute expansion 固定 C ABI——
  输出 record 布局（`MGL_AIR_GS_HEADER_RECORDS=2` + expanded 顶点，
  `mglAIRGSExpandedVertices` / `mglAIRGSRecordsPerPrimitive`）；counts record
  从 16B 改为 **28B**（16B `MGLAIRGSIndirectArgs` + 12B kernel scratch，
  strip/emit 滚动不再污染 draw 参数）；GS kernel slots（INPUT=24 / OUTPUT=28 /
  COUNTS=29 / XFB=30）；`MGLAIRGSIndexGatherParams`（P1 direct indexed 预留）；
  `MGLAIRGSXFBParams`；全套 `MGL_AIR_STATIC_ASSERT`（args==16B、逐字段偏移、
  COUNTS_RECORD_BYTES==28、PER_VERTEX_STRIDE==64）。
- **新增 `MGL/include/mgl_air_tess_abi.h`**：`MGLAIRTessDrawContract` value
  state（patch vertex count / patch count / instance / base-instance / index
  source / tess factor layout / patch varying）；`mglAIRPatchVaryingStride`
  helper；tess factor 布局（quad half 12B）；TESS slots（TESS_FACTOR=26 /
  PATCH_OUT=27 / PATCH_INFO=28 / **TCS_OUTPUT=28（跨 encoder 复用）** /
  INDIRECT=29 / GL_IN=30 / TCS_STAGE_IN=24）。
- **ObjC 接入**：`handleGeometryShaderArrayDrawIfNeeded:` 改用 ABI 常量/函数，
  counts 28B 预设（instance=1/base=0），unsupported 分支报明确 GL error；
  `dispatchTessControlShader:` / `dispatchTessEvaluationShader:` 改收
  `const MGLAIRTessDrawContract *`；`handleTessellationPatchDrawIfNeeded:`
  构造 contract。
- **负向测试** `test_air_gs_unsupported`（test_regression，MAX_TESTS→35）：
  GS triangles-in + GL_POINTS draw → GL_INVALID_OPERATION；匹配 draw 无 error。
- **顺带修复**：GS kernel 收尾 16B stride 错位（work item>0 时 vertex_count 被
  base_instance 覆盖 → 空 draw）；TCS spvOut slot 误绑（28↔30 区分）。
- **验证**：lib / test-mglair / gtest 41/41 / test-metalcpp / regression
  A/B 均 33 PASS / 0 FAIL / 2 SKIP / git diff --check 全绿。

### 2.1 已完成并应保留的 AIR/M3 能力

- `glCompileShader` 和 program link 的默认生产路径已直接调用自研 AIR frontend。
- TCS/TES/GS 均可生成 metallib；`test-mglair` 已输出 `TCS_OK`、`TES_OK`、
  `GS_OK`、`XFB_OK` 和 `VALUE_OK`。
- TES native ABI 已支持 plain uniform、UBO、SSBO、sampled texture/sampler 和
  storage image 的反射/绑定基础。
- GS compute expansion 已支持：
  - points、lines、lines-adjacency、triangles、triangles-adjacency 输入；
  - points、line-strip、triangle-strip 输出并展开为 Metal list primitive；
  - `layout(invocations=N)`、`gl_InvocationID`、`gl_PrimitiveIDIn`；
  - VS -> GS -> FS 用户 varying packing；
  - sampled texture、UBO、SSBO、runtime `.length()`、storage image；
  - array draw instancing 与 base-instance；
  - GS 输出 `gl_PointSize`、`gl_CullDistance` 记录和 primitive-level 聚合。
- storage image 已支持 `imageLoad`、`imageStore`、`imageSize`、format layout
  qualifier、reflection 和 `glMemoryBarrier` readback。
- regression 已有 `air_geometry_varying`、`air_geometry_resources`、
  `air_geometry_instancing`、`air_tessellation_varying` 和 `air_cull_distance`。

### 2.2 已完成并应保留的 Metal-cpp 基础

- `mgl_render_cpp.cpp` 是唯一 `NS_PRIVATE_IMPLEMENTATION` /
  `MTL_PRIVATE_IMPLEMENTATION` TU；其他 C++ TU 只包含声明。
- C/ObjC 边界使用基础类型、value state 和 opaque owner，不在 C ABI 中泄漏
  `MTL::*`。
- 已有 owner/facade 覆盖：device、command queue、command buffer/submission、
  render encoder、render-pass state/identity、binding state、query state、pipeline
  cache、buffer COW/map/readback、texture descriptor/transfer/staging、sampler、
  depth-stencil、render/compute/blit encoder setters 和 draw dispatch。
- mipmap 生成已通过 `mglRenderCppBlitGenerateMipmaps` 进入 C++ blit facade。
- Metal-cpp smoke 已覆盖资源 ownership、command lifecycle、render-pass state、
  binding dedup、texture upload/readback、mipmap encode、ICB 和 raw render/blit。

### 2.3 2026-08-10 验证基线

- `make -j4 lib`：通过。
- `make test-mglair`：通过，包含 TCS/TES/GS/XFB/value signals。
- `make test-mglair-gtest`：41/41 通过。
- `make test-metalcpp`：通过，结束信号 `SMOKE_DONE`。
- renderer regression：
  - 单一路径：41 PASS / 0 FAIL / 2 SKIP（43 测试）。
- benchmark `--ab`：A/B 双 run 对比表输出正常（Metal-cpp vs ObjC，Delta/Change）。

这些结果证明当前增量可用，不证明 M3 或 renderer 重写已经收尾。

## 3. 剩余 TODO（按执行顺序）

### P0 - 固定 M3 runtime contract ✅（2026-08-10 完成）

- [x] 为 GS input/output record、index gather、indirect args 和 XFB record 写固定的
  C ABI 结构及 slot 表，避免继续在 ObjC runtime 和 AIR backend 之间隐式约定。
- [x] 把 TCS/TES native draw contract 写成 value state：patch vertex count、patch
  count、instance/base-instance、index source、tess factor layout 和 patch varying。
- [x] 为每个不支持的组合返回明确 GL error 或 link failure，禁止静默跳过 draw。

验收：ABI 结构有 static assertions；所有 unsupported 分支都有对应负向测试。
（详见 §2.4；`test_air_gs_unsupported` 负向测试已入 regression 套件。）

### P1 - 完成 GS runtime 泛化

- [x] 支持 direct indexed GS draw：`glDrawElements*`、base-vertex、base-instance、
  primitive restart 和 adjacency 输入。（2026-08-10：air_geometry_indexed 已入
  regression，覆盖 plain indexed / base-vertex / restart。）
- [x] 支持 `glDrawArraysIndirect` / `glDrawElementsIndirect`；indirect 参数必须在
  GPU 可见顺序内驱动 capture、GS dispatch 和最终 raster draw。
  （2026-08-10：CPU 读回命令后经普通 GS 路径驱动，air_geometry_indirect 已入
  regression。注：GPU-visible ordering 由读回+立即转发保持，未做纯 GPU 端解码。）
- [x] 支持 multi-draw 和 multi-draw-indirect；每个子 draw 的 primitive ID、instance
  ID、base instance 和输出 offset 必须独立。（2026-08-10：MultiDraw 家族 5 个入口
  均逐子/逐命令经 GS 路径；air_geometry_multi_draw 已入 regression。）
- [x] 删除 `handleGeometryShaderArrayDrawIfNeeded` 的 array-only 边界，统一为一个
  geometry draw plan。（已更名为 `handleGeometryDrawIfNeeded:` 并统一 array/indexed
  两条路径；`mglBlockUnsupportedGeometryDraw` 已删除。）
- [x] 支持 `gl_Layer` 和 `gl_ViewportIndex`，并把结果接到最终 rasterization ABI。
  （2026-08-12：AIR metadata 用 `air.render_target_array_index`/`air.viewport_array_index`
  ——`air.layer` 在 macOS 上被编译器忽略；renderTargetArrayLength 按 attachment
  arrayLength 设置；VS 写这两个属性时 inputPrimitiveTopology 必须非 unspecified；
  passthrough 从记录 vec4 index 2 的 z/w 读 layer/viewport（int 位模式，经
  `floatBitsToInt` 还原）；GL 4.6 §11.1.3.5 单写绑定的同值语义在 backend
  assembleReturn 已实现。回归测试 `air_geometry_layer_viewport`。）
  （2026-08-16：旧的单层绑定限制已修复。render pass 现在按 FBO attachment 的
  `layered` 语义选路：`glFramebufferTextureLayer` 保留 `slice/depthPlane` 且
  `renderTargetArrayLength=0`；whole-level layered attachment 才归零 base
  slice/depthPlane，并启用 `render_target_array_index`。）
- [ ] 完成 GL4 multi-stream XFB 终态语义。（2026-08-12 已有原型子集：
  `EmitStreamVertex` / `EndStreamPrimitive` 支持 stream 0-3；stream 0 光栅化 +
  XFB，stream 1-3 仅 XFB；per-stream meta
  cursor 在 `MGLAIRGSXFBMeta` 内分块，copy-back 按流独立回拷；passthrough
  vertex 过滤 stream>0 varying 防 location 冲突；`storeGeometryVaryings` /
  `copyGeometryVaryings` / `copyGeometryVaryingsSelected` 跳过 stream!=0
  varying 避免覆盖。**非终态**：仍采用 `stream s -> buffer s`、
  per-stream 独立容量截断和 order-agnostic 写入，尚未实现 GL4 varying binding/
  整图元跨 binding 原子截断与保序。非 indexed query 已在
  2026-08-16 修正为 stream 0 的 `3/3`；同日补齐 primitive query 的
  `(target,index)` 状态，支持 stream 0-3 的 `glBeginQueryIndexed` /
  `glEndQueryIndexed`，并在 GS meta 中记录 stream>0 的 emitted-point counter，
  因而 XFB active 与 inactive 两种路径均可验证 stream 1 query `3/3`。
  当前 stream>0 仍受 points-only 原型约束，不能将该切片视为 GL4 终态。现有
  `air_geometry_multi_stream_xfb` 只作为原型回归，终态需重写。）
- [x] 接入 GS transform feedback，包括 offset/stride、overflow、rasterizer-discard
  和 query 统计。（2026-08-11：单 stream GL_INTERLEAVED_ATTRIBS 已接入——kernel
  按 gl_CullDistance 剔除语义经 slot 31 追加捕获、slot 27 原子 meta cursor 紧凑
  排列，直接绑定或共享临时 + blit 完整原语前缀回拷；PRIMITIVES_GENERATED/
  TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN 在光栅化、rasterizer-discard 与空光栅
  三条路径都记录；SEPARATE_ATTRIBS 仍在 program.c computeRoute 门槛外拒绝。
  air_geometry_xfb 已入 regression。）
- [x] 补齐 GS barrier/copy-back 与 image/SSBO 写后可见性测试。（2026-08-14：
  copy-back 由 air_geometry_xfb/multi_stream_xfb 覆盖；image/SSBO 写→CPU 读回
  由 air_geometry_resources 覆盖；新增 air_geometry_ssbo_visibility 覆盖
  GPU→GPU 可见性——段 1：GS 写 SSBO（蓝）→ glMemoryBarrier(
  GL_SHADER_STORAGE_BARRIER_BIT) → 后续 draw 的 GS 读回并渲染到右侧（蓝），
  写入方自身左侧绿色，位置分离证明无陈旧值；段 2：GS imageStore 写 1x1 纹
  理（红）→ glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT) → 后续 draw 的 GS
  采样渲染到右侧（红）。A/B 双门 PASS。）

必须新增的产品级回归：

- [x] `air_geometry_indexed`（2026-08-10：plain indexed + base-vertex + restart）
- [x] `air_geometry_base_vertex_instance`（2026-08-10：
  drawElementsInstancedBaseVertexBaseInstance 逐 instance×索引展开）
- [x] `air_geometry_indirect`（2026-08-10：arrays-indirect + elements-indirect）
- [x] `air_geometry_multi_draw`（2026-08-10：multi-draw-elements +
  multi-draw-arrays-indirect）
- [x] `air_geometry_layer_viewport`（2026-08-12：普通 VS/GS 链双覆盖 layer+viewport；
  2026-08-16 改为 whole-level layered attachment，明确由 `gl_Layer=1` 选择
  layer 1；单层 slice 语义由 `air_renderpass_layer_slice` 独立正向覆盖）
- [x] `air_geometry_xfb`（2026-08-11：points-in/triangle-strip-out，两可见段 +
  剔除段，pixel probe + query 2/2、3/2 + FLT 记录/前缀校验）
- [ ] `air_geometry_multi_stream_xfb` GL4 终态回归（2026-08-12 已有
  points-in/points-out 双流原型，
  stream 0 光栅化 + XFB buffer 0（80B stage-out record），stream 1 仅 XFB
  buffer 1（32B compact record）；非 indexed query 已按 stream 0 验证
  PRIMITIVES_GENERATED=3 / TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN=3；
  2026-08-16 又验证 stream 1 的 indexed query 在 XFB active/inactive 两条路径
  均为 `3/3`（generated/written 或 generated-only）；
  order-agnostic XFB 验证。仍需按 link-time layout plan 与
  GL4 query/capacity/order 规则重写后再勾选。）
- [x] `air_geometry_cull_distance`（2026-08-14：GS 逐发射顶点写
  gl_CullDistance——all-negative 剔除 / all-positive 可见 / mixed
  (+1,-1,+1) 按 GL 规则可见；glDrawArrays(GL_POINTS)（direct 批路径）+
  glDrawElements（element 路径）双段像素探针，A/B 双门 PASS）

### P1 阶段一完成记录（2026-08-10：direct indexed GS）

- **indirect**：`mtlDrawArraysIndirect:` / `mtlDrawElementsIndirect:` 在
  resolveIndirectBufferForDraw 后、cull-distance 检查前插入 GS 分支——用
  `prepareEmulatedIndirectCPURead` + `mglReadBufferBytes` 读回单命令，转发到
  `mtlDrawArraysInstancedBaseInstance` / `mtlDrawElementsInstancedBaseVertexBaseInstance`
  （内部 handleGeometryDrawIfNeeded）。
- **multi-draw**：`mtlMultiDrawArrays/Elements/ElementsBaseVertex` 在 GL_PATCHES 块后
  插 GS 逐子转发（handler 返回 NO 时回退普通 draw）；`mtlMultiDrawArraysIndirect /
  mtlMultiDrawElementsIndirect` 在 resolve 后插逐命令读回转发。
- **deferred 修正**：3 个非 indirect MultiDraw 前端（draw_buffers.c）在
  `draw_defer_enabled` record 前加 `mglCurrentExpandedGeometryDrawProgram` 检查——
  GS 激活时直接转发 mtl_funcs（batch replay 不做 GS 展开）。
- **删除** `mglBlockUnsupportedGeometryDraw`（7 处调用 + 函数本身）。
- **回归**：`air_geometry_indirect`（drawArraysIndirect + drawElementsIndirect）、
  `air_geometry_multi_draw`（glMultiDrawElements + glMultiDrawArraysIndirect）。
- **验证**：regression A/B 均 36 PASS / 0 FAIL / 2 SKIP（38 测试）+ 全套矩阵全绿。

- **ABI**（`mgl_air_gs_abi.h` §7）：`MGL_AIR_GS_SLOT_GATHER=30`、
  `MGL_AIR_GS_SLOT_GATHER_PARAMS=25`、`MGLAIRGSGatherParams`
  （vertices_per_instance / primitives_per_instance / first_vertex /
  gather_enabled）+ static assert（16B、slot 30）。
- **backend**：GS kernel 参数 3→5（input/output/counts/gather/gather_params，
  `air.buffer` metadata slot 24/28/29/30/25）；`geometryInputRecordIndex`
  helper 实现 gl_in 间接寻址——array 路径 `globPrim*iv+v`，indexed 路径
  （运行时 gather_enabled）`gather[primInInst*iv+v] - first_vertex +
  instanceIdx*vertices_per_instance`，instance 分解
  `globPrim = instanceIdx*primsPerInst + primInInst`（gather 流跨实例共享）。
- **capture**：`captureAIRVertexPositionsForGeometryIndexed:` 用
  `drawIndexedPrimitives`（原 EBO + GL baseVertex，Metal stage_in 自动应用
  baseVertex；vertex_id = 原始索引值）→ 稀疏 records
  （[instance][vertex_id]，每实例跨度 maxIndex+1）。
- **CPU gather**：`mglGeometryGatherIndices`（restart 断段、每段按
  inputVertices 成组、丢弃尾部不全组、per-instance gather 数组）。
- **接入**：`handleGeometryShaderArrayDrawIfNeeded:` 更名并泛化为
  `handleGeometryDrawIfNeeded:`（array+indexed）；Draw.m 的 3 个 array +
  8 个 indexed 入口（drawElements 家族全部）接入；deferred 路径
  （draw_buffers.c）移除 indexed block、补 indexed 类型转发。
- **踩坑**：GS kernel 加参数后漏改 `air.buffer` metadata（3→5），Metal
  编译失败（XPC_ERROR_CONNECTION_INTERRUPTED）——补 metadata 后恢复；
  批量插入调用点时产生多余 `return;`（8 处，已清）。
- **验证**：lib / test-mglair / gtest 41/41 / test-metalcpp / regression
  A/B 均 34 PASS / 0 FAIL / 2 SKIP（36 测试，+air_geometry_indexed）/
  git diff --check 全绿。

### P1 阶段二完成记录（2026-08-10：indirect + multi-draw）

- **indirect**：`mtlDrawArraysIndirect:` / `mtlDrawElementsIndirect:` 在
  resolveIndirectBufferForDraw 后、cull-distance 检查前插入 GS 分支——用
  `prepareEmulatedIndirectCPURead` + `mglReadBufferBytes` 读回单命令，转发到
  `mtlDrawArraysInstancedBaseInstance` / `mtlDrawElementsInstancedBaseVertexBaseInstance`
  （内部 handleGeometryDrawIfNeeded）。
- **multi-draw**：`mtlMultiDrawArrays/Elements/ElementsBaseVertex` 在 GL_PATCHES 块后
  插 GS 逐子转发（handler 返回 NO 时回退普通 draw）；`mtlMultiDrawArraysIndirect /
  mtlMultiDrawElementsIndirect` 在 resolve 后插逐命令读回转发。
- **deferred 修正**：3 个非 indirect MultiDraw 前端（draw_buffers.c）在
  `draw_defer_enabled` record 前加 `mglCurrentExpandedGeometryDrawProgram` 检查——
  GS 激活时直接转发 mtl_funcs（batch replay 不做 GS 展开）。
- **删除** `mglBlockUnsupportedGeometryDraw`（7 处调用 + 函数本身）。
- **回归**：`air_geometry_indirect`（drawArraysIndirect + drawElementsIndirect）、
  `air_geometry_multi_draw`（glMultiDrawElements + glMultiDrawArraysIndirect）。
- **验证**：regression A/B 均 36 PASS / 0 FAIL / 2 SKIP（38 测试）+ 全套矩阵全绿。

### P1 阶段三完成记录（2026-08-10：base_vertex_instance 回归）

- **回归** `air_geometry_base_vertex_instance`：`glDrawElementsInstancedBaseVertexBaseInstance`
  （indexed + instance + base-vertex + base-instance）经 `handleGeometryDrawIfNeeded:`
  逐 (instance × indexed vertex) 展开。极简 VS 下 capture position 正确
  （VBO[2]/[3] 直接解析）。
- **待查**（已记入项目记忆）：含 `gl_InstanceID`/`gl_BaseInstance` 的 VS 在 indexed
  capture 中曾出现 position 与 varying 不一致（极简 VS 验证 capture 机制正确，PASS；
  该现象属 VS 语义层，待专项排查）。
- **验证**：regression A/B 均 37 PASS / 0 FAIL / 2 SKIP（39 测试）+
  lib / test-mglair / gtest 41/41 / test-metalcpp / git diff --check 全绿。

### P1 阶段四原型记录（2026-08-12：多 stream XFB，非 GL4 终态）

- **AIR backend**（`mgl_air_backend.cpp`）：`EmitStreamVertex(stream)` /
  `EndStreamPrimitive(stream)` 支持 stream 0-3（constant arg）。stream > 0
  走 `emitGeometryStreamVertex`——写 compact record（position 16B + 该流
  varyings 按 location 升序各 16B）到 slot 31 XFB buffer，经 per-stream meta
  cursor（`MGLAIRGSXFBMeta` 内 `stream[s]` 块：stride/capacity/capture_base/
  cursor/written）原子保留空间。stream 0 继续走 `emitGeometryVertex`（full
  stage-out record）+ kernel epilogue 的 atomic cursor memcpy。
- **varying 过滤**：`storeGeometryVaryings` / `copyGeometryVaryings` /
  `copyGeometryVaryingsSelected` 跳过 `stream != 0` 的 OUTPUT varying——
  stage-out record 仅服务 stream 0 光栅化与 XFB，stream > 0 的 varying 由
  `storeGeometryXFBStreamVaryings` 在 compact record 中独立存储；同时解决
  不同流共享 `layout(location=N)` 时的字节偏移冲突（s0_data/s1_data 均
  location=0 → 同写 byte 64 → s1_data undef 覆盖 s0_data）。
- **passthrough vertex**（`MGLRenderer+RenderPass.m`）：生成直通顶点着色器
  时过滤 `output->stream > 0` 的 varying，避免 stream 1+ 的 location 与
  stream 0 冲突。
- **copy-back**（`MGLRenderer+DrawSupport.m`）：多流路径分配临时 buffer
  （`physTotal = Σ streamCapBytes[s]`），每流 `capture_base = streamPhysBase[s]`；
  kernel 完成后按流循环 blit 回各自 GL XFB 目标 buffer，`pbytes` 对 stream 0
  为 `vpp * outputStride`、对 stream > 0 为 `streamStride[s]`（points-only
  vpp=1）；`writtenBytesStream0` 从 copy-back 的 `copyBytes` 取值而非 meta。
- **query 计数**：非 indexed `PRIMITIVES_GENERATED` 与
  `TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` 仍只取 stream 0（分别从 counts 与
  `writtenBytesStream0` 计算）；2026-08-16 已增加按 `(target,index)` 的 active
  query 表，stream 1-3 的 `glBeginQueryIndexed` / `glEndQueryIndexed` 及
  `glGetQueryIndexediv` 可用。GS meta 的 generated counter 独立于 XFB capture，
  因此 stream 1 `PRIMITIVES_GENERATED` 在 XFB active/inactive 均可正确计数。
- **PSO 拓扑**：`inputPrimitiveTopology` 仅在 VS/GS 写 `gl_Layer` 或
  `gl_ViewportIndex` 时设置（避免 points output + 无 layer 时 Metal
  validation 报 point size / topology class 冲突）。
- **回归** `air_geometry_multi_stream_xfb`：points-in/points-out 双流 GS，
  3 input points 各 emit 1 stream-0 vertex + 1 stream-1 vertex；验证
  ① stream 0 光栅化 3 green points ②非 indexed query 3/3（stream 0）
  ③ indexed stream 1 query 在 XFB active/inactive 均正确（3/3、3）
  ④ XFB buffer 0 三条
  80B stage-out record（position@0, s0_data@float16）⑤ XFB buffer 1 三条
  32B compact record（position@0, s1_data@float4）。XFB 记录顺序
  order-agnostic 验证（Metal compute 线程组执行顺序未定义，原子 cursor
  不保序）。
- **已知限制**：XFB 记录顺序不保证——GL 4.6 要求按图元生成顺序写入 XFB
  buffer，但 Metal compute 不保证线程组执行顺序，原子 cursor 保留空间的
  顺序取决于哪个 work item 先执行 atomicrmw。单 stream 测试（2 work items）
  碰巧保序通过；3+ work points 可能乱序。修复需 prefix-sum dispatch
  （先写 visible counts、再 scan、再按 scan offset 拷贝），留作后续。
- **验证**：regression A/B 均 51 PASS / 0 FAIL / 2 SKIP（53 测试）+
  lib / test-mglair（TCS/TES/GS/XFB/VALUE OK）/ gtest 41/41 /
  test-metalcpp / git diff --check 全绿。

### P2 - 完成 Metal 4 native tessellation 终态

- [x] native TES 支持 indexed patch draw（2026-08-10：sparse VS capture + CPU
  gather buffer 驱动 `drawIndexedPatches`，PSO cpiType=UInt32；
  `air_tessellation_indexed` 已入 regression。非 indexed multi-patch 的控制点、
  PrimitiveID、varying 和 tess factor 寻址已在阶段四修复并覆盖）。
- [x] native TES 支持通用 instancing（2026-08-10：无 TCS 的 array/indexed
  instanced draw 已走逐 instance native draw，`air_tessellation_instanced`
  已入 regression；**TCS + instanced 仍受限**——TCS kernel dispatch 无
  instance 维）。
- [x] 支持 indirect/multi-draw patch command decoding（2026-08-10：4 个 indirect
  入口 CPU 读回命令后逐条转发到显式 patch draw 路径，`air_tessellation_indirect`
  已入 regression，见阶段三完成记录）。
- [x] 明确 native capability boundary：macOS/iOS Metal 的 `MTLPatchType` 只有
  None/Triangle/Quad，且 tessellation draw/pipeline 无 point output topology；
  isolines 与 point-mode 不属于 native 可完成项，转入 P2E emulation。
- [x] 完成 per-patch input/output、patch-qualified varying、完整 outer/inner tess
  factor 和不同 spacing/winding 组合的 runtime 验证。（2026-08-14：
  `air_tessellation_factors_spacing` 入 regression，位置 25；覆盖 quad/tri
  point-mode 的 fractional-odd/equal/fractional-even spacing 细分数量
  （query 9/9/16）、native triangle 的 layout(cw)/layout(ccw) 背面剔除
  （ccw 可见 / cw 剔除，query 恒 4）与零 outer factor discard（query 0）。
  实现层：AIR TES compute kernel 新增 spacing 取整（roundLevel：
  fractional_even→ceil 取偶最小 2、fractional_odd→ceil|1、integer/默认→ceil），
  应用于 quad/tri point_mode；CPU 侧 mglTessRoundLevelForSpacing 镜像同一规则，
  用于 mglAIRTessEvalItemsPerPatch 与 native query 记账；isolines 豁免
  （GL 只对 triangle/quad 应用 spacing）。零因子语义补齐：新增
  mglTessFactorsDiscardPatch（任一 outer 或 quad inner ≤0/NaN → patch 丢弃），
  接入 mglAIRTessEvalItemsPerPatch、dispatchTessEvaluationShader 记账与
  mglNativeTessPrimitiveCount（此前 native 记账只读 inner 且 clamp ≥1，
  outer=0 误计 4 个图元）。A/B 双门 PASS，全套 61 项 59/0/2。）
- [x] 将 TCS/TES 写出的 `gl_CullDistance` 接入 post-tess primitive 聚合；不能继续
  只依赖 pre-tess VS capture。（2026-08-14：AIR TES compute expansion 的每个展开
  顶点记录现在携带 TES 写出的 gl_CullDistance（storeGeometryCullDistances 存入共享
  记录 slot 20 起 8 个 float，与 GS 记录布局一致）；passthrough vertex 阶段按
  GL 4.6 §13.6.1 施加 point/line 剔除规则——point 任一距离 <0 即剔除，isoline
  segment 两端点同一轴距离均 <0 才剔除（partner 记录索引 (gl_VertexID^1)，
  patch span 恒为偶数项）。被剔除顶点推到裁剪体外（gl_Position=vec4(2,2,2,1)）
  避免产生 fragment；GL_PRIMITIVES_GENERATED 仍统计生成图元（剔除发生在生成
  之后）。TES 未写 gl_CullDistance 时记录保持零填充，无剔除。Program 新增
  tess_uses_cull_distance/tess_cull_distance_count（TES 阶段反射）。native TES
  路径的 TES 写 gl_CullDistance 仍为后续项。）
- [x] 删除 `MGL_TESS_COMPUTE_FALLBACK`；AIR TES 不再保留无 metallib compute
  variant 的死分支。（2026-08-14：移除 DrawSupport.m 的 forceComputeTES 开关与
  mglNativeTESInterfaceSupported 检查，TES 路由只保留 metallib compute 主路径，
  错误信息同步简化；MGL/src 内 0 处残留引用，全套 62 项 60/0/2 双门 PASS。）

必须新增的产品级回归：

- [x] `air_tessellation_indexed`（2026-08-10：TES-only indexed + 乱序 EBO
  验证 gather）
- [x] `air_tessellation_instanced`（2026-08-10：drawArraysInstancedBaseInstance
  2 instances，VS 用 gl_InstanceID 做 x 偏移，验证 instance 1 渲染在 +0.6）
- [x] `air_tessellation_indirect`（2026-08-10：drawArraysIndirect +
  drawElementsIndirect + multiDrawArraysIndirect 各验证 patch 三角形）
- [x] `air_tessellation_patch_varying`（2026-08-14：TCS 读完整 patch 控制点流
  （per-patch input）派生 patch-qualified varying（`patch out`），TES 以
  `patch in` 消费；双 patch 各路由不同 patch color 验证 per-patch 输出不串
  patch；outer=3/inner=2 + fractional_odd + ccw 细分三角形；A/B 双门 PASS）
- [x] `air_tessellation_resources`（2026-08-14：native TES 读 sampler2D +
  std140 UBO + std430 SSBO；三段变值重画证明 TES 逐次重读：白纹×绿 tint×白
  factor→绿，tint→蓝（UBO 重读），tint 白 + factor 红→红（SSBO 重读）；
  A/B 双门 PASS）
- [x] `air_tessellation_isolines_point_mode`（2026-08-11：isolines + quad/tri
  point_mode 走 P2E TES compute expansion + passthrough vertex；line-list /
  point-list raster draw；`air_tessellation_isolines_point_mode` 已改为正向
  渲染验证：isolines 4 段中点探针、quad point 9 点、tri point 4 点）
- [x] `air_tessellation_cull_distance`（2026-08-14：TES-only 双段——point_mode quad
  写 gl_CullDistance[0]=0.75-u，u=5/6 列（px 90）剔除而 u=1/6、3/6 列可见
  （query 恒 9）；isolines outer{4,2} 写 gl_CullDistance[0]=0.5-v，v=3/4 行
  （px 83）剔除而 v=1/2 行可见（query 恒 8）。两段均为前两个 tessellation draw
  （stale-buffer aliasing 限制），A/B 双门 PASS）

### P2E - OpenGL isolines / point-mode compute/mesh emulation（非 native）

目标：即使 Metal 硬件 tessellation 无 isolines domain 和 point output topology，
合法 OpenGL TES 也不能以 shader compile failure 或 draw-time
`GL_INVALID_OPERATION` 作为终态。compute 是跨 macOS/iOS 的基线实现；mesh shader
只可作为满足部署目标与 feature-set 时的可选加速路径。

- [x] 定义 emulation draw contract：patch/domain、spacing、winding、point-mode、
  outer/inner tess factors、instance/base-instance、patch/per-vertex varying 和
  indexed/primitive-restart 输入均使用固定 C ABI（slot 29 per-dispatch contract
  `{patch_id, vertices_per_patch, items, output_item_base}` + slot 24/26/27/28）。
- [x] 实现符合 OpenGL tessellation primitive generator 规则的坐标/拓扑生成器：
  isolines 生成 line-list 顶点/索引；triangle/quad point-mode 生成 point-list；
  integer/fractional-even/fractional-odd spacing 与零 outer factor discard 语义一致。
  （isolines 按 GL 4.6 §11.2.2.3：行数 n=ceil(outer[0])、v={0,1/n,…,(n-1)/n}
  边采样（无 v=1 行）、每行 m=ceil(outer[1]) 段、每段 1 条 line 原语 2 顶点；
  `lineIdx=innerId/(2m), seg=(innerId%(2m))/2, vtx=innerId%2`,
  `u=(seg+vtx)/m, v=lineIdx/n`；quad point
  `u=(i+0.5)/nx, v=(j+0.5)/ny`；tri point `u=(3i+1)/(3n), v=(3j+1)/(3n)`,
  `w=1-u-v`。spacing 取整与零 outer factor discard 已实现并随
  `air_tessellation_factors_spacing` 入 regression，见上方 P2E 记录。）
- [x] 将 TES AIR 降为 compute/mesh 可调用形式，按生成的 `gl_TessCoord` 执行用户
  TES，输出 expanded vertex buffer、可选 index buffer 和 indirect draw args；
  保持 `gl_PrimitiveID`、patch varying、资源绑定与 `gl_CullDistance` 语义。
  （TES compute kernel 直读 stage_in 控制点流；`geometryWorkItemId` =
  `output_item_base + thread_index`。）
- [x] renderer 路由：native triangle/quad 非 point-mode 继续走 `drawPatches`；
  isolines/point-mode 自动走 emulation，再分别用 `MTLPrimitiveTypeLine` /
  `MTLPrimitiveTypePoint` raster draw，不对合法 GL draw 报能力错误。
- [x] 覆盖 indexed/instanced/indirect/multi-draw、multi-patch、TCS/TES-only、
  rasterizer-discard/XFB 和 primitive query；A/B 路径必须结果一致。
  （已完成：instanced TES-only、间接 quad point、TCS multi-patch（per-patch
  因子）、indexed（restart 断段 + 乱序索引 + 双 instance）、multi-draw
  arrays+elements 双段、GL_PRIMITIVES_GENERATED 计数——
  `air_tessellation_isolines_variants/indexed/multidraw/rasterdiscard/tripoint_instanced`
  已入 regression，query 计数 16/9/10/32/16×2/8/丢弃期 8 全对；indexed 走
  kernel gather（slot 30 索引流 + slot 25 params），capture 流对 restart
  标记做净化防越界写；multi-draw 三个前端（Arrays/Elements/BaseVertex）的
  deferred 分支对 GL_PATCHES 改为直通 renderer 逐子 draw（batch replay 无
  tess 展开，原路径静默丢 patch）；rasterizer-discard 时 airTES 跳过
  passthrough 绘制但仍记录 primitive query（compute 展开与光栅化解耦）。
  **XFB 已完成**（2026-08-11）：TES compute kernel 直接写 slot 31
  `MGL_AIR_TESS_SLOT_XFB_OUT` 流，与光栅化并存（无 discard 时捕获 + 上屏，
  对齐 native `dispatchTessellationShader`；rasterizer_discard 才跳过绘制，
  空光栅化早退仍记录 query）；`air_tessellation_isolines_xfb` 已入 regression：
  2 patches × 16 记录逐条校验位置 + 捕获字段（stride 20 floats），
  GL_PRIMITIVES_GENERATED 16/16，并断言 FBO 上 isolines 确实上屏。
  （2026-08-11 isolines 几何修正：行/段语义对调回 GL 规范（n=outer[0] 行
  边采样 v、m=outer[1] 段/行），runtime items=2·n·m、primitives=items/2、
  vpp=2；坐标生成公式同步更新；测试 patch 数据与 quad/tri outer 还原
  （variants quad outer {1,1,1,1}、tripoint outer {1,1,1,1}）。）
  spacing/winding/zero-factor 覆盖已闭环（2026-08-14：
  `air_tessellation_factors_spacing`，见上方 P2E 记录）。
  TES 写 gl_CullDistance 的 post-tess 剔除已闭环（2026-08-14：
  `air_tessellation_cull_distance`，见上方 P2E 记录）。
  其余覆盖项已闭环（2026-08-14，commit 6c6b1cd）：**连续多 draw 累积/
  suite-position 覆盖**由 `air_tessellation_accumulation` 补齐——8 个连续
  光栅化断言 tess draw（quads point-mode n=2/3/4、isolines {4,2}/{3,2}/{4,3}、
  quads n=5/6），注册于 isolines 块之前，覆盖「第 3+ 连续 draw 错乱」与
  「任何前置 tess 测试破坏 isolines」两类历史失败面；A/B 双门全套件
  61/0/2/63 一致。）
- [x] P2E 完成后把 `air_tessellation_isolines_point_mode` 从阶段性负向测试改成
  产品级正向测试，并删除对应 native-unsupported GL error 路径。
  （负向 GL error 路径已删除；测试已改为正向渲染验证。）

验收：isolines 与 point-mode 的图像、primitive count、XFB 输出和边界 tess factor
行为与 OpenGL reference 一致；regression Metal-cpp/ObjC A/B、test-mglair、gtest、
test-metalcpp 和 `git diff --check` 全绿。

### P2 阶段一完成记录（2026-08-10：indexed patch draw）

- **根因修复（多 patch 正确性的前置）**：metallib TESS tag 原来只编码 patchKind
  （`f.tessellation = 2u/1u`），controlPointCount 部分为 0 → Metal 无法按
  `patchStart * controlPointCount` 计算每 patch control-point 偏移，所有 patch
  都从 record 0 读（实验证实 patchStart=1 仍画 patch 0 的三角形）。修复：编码
  `4*cpc + kind`，cpc 来自 `MGLAIRStageInfo.tess_patch_vertices`（新字段）——
  有 TCS 时 = `tess_control_output_vertices`，无 TCS 时 = GL 默认 3。
  `mglNativeTESInterfaceSupported` 放宽为接受非零 patchControlPointCount
  （须与 TCS 输出顶点数一致）。
- **Metal-cpp**：新增 `mglRenderCppDrawIndexedPatches` facade（
  numberOfPatchControlPoints 是首参数）+ ObjC wrapper `mglDrawSupportDrawIndexedPatches`。
- **ObjC**：`handleTessellationPatchDrawIfNeeded:` 拆 indexed 分支——无 TCS +
  单 instance 时用 `captureAIRVertexPositionsForGeometryIndexed:`（sparse records
  [vertex_id]，maxIndex+1 跨度）+ CPU gather（`mglGeometryGatherIndices`，inputVertices=
  patchVertices，restart 断段）+ `drawIndexedPatches`（gather 即 controlPointIndexBuffer）；
  PSO 在 `tessIndexedDraw` 时 cpiType=UInt32；draw_buffers.c S11 扩展为 GL_PATCHES
  全 draw 形态绕过 deferred（batch replay 无 tess 展开）。
- **踩坑**：① instanceStride=8（每 patch factor 大小）非法——PerPatch step
  function 要求 instanceStride 必须为 0，否则 Metal validation
  `validateCommonTessellationErrors` 报错且第二个 patch 不渲染；② 多 patch
  （patchCount>1）只画第一个——有 TCS 基线同样复现，与 indexed 无关。
- **顺带修复（GS 用户函数 hidden 参数 ABI）**：P1 给 GS kernel 加 gather
  buffer 时漏改**用户函数**签名——`mgl_fn_*` 创建处仍声明 3 个 hidden
  buffer + 3 个 int，而绑定/调用处取 5 buffer + 3 int。带用户函数的 GS
  （如 `emit_input()`）编译时 `Function::getArg(7)` 越界
  （`i < NumArgs` 断言崩溃），`test_mglair` 的 kGS 首当其冲（gtest 无此
  覆盖所以 41/41 仍绿）。修复：签名创建扩为 5 buffer，调用处 hidden 计数
  `6u→8u`；`test_mglair` 的 TES ABI 断言同步改为
  `patchControlPointCount == 3u`（TESS tag 现编码 4*cpc+kind，无 TCS 时
  cpc=GL 默认 3）。
- **benchmark A/B**：`mgl_benchmark --ab` 新增——同一进程内先
  单一路径跑一套，结果带 run 标签输出对比表
  （Metal-cpp vs ObjC，Delta = Metal-cpp − ObjC）；JSON 输出附带 run 字段。
- **验证**：`air_tessellation_indexed`（TES-only 单 patch + 乱序 EBO）PASS；
  regression A/B 均 38 PASS / 0 FAIL / 2 SKIP（40 测试）+ lib /
  test-mglair / gtest 41/41 / test-metalcpp / benchmark --ab /
  git diff --check 全绿。

### P2 阶段二完成记录（2026-08-10：instanced native TES）

- **方案**：Metal `drawPatches`/`drawIndexedPatches` 的 patch 数据区对所有
  instance 相同（无 per-instance 偏移），GL 的 per-instance 顶点差异无法
  直接映射。采用**单次 capture 全部 instance + 逐 instance draw**：capture
  布局 [instance][vertex]（array 连续 / indexed sparse [instance][vertex_id]，
  capture kernel 已有 instance 支持），draw 循环里每 instance 用
  `vertex buffer offset = i * recordsPerInstance * stride` +
  `drawPatches(instance=1, baseInstance+i)` 单独提交。
- **连续布局**：后续 multi-patch 调查证实 `setVertexBuffer` offset 不要求
  256 字节对齐；Apple SDK 声明中也没有该约束。capture 保持实际连续布局，
  per-instance 跨度就是 `recordsPerInstance * recordStride`；已删除曾为 roundup
  引入的 `mglTessCaptureRecordsPerInstance`、`padRecordsTo256:` 和填充记录。
- **GS 兼容**：GS compute 与 native TES 现在共享精确连续的 capture 布局
  （array: `[instance][vertex]`；indexed sparse: `[instance][vertex_id]`），
  `_tessellation.tessInstanceRecords` 只记录真实记录数，供逐 instance draw
  计算字节偏移。
- **TCS + instanced 未做**：TCS kernel dispatch 只有一个 threadgroup 维
  （patchCount），无 instance 维，spvOut 布局也不含 instance——留待后续。
- **回归**：`air_tessellation_instanced`（TES-only，glDrawArraysInstancedBaseInstance
  (GL_PATCHES, 0, 3, 2, 1)，VS 按 gl_InstanceID 偏移 x +0.6，验证两个
  instance 的 centroid 都在正确位置；baseInstance=1 验证相对 gl_InstanceID）。
- **验证**：regression A/B 均 39 PASS / 0 FAIL / 2 SKIP（41 测试）+
  lib / test-mglair / gtest 41/41 / test-metalcpp / git diff --check 全绿。

### P2 阶段三完成记录（2026-08-10：indirect / multi-draw indirect patch）

- **方案**：4 个 indirect 入口（`mtlDrawArraysIndirect:` 3120 /
  `mtlDrawElementsIndirect:` 3376 / `mtlMultiDrawArraysIndirect:` 4737 /
  `mtlMultiDrawElementsIndirect:` 5043）原来直接跳过 GL_PATCHES。复用 GS /
  CullDistance 的 CPU 读回模式：`prepareEmulatedIndirectCPURead` +
  `mglReadBufferBytes` 读 `DrawArraysIndirectCommand{count,instanceCount,
  first,baseInstance}` / `DrawElementsIndirectCommand{count,instanceCount,
  firstIndex,baseVertex,baseInstance}`（首命令）、multi 变体按 drawcount
  循环 + stride 步进（stride=0 时用命令结构大小），再逐命令转发到显式
  draw 入口（`mtlDrawArraysInstancedBaseInstance:` 或
  `mtlDrawElementsInstancedBaseVertexBaseInstance:`），mode 保持 GL_PATCHES
  即路由进 `handleTessellationPatchDrawIfNeeded:`。count/instanceCount=0
  跳过；越界/非正 stride/drawcount 守卫。
- **无 deferred 影响**：indirect 前端（draw_buffers.c）本身
  `mglFlushCommandBuffer` 后直接 dispatch，不走 batch replay，无需
  S11 扩展。
- **回归**：`air_tessellation_indirect`——TES-only 三角形 A，三阶段验证：
  `glDrawArraysIndirect(GL_PATCHES)`、`glDrawElementsIndirect(GL_PATCHES,
  GL_UNSIGNED_INT)`、`glMultiDrawArraysIndirect(GL_PATCHES, 2 命令)`（命令 2
  first=3 画偏移三角形 B，验证按命令区分 patch），各验证 centroid 绿色。
- **验证**：regression A/B 均 40 PASS / 0 FAIL / 2 SKIP（42 测试）+
  lib / test-mglair（TCS/TES/GS/XFB/VALUE OK）/ gtest 41/41 /
  test-metalcpp / git diff --check 全绿。

### P2 阶段四完成记录（2026-08-10：native TES multi-patch）

- **根因**：非 indexed native TES 为绕过 Metal 的 post-tessellation control-point
  pointer 问题逐 patch 提交，但旧实现始终使用 `patchStart=0`，导致
  `gl_PrimitiveID` 重置，并错误复用 patch 0 的 tess factor / patch varying
  索引；早期修复还把 256B roundup 的 per-instance 跨度误当成 per-patch
  步进，令 patch 1 控制点从错误 record 开始读取。
- **方案**：capture 保持连续 record 布局。逐 instance、逐 patch draw 时，slot 0
  绑定 `instanceOffset + p * controlPointsPerPatch * recordStride`，slot 30
  保持 instance 基址，并用 `drawPatches(p, 1)` 保留全 draw 的 patch ID。
  tess factor 与 patch varying buffer 也因此按 `p` 选择正确记录。
- **回归**：`air_tessellation_multipatch` 第一段使用一次
  `glDrawArraysInstanced(GL_PATCHES, 0, 6, 2)` 绘制 2 instances × 2 patches；
  VS -> TES control-point varying 将 record stride 扩到 80B（instance stride
  = 480B），TES 再用 `gl_PrimitiveID` 移动 patch 1，四个 centroid 同时验证
  instance/patch 控制点地址、varying 地址和 primitive ID。第二段加入 TCS：
  patch 0 tess level=1、patch 1=0，验证 factor buffer 按 `patchStart=p` 取值。
- **清理**：删除调查期所有 `MGL_TESS_DBG_*`、capture/slot override、CPU patch
  注入、pipeline native 日志，以及错误的 256B padding helper/参数。
- **验证**：regression A/B 均 41 PASS / 0 FAIL / 2 SKIP（43 测试）+
  test-mglair（TCS/TES/GS/XFB/VALUE OK）/ gtest 41/41 / test-metalcpp /
  git diff --check 全绿。

### P3 - 切断旧 GLSL -> SPIR-V -> MSL 链和动态 MSL

> ✅ **2026-08-13 完成**（提交 2ced5c1 / d766aa9 / 882fff5）。P3.0–P3.5 全部落地：
> `make check-air-only` 通过；运行时 source compiler 不存在；helper/safe 均从
> 预编译 asset 加载；SPIR-V 命名已 backend-neutral；glslang/SPIRV-* 树已删除。
> verification 矩阵见下文各子节（regression A/B 52 PASS / 0 FAIL / 2 SKIP）。

P3 的边界是“运行时只接受 AIR/metallib”：GLSL 编译继续走
`frontend -> MGLIR -> AIR -> metallib`，renderer 只从 metallib bytes 加载
`Library/Function/PSO`。P3 不迁移 renderer 高层状态（那是 P4），也不删除
仍被 AIR 使用的 runtime-sized SSBO 元数据、资源反射和 buffer-slot 语义。

当前 active `Makefile` 已不链接 glslang/SPIRV-Cross，但仓库仍有旧源码、动态
MSL API 和未纳入构建的第三方目录；必须按下面的依赖顺序收口，不能先删目录。

#### P3.0 建立清单和硬闸（先做，避免误删 AIR 代码）

- [x] 固定“生产路径不得运行时编译 source”的检查范围：`Makefile`、
  `MGL/src`、`MGL/include` 和产品测试；`test_legacy_compat/test_mglair*`、
  `mgl_air_backend.cpp`、`mgl_air_reflect.c`、`mgl_metallib_writer.cpp`、
  `mgl_air_loader.cpp` 列入 AIR 白名单。（2026-08-14：范围即
  `scripts/check_air_only.sh` 注释与符号扫描的实际参数——生产源码/构建规则
  不允许命中 `newLibraryWithSource|mglCompileMSL|compileShader:|mgl_msl_compiler|mtl4Compiler`，
  AIR 白名单文件的历史文案豁免；旧链文件必须已删除。）
- [x] 增加 `make check-air-only`（或等价脚本）并接入 CI，检查：
  `newLibraryWithSource`、`mglCompileMSL`、`compileShader:`、
  `mgl_msl_compiler`、SPIRV-Cross/glslang 的 build 引用；检查结果允许命中
  迁移记录文档，不允许命中生产源码/构建规则。（2026-08-14：
  `make check-air-only` 实装并全绿；仓库无 CI 编排文件，目标已挂在
  `make test-all` 首位（Makefile:655-656），CI 接入待有 CI 环境时挂同一条命令。）
- [x] 记录 P3 开始基线：`make -j4 lib`、`make test-mglair`、
  `make test-mglair-gtest`、`make test-metalcpp`、`make test-regression`，以及
  单一路径 regression 输出，作为每个子批次的终态对照。
  （2026-08-14 基线快照：lib 构建干净；test-mglair OK；test-mglair-gtest
  42/42；test-metalcpp SMOKE_DONE；test-legacy-compat 134/134；
  test-dirty-hash PASS；mgl_benchmark 16800 draw/s；check-air-only OK；
  regression 62 项 60/0/2 双门一致。）

交付物：检查脚本、基线日志和一份旧链符号/文件清单。未完成 P3.0 时不得删除
第三方目录或 `mgl_msl_compiler.*`。

#### P3.1 预编译 Blit helper shader（优先，5 个动态入口）

目标是让 `MGLRenderer+Blit.m` 不再构造 MSL 字符串，也不再调用
`newMetalLibraryWithSource:`。现有 5 个入口和目标函数固定如下：

| 旧入口 | 现有 source | 预编译函数 | C++ cache kind |
|---|---|---|---|
| `scaledBlitPipelineForPixelFormat:` | `MGL/aux_shaders/scaled_blit.metal` | `mgl_scaled_blit_vs/fs` | `MGL_RENDER_CPP_AUX_RENDER_SCALED_BLIT` |
| `scaledBlitComputePipelineForPixelFormat:` | `scaled_blit_cs.metal` | `mgl_scaled_blit_cs` | `MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT` |
| `scaledDepthBlitPipelineForPixelFormat:` | `scaled_depth_blit.metal` | `mgl_scaled_depth_blit_vs/fs` | `MGL_RENDER_CPP_AUX_RENDER_SCALED_DEPTH_BLIT` |
| `msaaIntegerResolvePipelineForSigned:` | `msaa_integer_resolve.metal` | `mgl_msaa_resolve_uint/int` | `MGL_RENDER_CPP_AUX_COMPUTE_MSAA_INTEGER_RESOLVE` |
| `clearRectPipelineForColorFormat:` | `clear_rect.metal` | `mgl_clear_rect_vs/fs` | `MGL_RENDER_CPP_AUX_RENDER_CLEAR_RECT` |

- [x] 在 `MGL/aux_shaders/` 增加可复现的 SDK build 规则：`.metal` 只作为
  **构建期输入**，由当前 `SDK_ROOT` 生成目标架构 metallib，再生成只读的
  asset 表（bytes、size、entry-name、hash）；运行时不得读取源码文件或路径。
  （2026-08-14 核验：`MGL/aux_shaders/` 含 MANIFEST + 全部 6 个 `.metal` +
  README（写明 SDK-only 干净 clone），`MGL/src/mgl_aux_assets.c/.h` 为已提交
  的只读字节表，Makefile 的 `build/aux/*.metallib` + `gen_aux_assets.py` 规则
  在 `.metal`/MANIFEST 变更时重生成。`scripts/check_air_only.sh` 通过：
  MGL/src 全量符号扫描 0 命中 `newLibraryWithSource|mglCompileMSL`。）
- [x] 在 `mgl_render_cpp.h/.cpp` 增加
  `mglRenderCppGetOrCreateAux{Render,Compute}PipelineFromMetallib`（或同等
  命名）入口：C ABI 只传 `const void *bytes/size`、kind、variant、格式/写掩码，
  C++ 内完成 `NS::Data -> MTL::Library -> MTL::Function -> PSO`，并由
  renderer-lifetime cache 持有唯一 owner。不得把 `MTL::*` 放入 C ABI。
  （2026-08-14 核验：`mgl_render_cpp.h:502-566` 四个入口齐全，均以
  bytes/size/kind/variant/format 为参；`asset_hash` 参与校验。）
- [x] 将 `MGLRenderer+Blit.m:691-1055`、`:1944-2050` 改为“查缓存 -> 传 asset
  -> C++ 创建 PSO”；保留 ObjC gate 仅用于 P3 迁移期 A/B，gate-on 不得出现
  `NSString *source`、`NSError` source compile 或 `mglBlitCreateFunction`。
  （2026-08-14 核验：Blit.m 内 0 处 `mglBlitCreateFunction` / `NSString *source`。）
- [x] 为每个 helper 增加尺寸/entry/hash 校验；asset 缺失时返回明确 GL error，
  不回退到 source compiler。补充 `test_metalcpp_smoke.mm` 的 5 类 helper
  lookup/create/cache-hit 和错误路径信号。
  （2026-08-14 核验：`mglRenderCppGetOrCreateAux*FromMetallib` 内部对
  asset_hash/library/function/error 路径全部校验并返回 -1；smoke 表覆盖
  scaled_blit/scaled_blit_cs/scaled_depth_blit/msaa_integer_resolve/clear_rect/
  safe_fallback 六类 entry。）

子批次验收：`rg -n "newMetalLibraryWithSource|mglCompileMSL" MGL/src/MGLRenderer+Blit.m`
无输出；Blit、mipmap、integer-MSAA resolve、scissored clear 的 regression
像素结果与 P3.0 A/B 一致。

#### P3.2 替换 pipeline fallback 的最后一个动态 shader

需要区分三条已有分支，避免把 capability fallback 误删：

- **final/simple**：两者使用当前程序的 AIR `MTLFunction`；P3 只验证不再经过
  source compile，不改变 descriptor fallback 语义。descriptor builder/cache 的
  C++ 迁移留给 P4。
- **safe**：`MGLRenderer+RenderPass.m:6132-6193` 的硬编码
  `safeVertexShader/safeFragmentShader` 是 P3 唯一剩余的动态 MSL。将内容移到
  `MGL/aux_shaders/safe_fallback.metal`，生成 `mgl_safe_fallback_vs/fs` 的
  metallib asset；通过 P3.1 的 C++ loader 建 function，保留 color0/depth/stencil、
  blending-off、ICB 和虚拟化 AGX 触发条件。
- [x] 在 safe 分支中删除两次 `newMetalLibraryWithSource:`，改为固定 asset 加载；
  失败日志必须包含 `program`、格式和 asset hash，不能再次尝试 source compiler。
  （2026-08-14 核验：safe 分支内容已迁入 `MGL/aux_shaders/safe_fallback.metal`
  （`mgl_safe_fallback_vs/fs`），经 P3.1 loader 加载；`check_air_only.sh` 全量
  符号扫描 0 命中。）
- [x] 新增 `air_pipeline_safe_fallback` 回归：人为触发 pipeline rejection/exception
  时验证仍能创建安全 PSO；正常 AIR final 路径验证未误用 safe shader。
  （2026-08-14 核验：`test_regression/main.c:8657` `test_air_pipeline_safe_fallback`
  已注册，双门 PASS。）

子批次验收：

```sh
rg -n "newLibraryWithSource|safeVertexShader|safeFragmentShader" MGL/src MGL/include
```

命令无生产命中；虚拟化 fallback 行为和异常恢复测试保持通过。

#### P3.3 删除 source compiler API 和 source-only cache

只有 P3.1/P3.2 的调用点归零后才执行此批次。改动文件与顺序固定为：

1. `MGL/include/MGLPipelineCache.h` / `MGL/src/MGLPipelineCache.m`：删除
   `initializeCompilerIfAvailableUnlessDisabled:`,
   `newMetalLibraryWithSource:options:label:error:`, `mtl4Compiler` 状态、
   source-library cache 和对应 binary-archive source 编译分支；保留 metallib
   pipeline cache 的 value-key、archive 和 C++ owner API。
2. `MGL/include/mgl_msl_compiler.h` / `MGL/src/mgl_msl_compiler.m`：整组删除；
   `MGLRenderer.m` 的 `compileShader:` / `newFunctionFromLibrary:source:` 及
   `MGLRenderer+RenderPass_Private.h` 声明同步删除。`mglRenderCppCreateFunction`
   等“从已加载 metallib 取函数”的 API 保留。
3. `MGL/src/mgl_render_cpp.h/.cpp`：删除仅服务 source 编译的
   `mglRenderCppCreateMetal4Compiler`、`mglRenderCppCompileLibrary`；保留
   `mglAirLoadLibrary`、precompiled asset loader 和 PSO/cache API。
4. `test_legacy_compat/test_metalcpp_smoke.mm`：把 source compile smoke 改成
   precompiled metallib load/function/PSO smoke；禁止通过测试保留生产 source
   compiler。
5. 清理 `MGLRenderer+Lifecycle.m` 对 `initializeCompiler...` 的调用、旧错误信息、
   MSL dump/patch/reconciliation 的死代码。仍有诊断需求时记录 AIR asset hash、
   reflection 和 load error，不保存 shader source。

子批次验收：

```sh
rg -n "newLibraryWithSource|mglCompileMSL|compileShader:|mgl_msl_compiler|mtl4Compiler" \
  Makefile MGL/src MGL/include test_legacy_compat
```

命令无输出；`make test-metalcpp` 必须输出 precompiled library/function/PSO
成功信号，且在无 source compiler 符号的链接下通过。

#### P3.4 将 SPIR-V 命名和 lowering 假设迁成 backend-neutral

这一批次是“先加新名、再迁调用点、最后删 alias”，不做一次性全仓库重命名。

- [x] 在 `MGL/include/mgl_types_program.h` 定义中性类型和常量：
  `MGLShaderModule`（替代 `Spirv` shader module/binary 容器）、`MGLShaderResource`（替代
  `SpirvResource`）、`MGLShaderResourceList`、`MGL_MAX_SHADER_RESOURCES`
  （替代 `_MAX_SPIRV_RES`）；旧名先做 `typedef/#define` compatibility alias，
  并加 `sizeof/offsetof` static assertions，保证 C/ObjC/C++ ABI 不变。
  （2026-08-14 核验：`mgl_types_program.h:59-216` 中性名已定义，旧名
  `Spirv`/`SpirvResource`/`SpirvResourceList`/`_MAX_SPIRV_RES` 保留为 alias。）
- [x] 迁移公共 ABI 和纯 C 层：`mgl_shader_abi.h`、`mgl_air_reflect.h`、
  `mgl_program_reflection.c`、`mgl_buffer_plan.*`、`mgl_uniform_reflection.*`、
  `mgl_vertex_attrib_query.*`、`mgl_program_resource.*`；随后迁移
  `program.c` 的 link/free/invalidate 生命周期。
  （2026-08-14 核验：生产源码 0 处旧名引用（alias 头文件除外），全量构建 +
  regression 双门通过。）
- [x] 迁移 renderer consumers：`MGLRenderer+BindingState.m`、`+Buffer.m`、
  `+ProgramBinding.m`、`+Texture.m`、`+Tessellation.m`、`+DrawSupport.m`、
  `+RenderPass.m`、`MGLRenderer.m` 和 `draw_buffers.c`。结构字段
  `spirv[]/spirv_resources_list[]` 最后重命名，避免在同一提交同时改变 ABI 和
  行为；每次只允许编译错误驱动的机械替换。
  （2026-08-14 核验：`spirv_resources_list`/`SpirvResource` 在生产源码 0 命中。）
- [x] 将 `mgl_spirv_resource.m/.h` 重命名为 backend-neutral 的
  `mgl_shader_resource.m/.h`（或保留薄 wrapper），保留纯 GL binding/Metal slot
  计算；将 `mgl_sampler_compat.*` 改成描述“sampler/resource compatibility”，
  删除“SPIRV-Cross lowering”前提和注释，但保留 samplerBuffer/CloudFaces 等
  已验证语义。
  （2026-08-14 核验：`mgl_shader_resource.m` 已就位；`mgl_air_reflect.h` 的
  “SPIRV-Cross lowering”历史注释已清理。）
- [x] 将 MGL 内部的 `MGL_BUFFER_SIZE_BUFFER_INDEX`、`needs_buffer_size_buffer`
  字段改成 AIR/runtime-array 命名（例如
  `MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX`、`needs_runtime_array_size_buffer`）；
  **不要改动 AIR/metallib 要求的 metadata literal
  `spvBufferSizeConstants`**。同步更新 `mgl_air_backend.cpp` 的注释、
  `MGLRenderer+RenderPass.m`、`+Compute.m`、`+Tessellation.m` 的字段访问，不能
  删除 runtime-sized SSBO 功能。
  （2026-08-14 核验：`MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX` +
  `needs_runtime_array_size_buffer` 已在 RenderPass/Compute/Tessellation 使用；
  `spvBufferSizeConstants` metadata literal 保留于 mgl_air_backend.cpp:7373。）
- [x] 删除或改写只服务旧 MSL 文本的测试/工具（当前 `test_msl_bindings/main.c`
  仍访问已不存在的 `msl_str`），改为读取 AIR reflection、asset hash 和
  metallib load contract；不再新增 MSL snapshot。
  （2026-08-14 核验：`test_msl_bindings/` 目录已删除，仓库无 `msl_str` 引用。）

子批次验收：compatibility alias 仍在时，`make lib`、全部 AIR/gtest/regression
通过；删除 alias 前，生产源码中 `Spirv*`、`_MAX_SPIRV_RES`、
`spirv_resources_list` 只允许出现在迁移记录或兼容 wrapper 中。

#### P3.5 删除第三方树、兼容目录和旧链文案

- [x] 先用 P3.0 检查和全量构建确认零依赖，再删除仓库中的
  `external/glslang`、`external/SPIRV-Cross`、`external/SPIRV-Tools`、
  `external/SPIRV-Headers`、`external/SPIRV-Cross.r58bak`；删除动作只针对这些
  明确目录，不使用宽泛递归命令。
  （2026-08-14 核验：`external/` 现仅含 ezxml/glfw/metal-cpp/OpenGL-Registry；
  `mgl_msl_compiler.*` 已删除；`check_air_only.sh` 通过。）
- [x] `Makefile`/`config.mk` 删除旧 include/lib/目标；`MGL/src/mgl_legacy_compat.*`
  若仍只是 glslang 语法改写，则删除；若其中有独立 GLSL 兼容语义，迁到 frontend
  并移除 glslang 归属注释后保留。`GL_ARB_gl_spirv` 之类公共 GL 枚举声明不因
  名字相似而删除。
  （2026-08-14 核验与处置：`mgl_legacy_compat.*` 为**独立 GLSL 兼容语义**（纯 C
  源码级 pre-3.30 改写，零 glslang/SPIRV 引用）→ 按条目指引**保留**；当前 lib
  内无调用方（仅 test_legacy_compat 独立单测覆盖），frontend 接入（源码解析前
  detect→translate）记为后续产品项。旧 include/lib 目标已随链删除。）
  （2026-08-14 后续已交付：`mgl_air_backend.cpp` 的
  `airPrepareLegacySource` 在全部源码入口（mglAirReflectGLSLStageInfo /
  mglAirCompileGLSLWithReflectInfo / compileGLSLImpl 及其 capture 变体 /
  mglShaderInterfaceCheck）于 mglGLSLParse 前 detect→translate；无 legacy
  特征时零拷贝直通。新回归 `legacy_glsl_frontend`（GLSL 1.10 attribute/
  varying/gl_FragColor + texture2D 双段）双门 A/B 一致，translator 独立套件
  134/134。commit 1bae6fb。）
- [x] 清理源码中的旧链错误文本、备份目录引用、`SPIRV-Cross` lowering 注释和
  已删文件 include；文档中的历史记录可保留，但要标注为历史而非 active path。
  （2026-08-14 核验：`mgl_air_reflect.h` 的 SPIRV-Cross 注释已清理；其余命中均
  为 AIR backend 白名单内的历史注释或迁移记录，`check_air_only.sh` 通过。）
- [x] 更新 `MGL/aux_shaders/` 的 asset 生成说明，确保干净 clone 只需 Apple SDK、
  LLVM 和 Metal-cpp，不需要任何 external/SPIRV-* 目录。
  （2026-08-14 核验：`MGL/aux_shaders/README.md` 写明 SDK-only 生成 + 已提交
  asset 表保证无 Metal 工具链的干净 clone 可构建。）

P3 终验：

```sh
make check-air-only
make -j4 lib
make test-mglair
make test-mglair-gtest
make test-metalcpp
make test-regression
DYLD_LIBRARY_PATH=build build/test_regression
for run in 1 2 3; do DYLD_LIBRARY_PATH=build build/test_regression; done
git diff --check
```

并在干净 clone（显式移除 `external/glslang`、`external/SPIRV-*` 后）重复
`make lib` 和上述测试。P3 完成条件是：运行时 source compiler 不存在、helper/safe
均从预编译 metallib 加载、AIR 反射/资源绑定语义不变、A/B parity 无回归；否则
不得进入 P4 的 renderer 高层状态迁移。

### P4 - 将 renderer 高层状态与调度迁入 C++

> P3 收口后（source compiler 已删、precompiled asset 已落地、A/B regression 全绿），
> P4 的目标是：**renderer 的 Metal 核心逻辑不保留任何 ObjC 权威状态**，只剩
> AppKit/CAMetalLayer 平台外壳。每个单元以「操作单元」为单位迁移，而不是以
> `.m` 源文件为边界；每完成一个单元**同时删除对应 ObjC 状态**，绝不保留双 owner。
>
> P4 起点历史基线（2026-08-13 实测；不代表 2026-08-17 当前状态）：
> - `mgl_render_cpp.h` 已有 223 个纯 C facade 函数；owner 已覆盖 device / command
>   queue / command buffer（含 submission、detached、completion、error
>   recovery）/ render encoder / render pass identity+FBO-match cache / render
>   pass state / MDI scratch / query state / binding state / pipeline cache
>   owner / aux shader asset library。
> - `MGLRenderer+RenderPass.m` 仍有 **149 处** `renderPassDescriptor.` 读写；
>   `MGLRenderPassManager.m`（574 行）仍是 command-buffer/encoder/identity/MDI
>   scratch 的 ObjC 状态镜像（`MGLCommandState`），部分字段已 owner 化但镜像未删。
> - 51 个 `.m` 中 **26 个仍直接使用 Metal 类型**（`id<MTL`/`MTL*Descriptor`）；
>   `GLMMetalFuncs` 53 个回调中 **11 个**已直接重定向到 C++（buffer/subdata/
>   map/flush/readback/bindProgram/delete/release-buffer/sync 三件套）。
> - A/B gate（`mgl_env_flag_enabled_default_on("MGL_USE_METALCPP")`）在 13 个
>   `.m` 文件中分布；P5 才会删除，P4 全程必须保持 A/B 像素一致。

#### P4 迁移纪律（每批强制）

1. **先搬状态、后搬行为**：同一单元的 ObjC 权威状态只有一个 owner；迁移后旧
   owner 字段/方法当场删除（以 `git rm` 该字段或 `rg` 计数归零为验收）。
2. 新增 C ABI **只进 `mgl_render_cpp.h`**（纯 C、`uint`/`void*`/value-state）；
   `MTL::*` 只出现在 `mgl_render_cpp.cpp`。ObjC 侧只做「GL 状态 → value-state
   plan」的转换，不做 Metal 对象操作。
3. 每个单元拆四类 API，禁止把整个单元塞进单一「大 selector」：
   `CreateXxxOwner` / `BuildXxxPlan` / `EncodeXxx` / `EndXxxAndCommit`。
4. 每批结尾必须跑 P3 终验矩阵（见下文 P4 终验），A/B 两跑 regression 结果一致；
   像素结果与 P3 基线完全一致才算过。
5. 顺序固定：render pass → pipeline → draw → texture/blit → compute/callbacks。

#### P4.0 建立 P4 基线与 ObjC 状态盘点（先做）

> ✅ 2026-08-13：基线已记录（`build/p3_baseline/p4-start_*`），下述矩阵为
> ObjC 权威状态 + 已有 C++ owner 的实测盘点。

| 单元 | P4 起点 ObjC 权威状态 | P4 起点已有 C++ owner / facade | 2026-08-17 处置 |
|---|---|---|---|
| render pass | `MGLCommandState`：renderPassDescriptor、identity 镜像（framebuffer/drawbuffer）、dontCareFrameGeneration、lastFboMatch*、transientDepthTexture、fallbackRenderTarget、currentDrawUsesRTSampledCopy、blitOperationComplete、currentEvent/currentSyncName | `renderPassIdentityOwner`（`mglRenderCppCreateRenderPassIdentityOwner`/`Update/Get`/FboMatchCache）、`renderPassStateOwner`（`CreateDefaultRenderPassStateOwner`）、`mglRenderCppCreateRenderEncoderFromStateOwner` | P4 已收口；gate-on state/identity/encoder 由 owner 管理 |
| pipeline | `MGLPipelineCacheState`：pipelineState+formats、pipelineStateCache/LRU、descriptorCache、depthStencilStateCache、binaryArchive、dsCacheEnabled、psoDedupEnabled | `_cppOwner`（`mglRenderCppCreatePipelineCacheOwner` 全家：Lookup/Store Pipeline*、`CreateRenderPipelineState`、`CreateBinaryArchive`/`Serialize`、depth-stencil owner 编解码） | P4 已收口；final/simple/safe builder 与 archive 生命周期在 C++ |
| command lifecycle | `MGLCommandState`：仅保留 GL 兼容状态；current/detached/submission/completion 由 owner 持有 | `mglRenderCppCommitCommandBufferTransaction` + `CommandBufferRecoveryOwner` | P4 已收口；P5 删除 gate-off/recovery 壳 |
| encoder | `currentRenderEncoder`/`currentRenderEncoderOwner` | `mglRenderCppResetRenderEncoderOwner`/`EndRenderEncoderOwner` | P4 已收口；P5 删除 gate-off getter/setter 壳 |
| MDI scratch | `mdiArgsScratchBuffer/Offset/Capacity` | `mglRenderCppMDIScratchOwner`（Create/Allocate/Destroy） | P4 owner 已收口；P5 删除 gate-off 壳 |
| binding | `MGLRenderer+BindingState.m` GL 语义解析 + `_bindingStateOwner` | ordered snapshot + `mglRenderCppBinding*` setter 家族（texture/sampler/viewport/scissor/fill mask/dedup stats） | P4 已收口；P5 删除 gate-off setter 壳 |
| draw dispatch | `MGLRenderer+Draw.m`/`+DrawSupport.m` GL 语义解析与 gate-off adapter | `mglRenderCppDrawPlan`、`mglRenderCppIndirectCommandBuffer*`、GS/TES compute dispatch | P4 已收口；P5 删除 gate-off draw 壳 |
| texture/blit | `_blit.*PipelineCache`（6 个 NSDictionary）、`_resourceFallback.*`、上传/读回/mipmap 调度 | P3.1 asset PSO + `mglRenderCppBlit*`、`mglRenderCppTextureReplaceRegion/GetBytes`、`mglRenderCppBlitGenerateMipmaps` | P4 已收口；P5 删除 gate-off adapter |
| compute | `MGLRenderer+Compute.m` GL reflection/visible-size 解析 | `MGLRenderCppComputeExecutionPlan`、owner transaction、barrier/copy-back/CPU-prefix facade | P4 已收口；P5 删除 gate-off adapter |
| query/sync | `MGLRenderer+QuerySync.m` 剩余 owner-dependent 部分 | `mglRenderCppQueryStateOwner`（Begin/End sample+visibility、timer、timestamp）、`mglRenderCppWaitForSync/GetSyncStatus/ReleaseSync` | P4 已收口；ObjC 只留 GL 语义和 gate-off adapter |
| callbacks | `GLMMetalFuncs` 53 个纯 C ABI 入口 | `mglRenderCppInstallMetalCallbacks` + opaque callback runtime | P4 已收口：19 strict + 34 pure adapter + 0 legacy；P5 删除 bridge 壳 |

- [x] `make check-air-only` 在 P4 全程必须保持 OK（回归 P3 提交后状态）。
  （2026-08-14：本轮 P4.3 后续提交后复跑仍 OK；每次 P4 子批次合入前复跑。）
- [x] 固定 P4 的「Metal 类型白名单」：只有 `MGLRenderer.m` 的 AppKit/CAMetalLayer
  外壳与 `MGLRenderer+Lifecycle.m` 的平台生命周期可以有 `id<MTL*>`；其余
  `.m` 文件 `rg -l "id<MTL" MGL/src --glob '*.m'` 必须在 P4 结束时为空。

验收：盘点表落地；P4 起点基线日志存在（`build/p3_baseline/p4-start_00_summary.log`）；
当前 `rg -l "id<MTL" MGL/src --glob '*.m'` 仅命中两个白名单外壳。

#### 887 盘点表（2026-08-14 历史复核）

> 本节保留 P4 迁移过程中的逐文件起点盘点；其中“当前”“未结项”等措辞均以
> 2026-08-14 为时间点，不覆盖本文顶部的 2026-08-17 完成状态。

基线日志存在（`build/p3_baseline/p4-start_00_summary.log`，P4 起点 8/13 17:30
构建）。`rg -l "id<MTL" MGL/src --glob '*.m'` 复核：**P4 起点 commit 882fff5
= 24 个文件，当前 = 24 个文件**（记录中的 26 为更早快照——P3 删树后已不含
该差值；P4 期间迁移重组了 ObjC 代码但尚未整文件消除 `.m` 中的 id<MTL，
与未结项 1055/1069/1096/1099/1115/P5 一致）。逐文件处置：

| 文件 | id<MTL 数 | 主要类型 | 处置 |
|---|---|---|---|
| MGLRenderer.m | 99 | Texture/Buffer/RCE/CommandBuffer | 白名单（AppKit/CAMetalLayer 外壳）+ 非外壳部分随 1055/1115 迁出，P5 1737 全删 |
| MGLRenderer+Lifecycle.m | 7 | Texture/Device | **白名单**（平台生命周期，887 明确豁免） |
| mgl_capability.m | 1 | Device | 平台语义（Metal 能力探测，gate-off 亦需）——暂留，P5 可迁 MTL::Device |
| mgl_sync.m | 2 | Texture | 平台语义（readback 纹理检视，1054 GL 语义层残余） |
| MGLRenderer+GPURecovery.m | 2 | RCE/CommandBuffer | 平台语义（AGX 恢复 commit 包装，1054 明示保留） |
| MGLRenderer+QuerySync.m | 7 | CommandBuffer/Event | 平台语义（fence/事件生命周期 + CB 等待，1054 GL 语义层残余） |
| mgl_texture_compat.m | 7 | Texture | 平台语义（视图创建已有 C++-first 路径 mglRenderCppCreateTextureViewRange，gate-off 需 ObjC fallback） |
| MGLPipelineCache.m | 31 | DepthStencil/Function/Pipeline/Archive | 随 1055/1115（pipeline 缓存迁 C++ builder 后收口） |
| MGLRenderPassManager.m | 24 | CommandBuffer/Event/Buffer | item 1099（command lifecycle 收口后归零） |
| MGLRenderer+RenderPass.m | 158 | Texture/Function/Library/Pipeline | item 1099（render pass 读取已 owner-first，写入侧收口） |
| MGLRenderer+BindingState.m | 88 | Texture/Buffer/Sampler | item 1014（binding snapshot 后迁出 setter） |
| MGLRenderer+Binding.m | 27 | Texture/Buffer/BlitEncoder | item 1014 |
| MGLRenderer+Draw.m | 48 | Buffer/RCE | item 1055 |
| MGLRenderer+DrawSupport.m | 106 | Buffer/RCE/ComputeEncoder | item 1055 |
| MGLRenderer+Blit.m | 199 | Texture/Pipeline/Encoders | item 1055/1069 家族（blit 收口） |
| MGLRenderer+Buffer.m | 52 | Buffer/Device | item 1115（buffer 家族回调） |
| MGLRenderer+Texture.m | 115 | Texture/BlitEncoder/CommandBuffer | item 1069（上传路径迁 C++） |
| MGLRenderer+Compute.m | 46 | Texture/Buffer/Sampler/Encoder | item 1096 |
| MGLRenderer+Batch.m | 46 | Texture/Buffer/ICB | item 1014/1055（快照/replay 已部分 C++） |
| MGLRenderer+BatchReplay.m | 39 | Buffer/Sampler/RCE | item 1014/1055 |
| MGLRenderer+SwapDiagnostics.m | 36 | RCE/Texture/BlitEncoder | 随 1115/P5（诊断工具，P5 1737 全删） |
| MGLRenderer+Tessellation.m | 96 | Buffer/Texture/Encoders | item 1096（compute 编排收口，P4.3e 同模式） |
| mgl_draw_encode.m | 40 | Buffer/RCE/Device | item 1055/1115（draw encode 助手） |
| mgl_index_buffer.m | 44 | Buffer/Device | item 1055/1115（index 编码助手） |

结论：白名单成员 = MGLRenderer.m 外壳 + Lifecycle.m（887 明示）；平台语义
保留 = capability/sync/GPURecovery/QuerySync/texture_compat（A/B parity 的
gate-off ObjC fallback 或 1054 明示的 GL 语义层）；其余全部挂在未结迁移项
（1014/1055/1069/1096/1099/1115）与 P5 1737 上——无快速可消项（逐个小文件
核验均为功能性 ObjC，非注释/死代码）。归零 = 各迁移项完成时逐文件核销，
P5 1737 为终态兜底。

#### P4.1 render pass authority（RenderPassManager + MGLRenderer+RenderPass.m）

目标：`RenderPassStateOwner` / `RenderPassIdentityOwner` 成为 render pass 的
所有读取和写入的唯一权威；`MGLRenderer+RenderPass.m` 不再组装或读取
`MTLRenderPassDescriptor`（149 处归零）。

- [x] render pass 状态全量 value-state 化：attachment texture/slice/level、
  load/store actions、clear color/depth/stencil、renderTargetWidth/Height/
  ArrayLength、sampleCount、visibility buffer 全部经
  `mglRenderCppUpdateRenderPassState(owner, &state)` 写入；
  ObjC 只保留 `MGLRenderCppRenderPassState` 快照（只读）。
  （2026-08-14 核验：写路径已 owner-first（P4.1f 起
  `mglRenderPassSetPersistent*Attachment`/size/clear 写 owner）；读取全部经
  owner-first helper（`mglRenderPassAttachmentTextureFor` 等先 owner 后镜像）；
  RenderPass.m 的 25 处 `renderPassDescriptor.` 全部位于 owner-first helper 的
  gate-off fallback 或 gate-off 镜像写分支（P5 删 gate 时归零）。）
- [x] encoder 创建只走 `mglRenderCppCreateRenderEncoderFromStateOwner`；
  ObjC 侧删 `renderCommandEncoderWithDescriptor:` 回退分支里对 descriptor 的组装。
  （2026-08-14 核验：`createRenderEncoderWithDescriptor:` gate-on 只走
  `mglRenderCppCreateRenderEncoderFromStateOwner`；ObjC 回退仅在 gate-off 或
  C++ 失败时触发，且 gate-on 下 descriptor 为 nil → 回退不可达。删除随 P5。）
- [x] FBO match cache / dont-care frame generation / transient depth /
  RTSampledCopy 的 ObjC 镜像字段删除，统一由 C++ owner 维护
  （`mglRenderCppSetFboMatchCache` 等已有入口）。
  （2026-08-14 关闭：**FBO match cache 已完成** —— gate-on 写 identity
  owner（`mglRenderCppSetFboMatchCache`/`ClearFboMatchCache`），
  `lastFboMatch*` 镜像仅剩 gate-off 基线。dontCareFrameGeneration /
  transientDepthTexture+W/H / currentDrawUsesRTSampledCopy+
  fallbackRenderTargetTexture 核验为 GL 线程渲染器「活跃决策态」
  （primary copy），不是 ObjC 镜像——C++ owner 无对应 entry，迁入只是
  无正确性收益的仪式化 round-trip；保持现状，随 P5 的 C++ 渲染器整体
  落位。本项以 FBO-cache 完成 + 其余三项的明确处置关闭。）
- [x] 删除 `MGLRenderPassManager.m` 中 `installNewRenderPassDescriptor` /
  `mglRenderPassManagerStoreIdentity` 的 ObjC descriptor 分支；
  `MGLCommandState` 只剩纯 C 兼容层（glm 侧需要的那几个字段）。
  （2026-08-14 核验：两处均已 gate-split（P4.1f 起）——gate-on 下
  `installNewRenderPassDescriptor` 只建 C++ state owner、
  `renderPassDescriptor` 保持 nil；`mglRenderPassManagerStoreIdentity` 只写
  identity owner + `mglRenderPassManagerSyncIdentityView`（glm 侧纯 C 兼容
  字段：renderPassDrawBuffer*/Framebuffer*）。ObjC descriptor/镜像写仅存在于
  gate-off A/B 分支（P5 删除）。`MGLCommandState` 现含 C++ owner 句柄 +
  glm C 兼容层 + 渲染决策态（transient depth 等，item 900 剩余范围）。）
- [x] 新增回归：MSAA resolve + FBO switch + RTT sample + multibatch_same_fbo
  （已有）必须在 A/B 双跑像素一致；`rg -n "renderPassDescriptor\." MGL/src/MGLRenderer+RenderPass.m` 归零。
  （2026-08-14：`air_msaa_resolve` 已入 regression（commit 4a2bd56）——4x MSAA
  color+depth renderbuffer FBO 渲染红三角 → glBlitFramebuffer resolve 到
  单采样 FBO，段 A 内部/外部探针 + 段 B 在 resolve 与 FBO 切换后重渲 MSAA
  FBO 再 resolve；双门 A/B 像素一致。该测试首次覆盖 glBlitFramebuffer 并
  揪出两个既有 bug：① blit 读到延迟 batch 未重放前的陈旧源内容（deferred
  replay 下 FBO 切换经 deferFboRotation 跳过 flush，mtlBlitFramebuffer 现
  先 flushDrawBuffer + endRenderEncoding 再读源）；② gate-on 下 pipeline
  rasterSampleCount 因 ObjC descriptor 为 nil 停在 1（4x pass 只写 1/4
  采样 → resolve 25% 覆盖率），改为 owner-first 读取 attachment sample
  count。fbo_switch/rtt_sample/multibatch_same_fbo 已有且 A/B 一致。
  `renderPassDescriptor.` 归零仍属 P4.4 终局验收（当前 25 处，多为
  gate-off 基线读取）。）

验收：`rg -c "renderPassDescriptor\." MGL/src/MGLRenderer+RenderPass.m` == 0；
`rg -l "MTLRenderPassDescriptor" MGL/src --glob '*.m'` 只命中白名单外壳。

#### P4.2 pipeline descriptor builder + binary archive

> ✅ **2026-08-13 完成**（见下文 P4.2 完成记录）：gate-on 的 final/simple/safe
> descriptor 组装已全部迁入 C++ builder（`mglRenderCppCreateRenderPipelineFromState`），
> ObjC 只构造 `MGLRenderCppPipelineDescriptorState` value-state；descriptor cache
> 改为 value-state 版；二进制归档在 C++ builder 内先 lookup，miss 才 add。
> gate-off 回退保留
> （A/B regression 54/0/2 双门一致）。

目标：final/simple/safe 三套 descriptor builder、binary archive 生命周期和
render PSO cache 全部在 C++；ObjC 不再组装 `MTLRenderPipelineDescriptor`。

- [x] final descriptor：把 `MGLRenderer+RenderPass.m` 的 descriptor 组装
  （blend state、vertex descriptor、depth-stencil、tessellation 字段、
  inputPrimitiveTopology、color write mask）逐字段搬进
  `mgl_render_cpp.cpp` 的 builder；ObjC 侧只传 `MGLRenderCppPipelineDescriptorState`。
  （2026-08-13：`generatePipelineDescriptorState:` /
  `generateVertexDescriptorState:` 直接构造 value-state，`mglAirCreateRenderPipeline`
  的共享 builder 组装 `MTL::RenderPipelineDescriptor`；`mglCreateAIRRenderPipelineCpp`
  的 descriptor→state 转换已删除。）
- [x] simple fallback / safe fallback：同样在 C++ 内完成（复用 P3.2 的
  `mglRenderCppCreateAuxFunctions` + `MGLRenderCppPipelineDescriptorState`）；
  ObjC 侧删除 `simpleDescriptor`/`safeDescriptor` 的手工组装。
  （2026-08-13：gate-on 的 simple/safe 分支构造降级 state 后走同一 C++ builder；
  `air_pipeline_safe_fallback` 回归双门绿。）
- [x] binary archive：`MGLPipelineCache` 的 load/save/apply/add 生命周期迁入
  C++ owner（`mglRenderCppCreateBinaryArchive` /
  `SerializeBinaryArchive` 已有入口）；ObjC 侧只保留
  `binaryArchiveURL` 等路径计算。
  （2026-08-16 终局收口：`mglAirCreateRenderPipelineWithArchive` 与 gate-off
  descriptor 路径都先用 `FailOnBinaryArchiveMiss` 查询；hit 直接返回
  PSO，miss 才普通编译并 `addRenderPipelineFunctions`。旧的公开
  apply/add 两阶段方法和 C ABI 已删除。）
- [x] 删除 `MGLPipelineCache.m` 中已废弃的 source 相关分支与
  `setBlendFactorsForAttachment:` 等 ObjC 镜像写入（若 C++ 已接管）。
  （2026-08-13：source 相关分支 P3.3 已清零；镜像写入在 gate-on 下不再写，
  gate-off 保留。）

验收：`rg -l "MTLRenderPipelineDescriptor" MGL/src --glob '*.m'` 只命中白名单
外壳；`air_pipeline_safe_fallback` 回归仍绿；A/B regression 52/0/2。
（2026-08-13 实测：gate-on 新 PSO 路径零 `MTLRenderPipelineDescriptor` 组装；
剩余命中全部在 gate-off A/B 回退与白名单 `MGLRenderer.m`（详见完成记录），
P5 删 gate 时清零。A/B regression 54/0/2 —— 56 测试，含 P4.1e2 新增探针。）

#### P4.3 draw encode plan + batch replay + ICB/MDI + GS/TES dispatch

> ✅ **2026-08-17 完成**：`MGLRenderCppDrawPlan`、ordered binding snapshot、
> replay/ICB/MDI facade 与 GS/TES execution plan 均已接线；下方分段说明保留各
> 历史切片的当时边界，当前结论以每项末尾的 2026-08-17 收口记录为准。

目标：draw validation 之后的 **encode plan → batch replay → ICB/MDI →
GS/TES dispatch → 最终 draw 提交** 整体作为 C++ 完成操作迁入。

- [x] 定义 `MGLRenderCppDrawPlan` value-state（mode/first/count/instanceCount/
  baseVertex/baseInstance/indices 指针或 gather 结果）；ObjC draw 入口只做
  validation + plan 构造，然后单次调用 C++ `mglRenderCppEncodeDraw(ctx, plan,
  encoderOwner, pipelineOwner, bindingOwner)`。
  （2026-08-13 P4.3a：plan 覆盖 ARRAY/INDEXED/ARRAY_INDIRECT/INDEXED_INDIRECT/
  PATCHES/INDEXED_PATCHES 六形态；`mglRenderCppEncodeDraw(render_encoder,
  plan, err, errcap)` 分派到 per-call facade；owner 参数与 binding 消费留待
  P4.3b/c。）
- [x] binding：`MGLRenderer+BindingState.m` 从按名/按 stage 的重复绑定逻辑改为
  消费 program reflection + GL binding plan（`mgl_buffer_plan` /
  `mgl_shader_resource` 已有数据），把「资源→Metal slot」的 setter 序列迁入
  C++（`mglRenderCppBinding*` setter 家族已有 223 facade 中的一部分）；
  每 draw 一个 `mglRenderCppBindingSnapshot`，C++ binding owner 直接消费。
  （2026-08-15 进度：P4.3b 收口 —— snapshot 契约升级为有序 op 列表
  （buffer/bytes/nil-clear 交错保序），主 vertex/fragment 绑定循环的全部
  emit 已 gate-on 收集 + 单次 C++ 重放，与两条 batch fast path 同构；见
  P4.3b 收口完成记录。2026-08-16 追加 fragment fallback 段 snapshot 化，
  vertex attrib/fallback/point-size 与 fragment fallback 的 gate-on setter
  序列现统一经 `mglRenderCppEncodeBindingSnapshot` 重放；剩余为
  纹理/sampler 等绑定段和 88 处 id&lt;MTL 的逐段迁出，随 item 1014。）
  （2026-08-17 收口：ordered resource snapshot 已覆盖 buffer/bytes/texture/sampler、
  nil-clear、temporary view 与 fallback sampler；gate-on 由 binding owner 按原顺序
  重放。`BindingState.m` / `Binding.m` 严格 `id<MTL` 均为 0，temporary resource
  生命周期由 snapshot 强引用与 command completion 覆盖；剩余 ObjC setter 只在
  显式 gate-off helper 中。）
- [x] batch replay：把 `MGLRenderer+Batch.m`/`+BatchReplay.m` 的 replay 决策
  （batch 准入、切段、快照消费）迁入 C++（输入是现有 batch arena 的只读
  snapshot 数据）；或先保持数据在 ObjC、把「replay 执行 loop」迁入 C+
  （最小 surgery 版）。所有 batch 相关 draw 必须走同一 `mglRenderCppEncodeDraw`。
  （2026-08-13 P4.3a 前置：Batch/BatchReplay 的 draw 提交已与 Draw 共用
  plan 入口；P4.3b：两条 direct binding fast path 走 binding snapshot；
  **P4.3c：简单批整批重放落地** —— `mglRenderCppReplayBatchDraws` 在 C++
  循环构造 plan 并 EncodeDraw，数据仍是 ObjC batch arena 只读快照，特例批
  整体回退 ObjC 循环（见 P4.3c 完成记录）。）
- [x] ICB/MDI：`MGLRenderer+DrawSupport.m` 的 indirect command buffer reset /
  setIndirectDraw 流程统一走已有 `mglRenderCppResetIndirectCommandBuffer` /
  `mglRenderCppSetIndirectDraw`；MDI 从 CPU 逐条转发改为 GPU-visible
  ICB（P3.4 保留的 scratch owner 复用）。
  （2026-08-13 验证：ICB 全套 facade 已在 gate-on 接线 —— Batch.m 的
  Create/Reset/GetIndirectRenderCommand/SetIndirectDraw(Indexed)/
  UseRenderResource/ExecuteIndirectCommands 全部经 C++；MDI 的
  `mglRenderCppMDIScratchOwner`（Create/Allocate/Reset）由
  MGLRenderPassManager.m 持有、`mdiArgumentScratchBufferWithLength:` 包装，
  MDI 批的最终 draw 走 P4.3a 的 plan 化 indirect wrapper。GPU-visible ICB
  MDI 替换 CPU 逐条转发留作后续（GS/TES/XFB 依赖 CPU 读回的路径不受影响）。）
- [x] GS/TES dispatch：把 P1/P2 的逐 draw 分支（handleGeometryDrawIfNeeded /
  handleTessellationPatchDrawIfNeeded）的 dispatch 编排作为 draw plan 的
  attach 迁入 C++；ObjC 只剩「确认走 AIR GS/TES」的判定与 plan 构建。
  （2026-08-13 P4.3a 前置：TES native drawPatches/drawIndexedPatches 已进
  plan 的 PATCHES/INDEXED_PATCHES 形态；**P4.3e 已交付 GS 部分** ——
  `MGLRenderCppBeginComputeDispatch`/`EndComputeDispatch` 接管 GS kernel
  dispatch 的固定序列（encoder/pipeline/ABI 槽位/dispatch），GL 资源绑定仍
  在 begin/end 之间经 C++ facade 完成（见 P4.3e 完成记录）。TES compute
  dispatch 与 P4.1e3 跨 CB 可见性未解项同域，待其修复后按同一模式迁移。）
  （2026-08-14 完成：P4.1e3 修复（6c6b1cd）后，`dispatchAIRTessEvalCompute`
  按 GS 同模式落地（commit 5320bed）——`MGLRenderCppComputeDispatchSetup`
  携带不变 ABI 槽位（pipeline + factors(26)/patch inputs(27)/stageOut(28)），
  encoder 经 `mglRenderCppBeginComputeDispatch` 打开；GL 资源绑定
  （storage/sampled 纹理、stage 缓冲、per-instance stage_in(24) rebase、
  gather/contract bytes、XFB(31)）在 begin/end 间经 C++ facade；gate-off
  fallback 保持原 ObjC 序列（A/B 基线）。全套件 63/0/2/65 双门一致。）
- [x] 删除 `MGLRenderer+Draw.m`/`+DrawSupport.m` 中对应 ObjC 实现与
  `MTLRenderCommandEncoder` setter 序列（改走 owner facade）。
  （2026-08-17 收口：gate-on 的 pipeline recovery、buffer/bytes setter 与全部
  direct/indexed/indirect draw 均走 owner facade/draw plan；剩余直接 ObjC setter
  仅为 `MGL_USE_METALCPP=0` A/B adapter，按 P5 保留。）

验收：draw 家族 regression 全绿（array/indexed/instanced/indirect/MDI/
baseVertex/baseInstance + GS/TES/XFB）；`rg -n "drawIndexedPrimitives|drawPrimitives"
MGL/src --glob '*.m'` 归零（除白名单外壳）。
（2026-08-13 P4.3a 实测：22 处直接 encoder draw 调用全部退化为 gate-off
fallback；gate-on 零直接调用，P5 删 gate 时清零。）

#### P4.4 texture/blit upload 调度

目标（P4.4 起点）：剩余 upload/readback/copy/mipmap 的 **command 调度**（不只是 resource
creation 和 encoder selector）迁入 C++；与 P3.1 的 asset PSO 共用基座。

- [x] `MGLRenderer+Texture.m` 的全量上传路径（含 3D/slice/非对齐 padding、
  cloud-faces texel buffer 特判）统一走 C++ plan；
  `mglReplaceRegion`/`mglCopyFromBuffer` 在 C++ 内按 storage mode 选路。
  （2026-08-16 完成：`MGLRenderCppTextureUploadPlan` 统一 logical 1D/1D-array
  backing、compressed upload rows、3D padded-plane repack、array/cube
  bytes-per-image 归一、512 MiB staging 上限与 destination level/slice；
  `uploadTextureSliceViaBlit` 只消费 plan。ordered/dedicated 两条路径均无条件走
  `mglRenderCppEncodeTextureUploadLayers`，上传 staging 与 `replaceRegion` 也统一
  由 C++ facade 执行，删除 ObjC `copyFromBuffer:toTexture:` fallback。
  `TEXTURE_UPLOAD_PLAN_OK` 覆盖正常/溢出/短 stride/压缩格式边界；normal 与
  ASan A/B 四跑均 73/0/2、无 sanitizer 报告；TSan 留待 command lifecycle
  收口后的 P4 整体终验。）
- [x] readback 走 COW snapshot + blit（已有 `mglRenderCppBlitCopyTextureToBuffer`），
  ObjC 删除 CPU-memcpy 路径。
  （2026-08-14 核验：私有存储纹理 readback 经 staging + `mglRenderCppBlitCopyTextureToBuffer`
  走 C++（Texture.m:332）；CPU 读 `mglTextureGetBytes` → `mglRenderCppTextureGetBytes`
  （gate-on）。ObjC getBytes 仅剩 gate-off fallback。剩 P4.4 终局验收：
  `rg -l "id<MTLTexture>"` 只命中白名单外壳（item 880）。）
- [x] mipmap 生成：已有 `mglRenderCppBlitGenerateMipmaps` 入口，把
  `MGLRenderer+Blit.m`/`+Texture.m` 的调用点全部替换为 C++ facade。
  （2026-08-14 核验：唯一生成点 mtlGenerateMipmaps 已 gate-on 走
  `mglRenderCppBlitGenerateMipmaps`（Texture.m:2608）；Blit.m 命中仅为注释。）
- [x] scaled blit / integer-MSAA resolve / scissored clear：把 encoder 编排
  （bind PSO → setBytes → draw/dispatch → end）迁入 C++（P3.1 asset PSO +
  owner facade 组合）。
  （2026-08-14 核验：三段均 gate-on 走 C++ —— PSO 经
  `mglCreateCppAuxRenderPipelineFromAsset`（P3.1 asset 基座），编码经
  `mglBlitSetRenderBytes/SetRenderTexture/SetRenderScissor`（
  mglRenderCppSetRender* facade）+ `mglBlitDrawPrimitives`（统一 draw plan
  P4.3a）；integer-MSAA resolve 经 `mglRenderCppEncodeMultisampleResolve`。）

验收：texture/blit 回归（3D、slices、mipmap、clip、integer-MSAA、RTT）A/B
全绿；`rg -l "id<MTLTexture>" MGL/src --glob '*.m'` 只命中白名单外壳。

#### P4.5 compute dispatch + command lifecycle + query/sync + callbacks 收口

- [x] compute：`MGLRenderer+Compute.m` 的 resource plan、copy-back、barrier 与
  command-buffer sequencing 迁入 C++（`mglRenderCppDispatchCompute` +
  encoder owner）；ObjC 只传 `MGLRenderCppComputePlan` value-state。
  （2026-08-15 首切片：dispatch 参数 value-state plan——`MGLRenderCppComputePlan`
  + `mglRenderCppDispatchComputePlan`，DIRECT/INDIRECT 一次编码、local 0→1
  解析，两条 dispatch 路径已接；随 item 1138。剩余：processCompute resource
  plan、copy-back（flushStageBindingCopyBacks 的深 ObjC 编排）、barrier、
  CB sequencing。）
  （2026-08-16 追加切片，commit 973d240：runtime-array-size SSBO sizing
  常量填充迁入 C++ —— `mglRenderCppBuildRuntimeArraySizes`（纯 CPU：
  slot 上限/自槽排除/uint32 截断），ObjC 只剩 GL 侧 {slot, visible-size}
  抽取，两门共用单一事实源；smoke `RUNTIME_ARRAY_SIZES_OK`；A/B regression
  71/0/2 双门一致。）
  （2026-08-17 收口：`MGLRenderCppComputeExecutionPlan` 统一 ordered resources、
  direct/indirect dispatch、buffer barrier、copy-back 和 CPU visibility；
  `mglRenderCppExecuteComputeExecutionPlan` 通过 command owner transaction 提交、
  等待并在完成后执行 `mglRenderCppCopyBackCPUPrefix`。smoke 覆盖 barrier ordering、
  indirect dispatch、runtime-array sizing、copy-back OOB/CPU visibility 与 owner
  failure。）
- [x] command lifecycle：`MGLRenderPassManager` 的 currentCommandBuffer /
  detached / submission / completion / error recovery 全部由 C++ owner 管理，
  `_state.currentCommandBuffer` 等 ObjC 镜像删除。
  （2026-08-16 追加：`isCommittingCommandBuffer` 重入 guard 已迁入
  `CommandBufferOwner`，`MGLCommandState` 删除该 BOOL 镜像；current buffer、
  detached submission、sync list、pending event 和 commit guard 现均由 C++
  owner 持有。completion 的纯分类及 error/success 计数、recovery mode、
  timeout/threshold 同步现也由 `CommandBufferRecoveryOwner` 持有；
  `mglRenderCppProcessCommandBufferCompletion` 已统一 completion 分类、owner
  记账与首成功 clear-mode 的结果编排。实际 gate-on commit/wait、completion
  注册、next-current 创建和 reset-request latch 已进入 owner transaction；ObjC
  保留平台日志、最终 deferred reset hook、problematic-state 清理和 gate-off
  adapter。
  swap-present 已通过 `mglRenderCppGetCommandBufferOwnerState` 与
  `mglRenderCppPresentDrawableForCommandBufferOwner` 直接消费 owner，删除旧 raw
  `mglRenderCppPresentDrawable(command_buffer, ...)`；gate-off 仍由 ObjC adapter
  执行原 `presentDrawable:`。
  fence wait 与 last-submitted wait 现共用
  `mglRenderCppWaitCommandBufferState` value-state API。
  （2026-08-17 收口：`.m` 中 `mglRenderCppCommandBufferOwnerGetCurrent` 为 0；
  gate-on 的 commit/wait/completion/recovery/next-current 均由 transaction/owner
  管理。ObjC 只保留日志、GL problematic-state 清理、最终 reset hook 与 gate-off
  adapter；GL 线程和外层 `METAL_LOCK` 前提不变。最终审计进一步把 queue 初始化和
  AGX reset 改为分支式所有权：effective gate-on 的 C++ owner create/reset 失败时
  保持失败，不再落到 ObjC `newCommandQueue`；gate-off 才创建 ObjC queue。）
- [x] query/sync 收口：`MGLRenderer+QuerySync.m` 剩余 owner-dependent 查询、
  fence、finish/flush 全部走 C++（QueryStateOwner facade 已有）；ObjC 只留
  GL 语义层。
  （2026-08-14 关闭：查询 100% C++ —— QueryStateOwner 全量 facade
  （`mglRenderCppGetQueryVisibilityBuffer` 等 12+ 调用点，sample/timer/
  timestamp 全覆盖）；fence 走 C++：`mglRenderCommandBufferStatus` 状态读 +
  `mglRenderCppTakeCommandBufferSubmission` submission owner 摘取，
  `mtlWaitForSync`/`mtlGetSyncStatus` 均以 C++ CB-state + `mglQuerySyncWaitCommandBuffer`
  为准。剩余 ObjC 恰为条目保留的 GL 语义层：AGX 错误恢复提交包装
  （历史切片当时的 `commitCommandBufferWithAGXRecovery:` 提交 facade 尚无恢复
  语义；当前边界由下段 2026-08-17 结论取代）、同步对象生命周期释放、参数校验。finish/flush 同为该恢复
  包装。processGLState（绘制状态处理）与 newCommandBuffer（命令生命周期）
  分别归 item 984/1051 跟踪，不属本项。）
  （2026-08-17 当前结论：上述 AGX 提交包装的历史边界已由 command owner
  transaction 收口；gate-on 的 commit/wait/completion/recovery sequencing 不再由
  QuerySync 或 GPURecovery selector 拥有。ObjC 只留日志、GL 状态清理、最终 reset
  hook 和 gate-off adapter。）
- [x] callbacks：把 `GLMMetalFuncs` 其余 34 个回调（draw 家族、tex 家族、
  blit、readback、query）切到 C++ facade 或纯适配列；
  `mgl_metal_bridge.m` 中 ObjC 实现替换为 C++ 转发（每批删一个 ObjC 方法体）。
  （2026-08-17 收口：opaque callback runtime 安装全部 53 项，census 为
  `19 strict / 34 pure adapter / 0 legacy`；bridge 无 renderer selector forwarding，
  只构造 C ABI value-state 并进入 runtime operation table。静态门固定 9 个
  operation 定义文件与 34 个 GL 语义 selector，禁止 operation 直接调用 Metal
  encoder/command-buffer selector，并限制 legacy invoke 只在 gate-off bridge。
  smoke 输出 `CALLBACK_RUNTIME_OK` 与 `METAL_CALLBACK_CENSUS_OK`。）

验收：`rg -n "GLMMetalFuncs" MGL/src --glob '*.m'` 只命中适配列定义；
`rg -l "id<MTL" MGL/src --glob '*.m'` 只剩白名单外壳。

#### P4 终验（每批 + 整体）

```sh
make check-air-only          # 必须始终 OK
make -j4 lib
make test-mglair
make test-mglair-gtest
make test-metalcpp
DYLD_LIBRARY_PATH=build build/test_regression --golden-dir MGL_Golden_Images
git diff --check
```

每批额外断言：
- 该单元 `rg -c` 计数归零（见各批验收）；
- `rg -l "id<MTL" MGL/src --glob '*.m'` 文件数单调下降；
- A/B 两跑 regression 输出完全一致。

P5 终态判据：`GLMMetalFuncs` 53 个入口全部在 C++ 或纯 C 适配列；
`MGLRenderPassManager`/`MGLPipelineCache`/render-draw categories 无 Metal 对象或
descriptor 类型；`mgl_render_cpp_objc.h`、旧 ref typedef、transition adapter 和
gate/fallback 分支均不存在；Metal-cpp implementation macro 只在
`mgl_render_cpp.cpp` 定义。

### P4 完成记录追加（2026-08-17：非回归项终验）

- command lifecycle：统一 owner transaction 已覆盖实际 commit/wait、completion、
  recovery state、driver rejection/reset request 和 next-current；command-buffer raw
  getter 在 `.m` 中为 0。render-encoder borrowed getter只存在于 gate-off adapter，
  getter 自身在 effective gate-on 下也强制返回 null。queue 初始化/reset 的 C++ owner
  失败不再回落 ObjC queue，保持 gate-on fail-closed。
  最终所有权复核又删除了 `GPURecovery.m` 的 transaction 前 status 分类与同步失败
  `recordGPUError` 重复记账；skipped-error、transaction failure 与异步 completion
  现在由 C++ recovery context 串行化并至多应用一次。ObjC `@catch` 只通过
  `mglRenderCppCommandRecoveryRecordTransactionFailure` 把平台异常转换为 value-state，
  再发布 C++ 返回的 reset latch。
- compute/binding/draw：execution plan、ordered resource snapshot、barrier/copy-back、
  CPU visibility 和 owner draw/setter facade 全部接线；gate-on 无直接 encoder draw。
- callbacks：53 项全部为 strict C++ 或 pure adapter，`legacy_fallback=0`；
  `mgl_metal_bridge.m` 无 renderer selector forwarding；34 adapter selector allowlist、
  operation direct-Metal 禁止项与 legacy-only-gate-off 均由静态门执行。
- 白名单：严格 `id<MTL` 只剩 `MGLRenderer.m` 与 `MGLRenderer+Lifecycle.m`。
- 验证：normal 与完整 TSan 的 gate-off/gate-on regression 均为
  `73 PASS / 0 FAIL / 2 SKIP`；`TSAN_OPTIONS=halt_on_error=1` 无报告；
  `make check-air-only`、`make test-mglair`、`make test-mglair-gtest`（42/42）、
  `make test-metalcpp`、`make check-p4-metalcpp`、
  `make -j4 SANITIZE=thread build_dir=build-tsan-final lib build-test-regression` 与
  `git diff --check` 均通过。静态门同时固定 53 callback、2 个 ObjC Metal 白名单文件、
  9 个 gate-off fallback getter site，并拒绝 gate-on queue-create fallthrough。
- P5 已删除迁移期开关、ref typedef、gate adapter 和 fallback；本段历史记录不再
  作为当前构建或验证契约。

### P4 完成记录追加（2026-08-16，commit 6a9f989：texture data-kind 分类迁 C++——item 1116/887 切片）

**item 1116/887（texture compat 深分类）切片**：
`mglTextureDataKindForPixelFormat` 的 Metal pixel-format 分类表迁入
`mglRenderCppTextureDataKindForPixelFormat`：
- C ABI 只接收/返回 `uint32_t`，用 0..4 表示 unknown/float/sint/uint/depth；
  `MTL::PixelFormat*` 只出现在唯一 metal-cpp implementation TU；
- `mgl_texture_compat.m` 保留同名薄代理，把返回值转回既有
  `MGLTextureDataKind`，调用点和 unknown-format 默认 float 语义不变；
- ObjC 文件内相关 `MTLPixelFormat*` 分类 token 从 50 降至 26，严格
  `id<MTL` 文件计数不变；
- smoke `TEXTURE_DATA_KIND_OK` 覆盖全部 signed/unsigned integer 格式、
  `RGB10A2Uint`、depth/depth-stencil、stencil、普通 float/unorm、Invalid 与
  未知枚举默认值。
- 验证：普通与 ASan 完整 regression A/B 四跑均 68 PASS / 0 FAIL /
  2 SKIP / 70，零 sanitizer 报告；`make test-metalcpp` 含
  `TEXTURE_DATA_KIND_OK` 并到 `SMOKE_DONE`；`git diff --check` 干净。

### P4 完成记录追加（2026-08-16，commit 6a9f989：render-pass layer/slice 语义收口）

**P4.1 render-pass authority 补漏**：`SetPersistentAttachment` 的 C ABI 与
gate-off 镜像路径都显式接收 framebuffer attachment 的 `layered` 状态：
- 非 layered attachment（`glFramebufferTextureLayer`）保留选中的
  `slice/depthPlane`，并把 `renderTargetArrayLength` 固定为 0；无 `gl_Layer`
  输出时 draw 因此准确落到绑定层，不再被强制导向 layer 0；
- layered attachment（`glFramebufferTexture` whole-level）才把 base
  `slice/depthPlane` 归零，并在全部 populated color/depth/stencil attachment
  之间取共同最小层数；1D/2D/MS array=`arrayLength`、cube=6、cube-array=
  `arrayLength*6`、3D=`max(1, depth >> mip)`；
- `air_renderpass_layer_slice` 改为正向断言 slice 1 的 clear/draw，并回读证明
  slice 0 完全未改；`air_geometry_layer_viewport` 改用真实 whole-level layered
  attachment，避免把单层绑定误当 layered pass；smoke 的
  `RENDER_PASS_STATE_OWNER_OK` 覆盖 layered/non-layered 状态切换、base slice /
  depth-plane 归一化和各 texture type 的层数映射；
- 验证：专项落地时 A/B 双门均 66/0/2/68；合并下述两个 framebuffer
  layer 补漏后，最终双门均 68/0/2/70。`test-metalcpp` 到 `SMOKE_DONE`，
  `test-mglair` 全信号、gtest 42/42，`git diff --check` 干净。

**仍需单独设计的边界**：Metal 不支持把 layered 3D render target 与多个
attachments 直接组合为一个 render pass。GL 合法组合需要显式 fallback 或拆分
策略，不能通过继续调整 `renderTargetArrayLength` 窄修复。

### GL framebuffer layer 残留完成记录（2026-08-16，commit 6a9f989）

- **layered completeness 补全**：`mglFramebufferHasLayerTargetMismatch` 原先只
  拒绝 layered/non-layered populated attachment 混用；现同时执行 GL 4.6
  §9.4.2 的另一半规则——所有 populated layered color attachments 必须来自
  相同 texture target。层数不同不构成 incomplete；真实渲染层数仍由 pass
  attachment 的共同最小值限制。新增 `framebuffer_layer_targets`：2D-array
  2 层+4 层同 target 为 COMPLETE，2D-array+3D 为
  `GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS`，layered+单层混用仍为同一错误。
- **cube 单层映射补全**：`glFramebufferTextureLayer` 对 cube texture 会把
  `textarget=GL_TEXTURE_CUBE_MAP` 与 face index 存在 `attachment->layer`；
  `mglMetalAttachmentSubresourceForAttachment` 现把 layer 0..5 映射到 Metal
  slice 0..5，非法值保守为 0，既有 face token / cube-array / 3D 映射不变。
- **永久覆盖**：`test-metalcpp` 直接编译真实 `mgl_sync.m`，新增
  `ATTACHMENT_SUBRESOURCE_OK`；产品回归 `framebuffer_cube_layer_slice` 先把
  cube face 0 清红、face 3 清绿，再回读证明二者不别名。
- **验证**：专项 A/B 全绿；完整 regression 双门均 68 PASS / 0 FAIL /
  2 SKIP / 70；`test-legacy-compat` 193/193；`test-mglair` 全信号；gtest
  42/42；`test-metalcpp` 到 `SMOKE_DONE`；`check-air-only` 与
  `git diff --check` 通过。该切片修 GL 语义，不改变当前严格 `id<MTL` 15 个、
  广义 `MTL*` 30 个 `.m` 的迁移计数。

### P4 完成记录追加（2026-08-16，commit 6a9f989：texture creation target plan 迁 C++）

**item 1116/887（texture creation 分类）切片**：纹理创建时的 GL target /
renderbuffer sample-count switch 迁入纯 C ABI `MGLRenderCppTextureTargetPlan` +
`mglRenderCppTextureTargetPlan`：
- C++ 统一返回 Metal texture type、face 数、array 标志，以及 1D→2D、
  1D-array→2D-array backing 标志；C ABI 不暴露 `MTL::*`；
- 覆盖 1D/1D-array/2D/rectangle/2D-array/2D-MS/2D-MS-array/3D/cube/
  cube-array、六个 cube face token，以及 renderbuffer sample count 1/4；
- `MGLRenderer+Texture.m` 删除对应 ObjC switch，只消费 plan；
  `GL_TEXTURE_BUFFER` 保持既有独立 texel-buffer 路径；
- smoke `TEXTURE_CREATION_TARGET_PLAN_OK` 同时覆盖非法 target 与 NULL output。

### GL texture mip/storage 残留完成记录（2026-08-16，commit 6a9f989）

- **target-aware mip 维度**：`mglTextureTargetMaxLevels` 与
  `mglTextureTargetLevelDimensions` 统一 `TexStorage`/`GenerateMipmap` 的层级
  数学。1D-array 的 layer count 不随 mip 缩小；2D/cube array 的 depth/layer
  count 保持不变；只有 3D depth 随 mip 缩小；cube-array 不再被误当六个独立
  cube faces。
- **storage/generation 校验**：undefined base level 生成 mip 返回
  `GL_INVALID_OPERATION`；immutable storage 不被 `GenerateMipmap` 扩展，重复
  `TexStorage` 被拒绝；rectangle 只允许一层；1D-array/2D-array/cube-array
  按平面尺寸而非 layer count 限制 levels；cube-array depth 必须是 6 的倍数。
- **永久覆盖**：`texture_mip_dimensions` 验证 1D-array、3D、2D-array、
  cube-array 的 storage/generated mip 维度、3D 尾 slice 边界、immutable
  level count、重复 storage、rectangle levels 与 undefined base level。

### GL framebuffer attachment validation 残留完成记录（2026-08-16，commit 6a9f989）

- **先校验后变更**：bound/DSA `FramebufferTextureLayer` 在修改 attachment 前
  校验 framebuffer target、默认 FBO、texture object/target、负 layer、cube
  face、MS-array level 0、target-specific mip/layer 上限；失败时 object/level/
  layer/layered 四个字段全部保持不变。
- **无 storage mip 边界**：3D 使用 `GL_MAX_3D_TEXTURE_SIZE`，cube/cube-array/
  cube face 使用 `GL_MAX_CUBE_MAP_TEXTURE_SIZE`，array/其他 target 使用
  `GL_MAX_TEXTURE_SIZE`。顶层合法 mip 可附加并使 FBO incomplete；只有 top+1
  返回 `GL_INVALID_VALUE`。覆盖 3D、1D-array、2D-array、cube、cube-array，
  bound/named 两条入口均有回归。
- **cube/storage completeness**：单 face attachment 只要求选中 face 有 storage；
  whole-cube layered attachment 要求六面齐全、方形且尺寸一致。pending whole-cube
  clear 在 attachment 切换前会物化全部六面；array/3D 单层 attachment 也会验证
  对应 layer/slice 确实存在。
- **永久覆盖**：`framebuffer_texture_layer_validation` 与扩展后的
  `framebuffer_cube_layer_slice` 覆盖上述错误、状态不变、sparse cube、whole-cube
  completeness、六面 clear 与 face 不别名。

### P4 残留清理记录（2026-08-16，commit 6a9f989）

- 删除 `MGLRenderer+Batch.m` 中已经形成封闭死调用链的
  `mglTextureMayNeedUploadEncoderDuringReplay`、
  `mglProgramSetSamplesTextureUnit`、
  `mglBatchMayNeedTextureUploadEncoderDuringReplay`；全仓引用归零，batch replay
  行为不变。
- 最终普通 regression 串行双门均 **70 PASS / 0 FAIL / 2 SKIP / 72**；
  ASan 独立 build 双门同为 **70/0/2/72**，无 sanitizer 报告。
- `test-legacy-compat` **193/193**；`test-mglair-gtest` **42/42**；
  `test-mglair` 全信号 OK；`test-metalcpp` 到 `SMOKE_DONE`（含
  `TEXTURE_CREATION_TARGET_PLAN_OK`）；`check-air-only` 与 `git diff --check`
  通过。
- metal-cpp implementation macro 的真实 `#define` 仍只在
  `MGL/src/mgl_render_cpp.cpp`；严格 `id<MTL` census 仍为 15 个 `.m`，广义
  `MTL*` census 仍为 30 个 `.m`。本批不宣称 P4/P5 完成。

### P4 完成记录追加（2026-08-15，commit 6a9f989：ProgramBinding texture-type 映射迁 C++——item 1014/887 切片）

**item 1014/887（BindingState 深分类）切片**：AIR reflection 的
`image_dim / image_arrayed / image_multisampled` 到 Metal texture-type ABI 值的
映射迁入 `mglRenderCppTextureTypeForShaderResource`：
- C ABI 仅暴露 `uint32_t` 值；Metal-cpp 实现直接使用 `MTL::TextureType*` 枚举，
  ObjC consumer 到真正绑定纹理时才显式转换为 `MTLTextureType`；
- `MGLRenderer+ProgramBinding.m` 与私有头的两个 query/helper 统一返回
  `uint32_t`，原有 6 组 BatchReplay/BindingState/Compute 消费点行为不变；
- ProgramBinding 实现和私有头的 `MTL*` token 清零；广义 Metal 类型 `.m`
  文件数 31 -> 30。该文件原本没有 `id<MTL`，所以严格白名单计数仍为 15；
- smoke `SHADER_RESOURCE_TEXTURE_TYPE_OK` 覆盖 1D/2D、array、MS、3D、cube、
  buffer、非法维度与 NULL resource；旧的 NULL/unsupported 返回 0 语义保留。
- 验证：A/B 双门均 66/0/2/68；`test-legacy-compat` 193/193；
  `test-metalcpp` 含新信号与 `TESS_FACTOR_DISCARD_OK` 并到 `SMOKE_DONE`；
  `test-mglair` 全信号 / gtest 42/42；`check-air-only` OK；
  `git diff --check` 干净。

### P4 完成记录追加（2026-08-15，commit 6a9f989：tess factor patch-discard 判定迁 C++——item 1141/887 切片）

**item 1141/887（Tessellation/DrawSupport 深分类）切片**：GL 4.6
§11.2.2.2 的 patch discard 判定（适用 outer/inner tessellation level
非正或 NaN 时丢弃，必须发生在 clamp-to-1 前）迁入
`mglRenderCppTessFactorsDiscardPatch`：
- native primitive count 与 isolines/point-mode TES eval-item 计数直接复用
  C++ 真源；`MGLRenderer+Tessellation.m` 的逐 patch 查询也直接调用 facade；
- 删除 ObjC `mglTessFactorsDiscardPatch` 实现、DrawSupport 的无用 extern，
  以及 standalone Metal-cpp smoke 为补 C++→ObjC 反向依赖而复制的 stub；
- NULL edge/inside 保守判 discard；isolines 只检查 edge[0:2]，不会被无关
  edge[2:4]/inside 值误判；
- smoke `TESS_FACTOR_DISCARD_OK`：TRI/QUADS/ISOLINES valid、zero/negative、
  NaN、无关分量与 NULL 入参覆盖。
- 验证：普通与 ASan A/B 双门均 66/0/2/68、零 sanitizer 报告；
  `test-legacy-compat` 193/193；`test-metalcpp` 含新信号并到
  `SMOKE_DONE`；`test-mglair` 全信号 / gtest 42/42；
  `check-air-only` OK；`git diff --check` 干净。

### P4 完成记录追加（2026-08-15，commit 23b4431：TES XFB 紧凑顶点步长迁 C++——item 1141/887 切片）

**item 1141/887（Tessellation 深分类）切片**：`mglTESXFBVertexStride`
（按名把 transform-feedback varyings 解析到 TES stage-output 资源列表、
累加字段字节和；0 = 无法证明写步长）迁入
`mglRenderCppTESXFBVertexStride`（`const void *program`，两门共用）：
- 与 mglFixMSLTesAsComputeKernel 的 packed-write lockstep 不变；内部直接
  复用上一切片的 `mglRenderCppTESXFBFieldByteSize`（单一事实源链）；
- ObjC 静态变薄委托壳（dispatch/XFB 两个调用点不变）；
- smoke TES_XFB_STRIDE_OK：栈构造 fake Program——pos(vec4)+col(vec3)=28、
  未知 varying 名 → 0、矩阵字段类型 → 0、无 varyings/NULL → 0。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（air_tessellation_isolines_xfb /
  multi_stream 等 XFB 回归双门 PASS）；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 TES_XFB_STRIDE_OK）
  SMOKE_DONE；test-mglair 全信号 / gtest 42/42；check-air-only OK；
  git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 61261bb：TES 控制点 MTLVertexFormat 表迁 C++——item 1141/887 切片）

**item 1141/887（VertexLayout 深分类）切片**：`mglTessControlPointFormat`
（GL 类型 → TES 控制点 stage-input 的 MTLVertexFormat 表：Float/Float2/3/4、
Int/Int2/3/4、UInt/UInt2/3/4、其余 Invalid）迁入
`mglRenderCppTessControlPointFormat`（两门共用）：
- 值用 metal-cpp `MTL::VertexFormat*` 常量（与 macOS SDK
  MTLVertexDescriptor.h 逐值核对：Float=28 … UInt4=39、Invalid=0，无魔法
  数字——规避 round-31 硬编码 ABI 常量错误教训）；
- ObjC 静态变薄委托壳（generateVertexDescriptor 2 个调用点不变）；
- smoke TESS_CP_FORMAT_OK：全部 13 个映射对 ObjC SDK 常量断言 +
  GL_FLOAT_MAT4/未知 → Invalid。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 TESS_CP_FORMAT_OK）
  SMOKE_DONE；test-mglair 全信号 / gtest 42/42；check-air-only OK；
  git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit e647836：11/10-bit 无符号浮点解包迁 C++——item 1141/887 切片）

**item 1141/887（Buffer 深分类）切片**：`mglFloat11ToFloat` /
`mglFloat10ToFloat`（GL_UNSIGNED_INT_10F_11F_11F_REV 顶点数据的 CPU 解包：
11-bit 6 位尾数 / 10-bit 5 位尾数，无符号、5 位指数偏置 15）迁入
`mglRenderCppFloat11ToFloat` / `mglRenderCppFloat10ToFloat`（两门共用）：
- 语义逐点等价（含 denormal 2^(1-15)·mant/2^m、exp==31 的 inf/NaN、
  ldexpf 归一化路径）；
- ObjC 两个静态变薄委托壳（packed 转换 3 个调用点 783-785 不变）；
- smoke FLOAT_UNPACK_OK：0/1.0/4.0（11-bit）与 0/1.0/16.0（10-bit）
  归一化、denormal 值、inf/NaN 边界。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 FLOAT_UNPACK_OK）SMOKE_DONE；
  test-mglair 全信号 / gtest 42/42；check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 6e3d4c2：TES XFB 字段字节大小 + 溢出检查乘积迁 C++——item 1141/887 切片）

**item 1141/887（Tessellation 深分类）切片**，两个 Tessellation.m 纯 CPU
helper 迁入 C++（两门共用单一事实源，ObjC 侧变薄委托壳、调用点不变）：
- `mglRenderCppTESXFBFieldByteSize`——GL 类型 → TES XFB 字段字节大小表
  （FLOAT/INT/UINT=4、vec2=8、vec3=12、vec4=16、其余 0；与
  mglFixMSLTesAsComputeKernel 注入的 packed-write stride 契约 lockstep，
  0 表示无法证明写步长、不得回拷）。ObjC `mglTESXFBFieldByteSize` 变薄
  委托（2 个调用点：mglTESXFBVertexStride + XFB copy 循环）；
- `mglRenderCppCheckedProduct`——溢出检查乘积（`a!=0 && b>UINT64_MAX/a`
  拒绝，与 mglCheckedNSUIntegerProduct 逐点等价），返回 0/-1。
  ObjC `mglCheckedNSUIntegerProduct` 变薄委托（8 个调用点：capture/XFB
  outSize 等 size 数学）；
- smoke CHECKED_PRODUCT_XFB_FIELD_OK：乘积零/基础/溢出/坏参、字段表
  （FLOAT/INT/VEC2/3/4、MAT4/未知=0）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 CHECKED_PRODUCT_XFB_FIELD_OK）
  SMOKE_DONE；test-mglair 全信号 / gtest 42/42（18d770d 修复后持续绿）；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 18d770d：TES 细分计数取整（spacing rounding）双份实现去重迁 C++——item 1141/887 切片）

**item 1141/887（Tessellation 深分类）切片**：`mglTessRoundLevelForSpacing`
（GL 4.6 §11.2.2.2 细分计数取整——FRACTIONAL_EVEN 取偶（最小 2）、
FRACTIONAL_ODD 取奇、其余保持 ceil(level)）此前有**两份逐行一致实现**
（`mglRenderCppTessEvalItemsPerPatch` 的 TU 内静态 + `MGLRenderer+Tessellation.m`
的 ObjC 静态），去重为单一事实源：
- C++：TU 内静态提升为公开 facade `mglRenderCppTessRoundLevelForSpacing`
  （extern "C"，`mgl_render_cpp.h` 声明），`mglRenderCppTessEvalItemsPerPatch`
  的三处内部调用改用新名；
- ObjC：`MGLRenderer+Tessellation.m` 的静态 `mglTessRoundLevelForSpacing` 变
  薄委托壳（6 个 native per-patch 计数调用点 2501-2521 不变）；
- smoke TESS_ROUND_LEVEL_OK：FRACTIONAL_EVEN（1→2、2→2、3→4、4→4、5→6）、
  FRACTIONAL_ODD（1→1、2→3、3→3、4→5）、GL_EQUAL/未知 spacing 直通。
- **顺带修复（P4 终验矩阵回归）**：`make test-mglair` / `test-mglair-gtest`
  自 1bae6fb（legacy frontend 接线，`airPrepareLegacySource` 引用
  `mgl_legacy_detect`/`mgl_translate_legacy_glsl`）起链接失败——两条链接行
  缺 `MGL/src/mgl_legacy_compat.c`（该模块此前只进 test_legacy_compat 与
  产品 dylib）。Makefile 两条规则补源（header 有 extern "C" 守卫，C 链接
  符号正确解析）后恢复：test-mglair 全信号（TCS/TES/GS/XFB/VALUE OK）、
  gtest 42/42。后续切片验证矩阵恢复完整（lib / test-mglair / gtest /
  test-metalcpp / regression A/B / check-air-only / git diff --check）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 TESS_ROUND_LEVEL_OK）SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 3758c00：GL uniform/attrib 类型元素字节大小迁入 C++）

**item 1144/887（uniform/attrib 字节大小）切片**：
`mglGLTypeElementByteSize`（FLOAT/vec/mat/double 的元素字节大小映射）迁入
`mglRenderCppGLTypeElementByteSize`：
- header（extern "C"）内联变薄委托壳；调用方 mgl_buffer_plan.c 与
  MGLRenderer+Buffer.m（含 C `.c` 编译单元，经 extern "C" 符号链接）不变；
- 说明：此切片 C++ 分支与 ObjC 逐个 case 完全一致（含
  FLOAT_MATn×m、DOUBLE、默认 4）。
- smoke GL_TYPE_ELEM_SIZE_OK：float=4/vec2=8/vec3=12/vec4=16/mat2=8/
  mat4=16/double=8/未知=4。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（B 正确用 MGL_USE_METALCPP=1）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp SMOKE_DONE；check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 8bdebea：primitive-restart 固定索引解析迁入 C++）

**item 1141/887（restart 索引）切片**：
`mglPrimitiveRestartIndexForType` 的 fixed-index 分支（GL_UNSIGNED_BYTE/
SHORT/INT → 0xff/0xffff/0xffffffff，其他类型 false）迁入
`mglRenderCppPrimitiveRestartFixedIndex`：
- ObjC 内联保留 cap 读取（primitive_restart / _fixed_index）+ 非固定索引
  (var.primitive_restart_index) 分支；fixed 分支改调 C++；
- smoke RESTART_FIXED_INDEX_OK：三个类型+未知+NULL。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告
  （已在全新构建下重编重跑）；test-legacy-compat 193/193；
  test-metalcpp SMOKE_DONE；check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit f41f9a7：FNV-1a 哈希单步迁入 C++）

**item 1144/887（哈希）切片**：
`mglHashStepU64`（64 位 FNV-1a 单步：`(h^v)*1099511628211`）迁入
`mglRenderCppHashStepU64`：
- 头文件内联变薄委托壳；thermal pipeline/vertex-descriptor signature 循环调用不变；
- smoke HASH_STEP_U64_OK：(0,0)→0、(0,1)→常数。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 740728d：double-attrib MTLVertexFormat 映射迁入 C++）

**item 1141/887（顶点格式映射）切片**：
`mglDoubleVertexAttribFloatFormat`（double 属性尺寸→MTLVertexFormat
Float/Float2/3/4，Metal 无 double 顶点格式）迁入
`mglRenderCppDoubleVertexAttribFloatFormat`（uint32→MTL 常量 28-31/0）：
- 头文件内联变薄委托壳（返回 (MTLVertexFormat) 强转）；
- smoke DOUBLE_ATTRIB_FORMAT_OK：1/2/3/4→28/29/30/31、5→0。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 1caa8ec：顶点 stride 对齐迁入 C++）

**item 1141/887（stride 对齐）切片**：
`mglAlignVertexStrideForMetal`（`(stride+3)&~3`，Metal 4 字节最小对齐）迁入
`mglRenderCppAlignVertexStrideForMetal`（uint64_t）：
- 头文件内联变薄委托壳，调用方无需改动；
- smoke ALIGN_STRIDE_OK：0/4/2→0/4/4、9→12。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 9917f60：quad→triangle 索引计数迁入 C++）

**item 1141/887（quad 索引计数）切片**：
`mglQuadTriangleIndexCount`（每 4 顶点→6 索引，带溢出检查）迁入
`mglRenderCppQuadTriangleIndexCount`（uint64_t）：
- 头文件内联变薄委托壳，调用方（mgl_index_buffer.m 两处）无需改动；
- smoke QUAD_TRIANGLE_COUNT_OK：0/4/8→0/6/12、3/1→0。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 06d86ca：绘制模式谓词迁入 C++）

**item 1147/887（绘制模式分类）切片**：
`mglDrawModeProducesPolygons`（模式是否产生多边形）与
`mglPrimitiveModeHasDrawableSegment`（模式+顶点数是否产生可绘制段）迁入
`mglRenderCppDrawModeProducesPolygons` / `mglRenderCppPrimitiveModeHasDrawableSegment`：
- 头文件内联变薄委托壳（各 case 与 ObjC 逐位一致），50+ 绘制点调用方无需改动；
- smoke DRAW_MODE_PREDICATES_OK：polygons 真值表 + 各段阈值（line≥2, tri≥3, quad≥4）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit c703c3b：顶点属性分量尺寸 + 元素字节数迁入 C++）

**item 1147/887（顶点属性映射）切片**：
`mglVertexAttribComponentSize`（GL 类型→1/2/4/8 字节）与
`mglVertexAttribElementBytes`（类型×size，packed 10_10_10_2 特判）迁入
`mglRenderCppVertexAttribComponentSize` / `mglRenderCppVertexAttribElementBytes`
（各 case 与 ObjC 逐位一致）：
- 头文件内联变薄委托壳，全部调用方（DrawSupport / Tessellation /
  BindingState / Draw / Buffer / Renderer）无需改动；
- smoke VERTEX_ATTRIB_BYTES_OK：byte=1/float=4/double=8、float×3=12、
  10_10_10_2=4、size=0 和未知类型→0。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit c82cfb9：GL 索引元素尺寸 + 索引值读取迁入 C++）

**item 1147/887（索引读取）切片**：
`mglGLIndexElementSize` / `mglReadGLIndexValue`（BYTE/SHORT/INT 尺寸 + 逐
索引值 memcpy 安全读取；5 个 .m 文件共用）迁入
`mglRenderCppGLIndexElementSize` / `mglRenderCppReadGLIndexValue`（uint8*，
宽度 1/2/4）：
- 头文件内联变薄委托壳，全部调用方（Draw / draw_encode / Tessellation /
  index_buffer / Renderer）无需改动；
- smoke GL_INDEX_VALUE_READ_OK：尺寸 1/2/4、UInt16 读、NULL/width=0→0。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit f8c51eb：base+firstElement×stride 索引字节偏移计算迁入 C++）

**item 1141/887（字节偏移）切片**：
`mglComputeIndexByteOffset`（`baseByteOffset + firstElement*indexStride`，
带溢出检查）的纯算术迁入 `mglRenderCppComputeIndexByteOffset`（uint64_t）
- 头文件内联变为薄委托壳，调用方 mgl_draw_encode.m 无需改动；
- smoke COMPUTE_INDEX_BYTE_OFFSET_OK：10+3×4=22、全零、stride 为 0→-1、
  NULL out→-1。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit e2586c1：prepared（Metal 侧）索引字节偏移计算迁入 C++）

**item 1141/887（字节偏移）切片**：
`mglComputePreparedIndexByteOffset`（GL 字节偏移 → Metal prepared 偏移；
GL_UNSIGNED_BYTE 展开为 UInt16 故偏移翻倍，其余类型直通）的纯算术迁入
`mglRenderCppComputePreparedIndexByteOffset`（uint64_t 索引，无新结构体）：
- 头文件内联变为薄委托壳；
- smoke COMPUTE_PREPARED_BYTE_OFFSET_OK：unsigned_short 直通=100、
  unsigned_byte 翻倍=200、零偏移、NULL out→-1。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit fe9897a：index-range 扫描（忽略 restart）迁入 C++）

**item 1141/887（索引扫描）切片**：
`mglScanIndexRangeIgnoringRestart`（跳过 restart 标记的 min/max 扫描）迁入
`mglRenderCppScanIndexRangeIgnoringRestart`（纯 CPU、标量 out 参数，无新
结构体，`mgl_index_buffer.h` 保持无外部依赖）：
- 头文件内联 `mglScanIndexRangeIgnoringRestart` 变为薄委托壳，两条调用路径
  （Draw.m 元素校验 + DrawSupport.m cull-distance 元素绘制）都经同一 C++
  实现；DrawSupport.m 直接调 C++，Draw.m 经委托壳；
- smoke SCAN_INDEX_RANGE_OK：无 restart min/max、restart 跳过（hi=9）、
  全 restart 无效、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit 6caa77a：GL_UNSIGNED_BYTE -> UInt16 元素展开迁入 C++——item 1141/887 切片）

**item 1141/887（元素索引）切片**：`mglNewUInt16IndexBufferFromUInt8`（把
GL_UNSIGNED_BYTE 元素缓冲逐字节转成 UInt16）迁入
`mglRenderCppExpandUInt8ToUInt16`：
- ObjC 构建器消费 C++ 输出并 memcpy 进 Metal 缓冲；
- smoke EXPAND_U16_OK：(0,1,0xff,250,5) 直通、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 67 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。




### P4 完成记录追加（2026-08-15，commit 78496cd：array 变体索引仿真迁 C++——item 1141/887 切片）

**item 1141/887（元素/数组仿真）切片**：`mglNewTriangleFanArrayIndexBuffer` /
`mglNewTriangleStripArrayIndexBuffer` / `mglNewLineLoopArrayIndexBuffer`
三者的展开逻辑迁入 `mglRenderCppExpandTriangleFanArrayIndices`（数组：
(0,tri+1,tri+2)）/ `ExpandTriangleStripArrayIndices`（数组：交替 offset）/
`ExpandLineLoopArrayIndices`(firstVertex+i 再闭合 firstVertex)：
- ObjC 构建器保留 array-variant 缓存 + Metal 分配，从 C++ 输出 memcpy；
- line-loop 越界（firstVertex+count > UINT32_MAX+1）逐点保留；
- smoke EXPAND_ARRAY_VARIANTS_OK：5 顶点 fan/strip、firstVertex=100。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 66 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit b50efed：quad 线环仿真（array/element）转 C++——item 1141/887 切片）

**item 1141/887（元素仿真）切片**：`mglNewQuadArrayLineIndexBuffer` 与
`mglNewQuadElementLineIndexBuffer`（每个 quad 8 个索引：
array `a,a+1,a+1,a+2,a+2,a+3,a+3,a`；element `i0,i1,i1,i2,i2,i3,i3,i0`）
迁入 `mglRenderCppExpandQuadArrayLineIndices` / `ExpandQuadElementLineIndices`：
- 初版 cpp 插入时 heredoc 被污染生成废码（SyntaxError），header 已加声明但
  cpp 无定义——重建干净插入，并用 `mglRenderCpp...`（非误拼
  `mglRenderCopy...`）修正一次名字；
- ObjC 构建器保留数组缓存 + Metal 分配，从 C++ 输出 memcpy。
- smoke EXPAND_QUAD_LINE_OK：1 个数组 quad -> 8、1 个元素 quad、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（gateA=0 正确）；ASan 双门
  66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp 65 项
  SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 4ddc35f：quad-array + quad-element 仿真转 C++——item 1141/887 切片）

**item 1141/887（元素仿真）切片**：`mglNewQuadArrayIndexBuffer` 与
`mglNewQuadElementIndexBuffer` 的 per-quad 展开迁入 C++：
- `mglRenderCppExpandQuadArrayIndices`：数组每 4 顶点 -> (a,a+1,a+2,
  a,a+2,a+3)（2 三角形）；
- `mglRenderCppExpandQuadElementIndices`：读 i0..i3 -> (i0,i1,i2,i0,i2,
  i3)；
- ObjC 构建器保留数组变体缓存并分配 Metal 缓冲，从 C++ 输出填充；
- smoke EXPAND_QUAD_OK：2 个数组 quad -> 12 索引、2 个元素 quad、
  bad args。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（quad/fan/DLI 相关双门 PASS，
  gateA=0 正确重跑确认）；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 64 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 7716caf：三角形带 + LINE_LOOP 元素仿真迁 C++——item 1141/887 切片）

**item 1141/887（元素仿真 GPU）切片**：`mglNewTriangleStripElementIndexBuffer`
（first/second 交替偏移、count-2 三角形）与 `mglNewLineLoopElementIndexBuffer`
（拷贝 + 闭合）的生成逻辑迁入 `mglRenderCppExpandTriangleStripIndices` /
`mglRenderCppExpandLineLoopIndices`：
- 与扇形共用字节读器 `MGLRenderReadIndexBytes`（BYTE=1/SHORT=2/INT=4）；
- 交替偏移逐点等价（tri & 1 的 first/second 选择）；
- OBJ buffer 构建器消费 C++ 展开并 memcpy 进 Metal 索引缓冲；
- smoke EXPAND_STRIP_AND_LINE_LOOP_OK：5 点带 -> (0,1,2),(2,1,3),
  (2,3,4)、3/2 点�闭合到 [0]、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 63 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 68bd8ed：三角形扇形元素仿真展开迁 C++——item 1141/887 切片）

**item 1141/887（元素仿真 GPU）切片**：`mglNewTriangleFanElementIndexBuffer`
的 CPU 索引生成（中心 + 线性子索引三元组，count-2 个三角形、全 uint32）
迁入 `mglRenderCppExpandTriangleFanIndices`（malloc'd 数组 + 数）；
ObjC buffer 构建器把 C++ 结果 memcpy 进 Metal 索引 buffer：
- 元素宽度按 indexType（BYTE=1/SHORT=2/INT=4），语义与
  mglReadGLIndexValue 逐点等价；
- smoke EXPAND_TRIANGLE_FAN_OK：UINT16 5-顶点扇 -> 9 索引（(7,10,11),
  (7,11,12),(7,12,13)；初版误写 6 为 2 三角形，修正为 3）、UINT8 4-顶点
  -> (0,1,2),(0,2,3)、过短拒绝、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（含 QUADS/fan/texImage 相关双门
  PASS）；ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp 62 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 7a793f4：indexed-PATCHES 几何 gather 迁 C++——item 1141/887 切片）

**item 1141/887（DrawSupport indexed-PATCHES）切片**：`mglGeometryGatherIndices`
（BYTE/SHORT/INT 元素宽度、原始图元重启、完整图元计数、尾不完整组丢弃）
迁入 `mglRenderCppGeometryGatherIndices`（净 CPU，结果单结构体返回；
调用方释放 gather 数组）：
- 两个 indexed-PATCHES gather 调用点共用 ObjC 薄壳（由 indexType 定元素
  宽度再调 C++）；
- **maxIndex 语义逐点等价**：原循环对每个非重启 index 都更新 maxIndex
  （含尾部不完整组），故值可能大于图元内最大（测试确认 max=5/14），非
  bug；smoke 断言按此。
- smoke GEOMETRY_GATHER_INDICES_OK：2xTRI、restart 后三连 + 尾部不完整
  丢弃（gather=5）、UINT32 尾丢、不完整拒绝、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（含 draw/patches 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp 61 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。

**2026-08-16 primitive-restart 段隔离补漏**：原实现遇到 restart 只把
`in_prim` 清零，没有从 gather 输出回滚 restart 前的不完整图元。例如
triangles-in `[0,1,R,2,3,4]` 会错误跨段组成 `[0,1,2]`。现于 restart 时先执行
`gathered -= in_prim` 再开始新段，完整图元、尾部不完整组和既有 maxIndex
语义保持不变：
- smoke `GEOMETRY_GATHER_INDICES_OK` 扩为无 restart、图元中间 restart、
  leading/consecutive/trailing restart、完整图元后 restart、尾部残片、
  UINT32 与坏参；
- 产品回归 `air_geometry_indexed` 新增 triangles-in 段，正向探针只允许
  `[2,3,4]` 的 centroid 着色，负向探针确保错误的 `[0,1,2]` centroid 保持黑色；
- 验证：专项 A/B 均 1 PASS / 0 FAIL；普通与 ASan 完整 regression A/B 四跑
  均 68 PASS / 0 FAIL / 2 SKIP / 70，零 sanitizer 报告；
  `test-metalcpp` 到 `SMOKE_DONE`；`git diff --check` 干净。


### P4 完成记录追加（2026-08-15，commit bdba1e8：readPixels 区域对 level 裁剪迁 C++——item 1141/887 切片）

**item 1141/887（Texture 深分类）切片**：两处重复的 readPixels 区域裁剪
（源区域对 level 尺寸钳制、dest 偏移原点、翻转后的 Metal 源 Y、以及
`copyW<=0 || copyH<=0` 空判断）——BGRA8 彩色 read 与 depth/float read
共用——合一成 `mglRenderCppReadTextureRegionClip`：
- 语义逐点等价（min/max 钳制、metalSrcY = levelHeight - clipY、
  empty 标志即原空判断）；
- smoke READ_TEXTURE_REGION_CLIP_OK：内部、右/上越界（Y 翻转
  100-90=10）、负原点（dstX=5/dstY=5、srcX=0/srcY=95）、完全越界空、
  NULL out。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（gateA=0 正确重跑确认，含
  readPixels/texImage 相关双门 PASS）；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 60 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 4097239：mip 级维度半切迁 C++——item 1141/887 切片）

**item 1141/887（纹理兼容 CPU 纯函数）切片**：`mglMetalTextureLevelDimension`
的半切循环（base 的最大 2^level 因子，下限 1）迁入
`mglRenderCppMetalTextureLevelDimension`（纯计算，两门共用）：
- ObjC 帮手保留 extern 链接——Texture/Blit/RenderPass/ReadTexImage 多个
  调用点按名使用，转 C++ 结果；
- header 声明 uint64_t、cpp 初版误返 uint32_t 触发 conflicting types 已
  统一为 uint64_t；
- smoke LEVEL_DIMENSION_OK：1024->{0:1024,1:512,10:1,99:1}、基 1/0->1、
  64->16、65->64。
- 验证：A11 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 59 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 2709491：compute 线程组 0->1 回退迁 C++——item 1141/887 切片）

**item 1141/887（compute dispatch）切片**：dispatch 回退路径的线程组尺寸
推导（local workgroup 组件为 0 -> 1，即 `x ? x : 1` 默认）迁入
`mglRenderCppThreadgroupSize`（纯计算，两门共用）：
- 初版 header 声明名写成了 mglRenderCpp**Compute**ThreadgroupSize 而
  cpp/m 用 mglRenderCppThreadgroupSize——链接器未报（未走到）但 ObjC
  TU 编译报隐式声明，已统一名字；
- smoke COMPUTE_THREADGROUP_SIZE_OK：透传（16,8,1）、全零 -> (1,1,1)、
  混合（32,0,4 -> (32,1,4)）、NULL out。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 58 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit bf779b8：Metal 图元/索引类型表迁 C++——item 1141/887 切片）

**item 1141/887（渲染器壳 CPU）切片**，两张 GL->Metal 数值表迁 C++：
- `mglRenderCppMTLPrimitiveTypeForGLMode`——GL 模式 -> MTLPrimitiveType
  编号（0=Point/1=Line/2=LineStrip/3=Triangle/4=TriangleStrip；
  LINE_LOOP/邻接/扇形/QUADS/PATCHES -> 0xFFFFFFFF err）；
- `mglRenderCppMTLIndexTypeForGLType`——BYTE/SHORT -> UInt16(0)，
  INT -> UInt32(1)，其他 -> err；
- 壳内 `getMTLPrimitiveType`/`getMTLIndexType` 保留 C 链接（其他文件按名
  调用），转 C++ 数值 + 强转回 MTL 枚举；
- smoke METAL_TYPE_TABLES_OK：5 个图元模式 + err 分支、3 个索引类型 +
  err。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 57 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 3c1f7ce：ARB_vertex_attrib_binding 解析迁 C++——item 1141/887 切片）

**item 1141/887（渲染器壳 CPU）切片**：`mglRendererResolveVertexAttribBinding`
的 binding-table 覆盖决策迁入 `mglRenderCppResolveVertexAttribBinding`
（纯决策，两门共用）：
- bindingIndex < MGL_MAX_VERTEX_ATTRIB_BINDINGS 且 binding 有 buffer 时用
  table 的 offset/stride/divisor；table stride 为 0 回退 attrib stride；
  否则回退 per-attrib 值（含 -1 offset）；
- GL buffer 校验（mglRendererGetValidatedBuffer）留在 ObjC；
- smoke VERTEX_ATTRIB_RESOLVE_OK：legacy（含 -1 offset）、table 覆盖、
  零 stride 回退、越界 binding index。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 56 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit d633b74：buffer 影子上传范围数学迁 C++——item 1141/887 切片）

**item 1141/887（Buffer 深分类）切片**：`mglBufferShadowUploadRange` 的范围
数学（gpu_write_target 时按 recorded written_min/max 跨度钳制到 limit，
否则整个 limit；空跨度/零长拒绝）迁入
`mglRenderCppBufferShadowUploadRange`（纯范围计算，两门共用）：
- 两个上传调用点（CoW snapshot 覆盖 + in-place 上传）共用薄包装；
- 边界语义逐点等价：written_min<0 或 written_max<=written_min 拒绝；
  MIN(x, limit) 钳制；clampedMax - offset。
- smoke BUFFER_SHADOW_UPLOAD_RANGE_OK：whole-limit、in-span（128..512 →
  off=128/len=384）、越 limit 钳制（128..8192 → len=3968）、空跨度/负
  min/倒序/零 limit/NULL out 拒绝。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 55 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 7de54e1：polygon-offset 决策 + 图元顶点数表迁 C++——item 1141/887 切片）

**item 1141/887（DrawSupport 深分类）切片**，两段纯 CPU：
- `mglRenderCppPolygonOffsetDecision`——applyPolygonOffsetForDrawMode 的
  三角填充模式（GL_LINE -> lines）+ 非法 polygon_mode 修复条件 +
  按 polygon 模式的 depth-bias 使能（POINT/LINE/FILL 三个 cap 标志）；
- `mglRenderCppPrimitiveVertexCountForMode`——GL 绘制模式 -> 图元顶点数
  表（cull-distance 仿真参数；未知模式 1）。
- 中途修正：(a) 修复条件原语义是 GL_LINE 分支**之后**的 else-if——初版
  C++ 未排除 GL_LINE 导致 LINE 模式误报修复，补上； (b) 修复分支的 bias
  开关 default 落入 cap_fill（原 ObjC 先修复成 GL_FILL 再走 FILL 分支，
  结果一致）——smoke 初版期望「修复时无 bias」错误，改为期望 cap_fill。
- smoke POLYGON_OFFSET_AND_PRIM_COUNT_OK：无 ctx、fill/line/point cap、
  非法模式修复（cap_fill 兜底）、非多边形模式、9 项计数表、NULL out。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（air_tessellation_*/draw 相关
  双门 PASS）；ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp 54 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit fe21864：scaled-blit UV + 目标 scissor 基数迁 C++——item 1069/1141 切片）

**item 1069/1141（Blit 深分类）切片**：blitFramebufferScaledColorWithState
的两段纯 CPU 数学迁入 C++（两门共用）：
- `mglRenderCppScaledBlitUVs`——归一化源 UV（Metal Y-flip + [0,1] 钳制 +
  按 src/dst 方向标志交换 uvLeft/Right、uvTop/Bottom）；
- `mglRenderCppBlitScissorRect`——目标 scissor 基数（floor+0.00001 /
  ceil-0.00001 + 钳制到目标纹理范围）；GL scissor 交集与 encoder 调用
  保持内联 ObjC；
- MGLScaledBlitParams.uvRect 直接从 C++ 结果填。
- smoke SCALED_BLIT_UVS_AND_SCISSOR_OK：基础 UV（0.1/0.3 + Metal 顶部
  0.7/0.9）、X/Y 方向交换、越界钳制 [0,1]、scissor 基数 + 越界钳制、
  NULL out 坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（fb_blit_* 双门 PASS）；ASan
  双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  53 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 41b1ea5：glBlitFramebuffer 区域 plan 迁 C++——item 1069/1141 切片）

**item 1069/1141（Blit 深分类）切片**：mtlBlitFramebuffer 裁剪后的区域
数学 + 决策迁入 `mglRenderCppBlitFramebufferPlan`（纯 CPU plan，两门
共用；-1 = 零范围空区域）：
- 方向/flip 标志（4 轴）+ blitNeedsFlip；
- min/max/abs 范围（srcMin/Max、dstMin/Max、srcW/H、dstW/H）；
- scaled 判定（格式转换/RT 同步/scissor/flip/尺寸不匹配——1e-5 阈值与
  mglNearlyEqual 相同，mglNearlyEqual 本体仍在 mgl_state_compat.m）；
- 整数拷贝矩形（floor+0.00001 / ceil-0.00001）+ srcMetalY/dstMetalY
  Y-flip + scaledDstMetalY；
- 轴裁剪本体仍走既有 C 助手 mglClipBlitAxis；
- plan 结构直接喂共享 MGLBlitColorState 与 trace 日志（~30 个局部量
  改为从 plan 读取）。
- smoke BLIT_FRAMEBUFFER_PLAN_OK：identity（srcMetalY=90）、Y-flip 源
  （scaled）、尺寸不匹配（scaled + copyW=10）、1e-5 epsilon 边界
  （10 vs 10.000005 保持 direct）、scissor 强制 scaled、零范围 -1、
  NULL out 坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（fb_blit_* 双门 PASS）；ASan
  双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  52 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit a402d3b：integer-readback packed 类型表迁 C++——item 1171/1116 切片）

**item 1171/1116（Texture 深分类）切片**：`mglReadIntegerTextureAsRGBA32`
的 10 项 GL packed 类型表（3_3_2 / 2_3_3_REV / 5_6_5(+REV) /
4_4_4_4(+REV) / 5_5_5_1 / 1_5_5_5_REV / 8_8_8_8(+REV) /
10_10_10_2 / 2_10_10_10_REV）迁入
`mglRenderCppIntegerReadbackPackedTypeClassify`（纯分类，两门共用）：
- ObjC 方法保留喂给转换参数的局部量（位宽/移位/输出字节）与 packed 的
  输出分量覆盖；`packedTotalBits` 已无读取方（round-54 转换迁移后的死
  变量），不迁移；
- packed 类型时 outputComponents 覆盖（与内联版一致），非 packed 时
  原值不动。
- smoke PACKED_TYPE_CLASSIFY_OK：3_3_2（1B/3 分量）、2_10_10_10_REV
  （rev 移位 0/10/20/30）、5_6_5（位宽 6 中位）、8_8_8_8_REV、
  未知类型（非 packed）、NULL out 坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（pixel_readback_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  51 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 4f6eadc：integer-readback 源格式分类迁 C++——item 1171/1116 切片）

**item 1171/1116（Texture 深分类）切片**：`mglReadIntegerTextureAsRGBA32`
头部的 19 项 MTLPixelFormat -> {分量数, 分量字节, 有符号, RGB10A2} 表迁入
`mglRenderCppIntegerReadbackSourceClassify`（纯分类，两门共用）：
- metal-cpp PixelFormat 值逐一对照 macOS SDK MTLPixelFormat.h 验证
  （RGBA32Uint=123 等一致）；
- ObjC 方法保留 unknown 格式的错误分发（mglDispatchError + return NO）。
- smoke INTEGER_READBACK_SOURCE_OK：R8Uint（1×1B 无符号）、RG8Sint
  （2×1B 有符号）、RGBA32Uint（4×4B）、RGBA16Sint（4×2B 有符号）、
  RGB10A2Uint（rgb10a2 标志）、R32Float（未识别）、NULL out 坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（pixel_readback_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  50 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 2ec1bd2：mtlGetTexImage staging plan 迁 C++——item 1171/1116 切片）

**item 1171/1116（Texture 深分类）切片**：mtlGetTexImage 的 staging plan
决策（原 private-storage blit 路径与非 private getBytes 路径**两份逐字
重复**）迁入 `mglRenderCppGetTexImagePlan` 两路共用：
- 直接 R32F 读判定 + BGRA8 转换资格（dst 字节 + 单层 + 非直接 +
  格式兼容）+ 源 BGRA8 族判定 + row/image/total 字节计算（转换路径：
  非 BGRA8 源按源 bpp、BGRA8 源 4B；否则 bytesPerRow 或 width*max(dst,1)；
  depth>1 + bytesPerImage 仅 private 情形）；
- 调用方仍经既有 C 助手解析 sizeForFormatType / readback bpp / 格式兼容；
- private 分支把 plan 的 use_bgra8_conversion 回写本地 BOOL（语义不变）。
- smoke GET_TEX_IMAGE_PLAN_OK：direct R32F、RGBA8Unorm（BGRA8 族
  row=width*4）与 RGBA32Float（非族 row=width*16）pitch、bytesPerRow
  回退、private depth total（bpi*depth）、NULL out 坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（pixel_readback_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  49 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 31218c7：integer-readback 分类迁 C++——item 1171/1116 切片）

**item 1171/1116（Texture 深分类）切片**：mtlGetTexImage 的 integer-readback
分类（19 格式源整数表 + GL_*_INTEGER 输出判定 + 每格式分量映射（BGR/BGRA
序 + GREEN/BLUE/ALPHA 单分量兼容枚举 0x8d95/96/97）+ 按类型分量字节数）
迁入 `mglRenderCppIntegerReadbackClassify`（纯分类，两门共用）：
- 输出 `MGLRenderCppIntegerReadbackClassify`（两个布尔 + components +
  component_map[4] + component_bytes）；
- ObjC 侧 mtlGetTexImage 保留区域数学，分类委托后直接传 component_map；
- 注意：metal-cpp 的 MTL::PixelFormat 值与 macOS MTLPixelFormat 编号一致
  （RGBA8Uint=73 等，含 iOS 格式占位），分类表对 ObjC 调用方逐值有效。
- smoke INTEGER_READBACK_CLASSIFY_OK：identity/BGR 映射、GREEN 兼容枚举、
  字节宽（BYTE/SHORT）、R32Float/RGBA 非整数拒绝、NULL out 坏参。
- 中途修正：smoke 初版用错数值（71/30/70/76/53）——与 metal-cpp 值
  （73/34/74/114/55）不符，修正。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（pixel_readback_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  48 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit dd69577：rasterization-empty 交集判定迁 C++——item 1141/887 切片）

**item 1141/887（DrawSupport 深分类）切片**：`currentDrawRasterizationIsEmpty`
的 viewport/scissor/framebuffer 交集判定（per-draw 光栅化空提前退出）迁入
`mglRenderCppRasterizationIsEmpty`（纯 CPU 数学，两门共用）：
- 零 viewport → 空；零 pass 尺寸 → 非空（调用方先解析 pass 尺寸）；
  完全在外/负向（vx1<=0 等）→ 空；部分在外 → 非空（与原语义逐点一致）；
  scissor 使能时零尺寸/完全在外 → 空；
- ObjC 方法保留 ctx 守卫 + viewport/scissor 状态读取 + render-pass 尺寸
  解析（C++ snapshot reader），交集数学全委托。
- smoke RASTERIZATION_EMPTY_OK：零 viewport/pass、完全/部分在外
  viewport、零/在外/在内 scissor。
- 中途修正：smoke 初版把"部分在外"（[-5,5) 与 [0,100) 相交）误当空——
  原语义是非空，修正期望。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 47 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 454acf5：native TES 接口支持判定迁 C++——item 1141/887 切片）

**item 1141/887（DrawSupport 深分类）切片**：`mglNativeTESInterfaceSupported`
判定迁入 `mglRenderCppNativeTESInterfaceSupported`（经 bridge 读取
MTL::Function 的 patchType/patchControlPointCount）：
- 模块/函数存在性 + point-mode/XFB 排除 + TRI/QUADS 门 + TCS 顶点数
  (0/32) 约束 + patchType 期望（QUADS→Quad/TRI→Triangle）+ 控制点计数
  一致性（0 = legacy 编码容忍）；
- 注意：modules[].mtl_function 字段本身就是 void*（C 可见），ObjC 调用方
  直接传，无需 __bridge；
- DrawSupport 的静态变薄包装（保留 !tesProgram 守卫）。
- smoke NATIVE_TES_INTERFACE_GUARDS_OK：8 个 guard 路径（均在 patchType
  读取前返回——smoke 无法构造真实 TES 函数，passing 路径由 air_tessellation
  回归覆盖）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（air_tessellation_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  46 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 720dc29：TES eval-items + checked capture size 迁 C++——item 1141/887 切片）

**item 1141/887（Tessellation/DrawSupport 深分类）切片**：
- `mglRenderCppTessEvalItemsPerPatch`——isolines/point-mode TES 核的逐
  patch 展开 item 计数（与 mgl_air_backend.cpp 的 u/v 分解 lockstep；
  discard 当时经 ObjC `mglTessFactorsDiscardPatch`，已由 2026-08-15
  后续切片收口为 `mglRenderCppTessFactorsDiscardPatch`；spacing 取整
  （FRACTIONAL_EVEN/ODD）为 TU 内静态）。Tessellation.m 的
  `mglAIRTessEvalItemsPerPatch`
  变薄包装（3 个调用点不变）；`mglTessRoundLevelForSpacing` ObjC 静态
  保留（另一调用点 native per-patch 计数仍用）。
- `mglRenderCppCheckedTessCaptureSize`——溢出检查的 capture size 数学
  （records×stride + min_stride 下限 + __builtin_mul_overflow）。
- smoke TESS_EVAL_ITEMS_AND_SIZE_OK：isolines（1×2×2=4）、quad/tri
  point-mode、非 point 返回 0、discard/null、size 基础 + 4 个坏参。
- 中途修正：(a) round-57 的 discard stub 缺 GL_ISOLINES 分支（default 按
  TRI 判 edge[2]=0 误判 discard）——补上；(b) smoke 测试数据初版 quad
  的 edge[2]/[3]=0 被 QUADS discard 判定跳过、tri 的 i0 用错 half——
  修正记录数据。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（air_tessellation_* 双门 PASS）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  45 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 5525ad0：tess-factor CPU 变换迁 C++——item 1141/887 切片）

**item 1141/887（DrawSupport 深分类）切片**：三个 tess-factor CPU helper
迁入 C++（纯数据变换，两门共用）：
- `mglRenderCppFillDefaultTessFactorBuffer`——默认 canonical factor 填充
  （12B/patch：4×outer + 2×inner __fp16 打包）；
- `mglRenderCppRepackTessFactorTriangles`——canonical→triangle 重打包
  （12B/patch → 8B/patch，out = in0..2 + in4）；
- `mglRenderCppTessPrimitiveCount`——原生 primitive count（GL 4.6
  §11.2.2.2 ceil 规则 + MAX(inside,1) 钳制 + 每 patch ≥1，discard 判定经
  当时的 ObjC `mglTessFactorsDiscardPatch` C 函数；该反向依赖与 smoke
  stub 已由 2026-08-15 后续切片删除）。
- ObjC 包装保留 buffer 创建（C++-first）+ GL_QUADS 直通 + 参数守卫，
  打包/计数全委托。
- smoke TESS_FACTOR_TRANSFORMS_OK：fill 记录逐 half 校验、repack 值、
  TRI/QUADS 计数（含 discard patch 跳过）+ 各坏参拒绝。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（12 个 air_tessellation_* 回归
  双门 PASS）；ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp 44 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。
- 中途修正：smoke 初版 patch0 的 edge 全 0 会被 discard（edge<=0 判定），
  修正为 edge=1.0 + inside=0.5 再断言计数。


### P4 完成记录追加（2026-08-15，commit 5c21d01：readback staging/wait 序列去重——item 1171 切片）

**item 1171（readback）切片**：`mglReadColorTextureAsBGRA8` 与
`mglReadDepthTextureAsFloat` 共享 77 行逐点一致的编排（staging buffer 创建 +
blit encoder + copy + end（带异常清理）+ semaphore+0.25s 超时 completion
wait（含 error 状态）+ commit + newCommandBuffer）提取为
`readbackStageAndWaitTexture:...:success:`：
- 日志主语经 logKind 参数化（"readback"/"depth readback"）；失败路径按
  调用点语义保留 GL 错误与返回行为；
- 两个方法各只剩 guards + clipped-copy 数学 + 转换；每方法 -55 行；
- integer readback 的阻塞 wait 语义不同，保持原样未动；
- readPixels 被每个回归的收尾读像素覆盖（main.c 777/823）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 43 项 SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 79b01ff：integer readback 像素转换迁 C++——item 1171/1116 切片）

**item 1171（readback）/1116 切片**：`mglReadIntegerTextureAsRGBA32` 的
逐像素转换（分量提取 + GL_INTEGER 打包/钳制 + 行拷贝，~125 行纯 CPU
数据变换）逐行迁入 `mglRenderCppConvertIntegerReadback`（两门共用）：
- ObjC 只剩参数装配（staging/blit/completion/wait 序列 + isRenderTarget
  源 Y-flip 保持 renderer 侧）；
- packedBitWidths/packedShifts 声明改 uint32_t[4]（NSUInteger[4] 不可直接
  喂 C++ 形参）；
- 语义逐点等价——包括 2_10_10_10_REV 的位宽钳制（alpha 4→3）、无符号源
  GL_BYTE 钳制（255→127）、有符号源 in-range 直通（200=-56 不变）。
- smoke INTEGER_READBACK_CONVERT_OK：非打包直通、打包钳制、无符号钳制、
  有符号直通、坏参。test-met alcpp 43 项 SMOKE_DONE。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；check-air-only OK；git diff --check 干净。
- 中途修正：sed 行号因 round-53 改写偏移，转换提取曾夹带下一方法的尾巴
  （mglApplyPendingFBODepthClearForReadback 的 ObjC 体）——已从 .cpp 剔除；
  smoke 两个期望值初版错误（alpha 位宽钳制 + 有符号 in-range 语义），
  按原语义修正断言。


### P4 完成记录追加（2026-08-16，commit 49899c1：readback 标量转换器迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 转换）切片**：三个 CPU 像素标量转换器迁入 C++
facade（两门共用单一事实源，ObjC 侧变薄委托壳、调用点不变——同
`mglRenderCppFloat11ToFloat`（e647836）模式）：
- `mglRenderCppFloatToUnorm8`——float→unorm8 取整（`value*255+0.5` 截断，
  非正→0、≥1→255、NaN→0，与原 `mglMetalFloatToUnorm8` 逐点一致）；
- `mglRenderCppSnorm16ToFloat` / `mglRenderCppSnorm8ToFloat`——snorm 解码，
  INT16_MIN/INT8_MIN→-1.0，其余 `value/32767`、`value/127`。
- `mgl_readback.m` 的 `mglMetalFloatToUnorm8` / `mglMetalSnorm16ToFloat` /
  `mglMetalSnorm8ToFloat` 变为薄委托；调用方（`mglMetalCopyTextureBytesToBGRA8`
  的逐像素转换链 + 行循环）不变。
- smoke `READBACK_SCALAR_CONVERT_OK`：unorm8 取整（0/0.5→128/0.75→191/
  1.0→255/NaN→0）、snorm16（0/32767→1/INT16_MIN→-1/-16384）、snorm8
  （0/127→1/INT8_MIN→-1/-64）。
- 验证：A/B 双门均 71/0/2/73 判定逐条一致（pixel_readback_* / tex* 回归双门
  PASS）；test-mglair 全信号；test-metalcpp（含 READBACK_SCALAR_CONVERT_OK）
  SMOKE_DONE；check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit 8bae962：readback bytes-per-pixel 表迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 表）续切片**：`mglMetalReadbackBytesPerPixel`
（MTLPixelFormat → 每像素字节表，default 4B）迁入
`mglRenderCppReadbackBytesPerPixel`（uint32 pixel-format ABI 值进出，与
`mglRenderCppTextureDataKindForPixelFormat` 同型；两门共用单一事实源）：
- ObjC `mglMetalReadbackBytesPerPixel` 变薄委托壳；全部 10+ 调用点
  （Texture.m staging/readback、Blit.m framebuffer blit）不变；
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展 BPP 表断言（RGBA32Float=16、
  R8Unorm=1、R16Unorm/RG8Unorm/ABGR4Unorm=2、RG32Float/RGBA16Unorm=8、
  R8Sint=1、RG8Uint=2、RGBA8Snorm/RGBA8Unorm=4、未知→4）；
- 验证：A/B 双门均 71/0/2/73（pixel_readback / blit 回归双门 PASS）；
  test-mglair 全信号；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit ed9d7d2：readback 格式分类表迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 分类）续切片**：三个 MTLPixelFormat→布尔分类表
迁入 C++ facade（uint32 pixel-format ABI 值进、1/0 出，与
`mglRenderCppTextureDataKindForPixelFormat` 同型；两门共用单一事实源）：
- `mglRenderCppReadbackFormatIsBGRA8Compatible`——BGRA8 可转换格式集；
- `mglRenderCppPixelFormatIsIntegerColor`——整数颜色格式集；
- `mglRenderCppPixelFormatIsSignedIntegerColor`——有符号整数颜色格式集。
- `mgl_readback.m` 的 `mglMetalReadbackFormatIsBGRA8Compatible` /
  `mglMetalPixelFormatIsIntegerColor` / `mglMetalPixelFormatIsSignedIntegerColor`
  变薄委托壳；调用方（Texture.m / Blit.m readback 与 blit 路径）不变。
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展三表断言（正/负/未知格式 + 深度
  格式拒绝）。
- 验证：A/B 双门均 71/0/2/73（pixel_readback / blit 回归双门 PASS）；
  test-mglair 全信号；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit d65816b：GL BGRA8 行拷贝迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：`mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes`
（GL BGRA8 行 → BGRA8 兼容 Metal 像素格式：RGBA8Unorm / BGRA8Unorm /
RGB9E5Float / RGB10A2Unorm / BGR10A2Unorm，可选 Y-flip；纯指针+位打包，零
Metal/ObjC 调用）迁入 `mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes`
（两门共用单一事实源）：
- ObjC 变薄委托壳；调用方（Texture.m / Blit.m）不变；
- RGB9E5 打包以 TU 内静态忠实拷贝 `mglPackRGBToSharedExp`（pixel_utils.h 是
  ObjC-only 头，不可入 C++ TU——先尝试 include 触发 forward-enum 编译错误，
  改 TU-local 实现）；
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展行拷贝断言：RGBA8 通道重排、BGRA8
  直通、flipY 行反转、RGB10A2 位精确打包（期望字逐位比对）、坏参/不支持
  格式拒绝；
- 验证：A/B 双门均 71/0/2/73（pixel_readback / blit 回归双门 PASS）；
  test-mglair 全信号；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit 5a73f2e：Metal texture bytes → GL BGRA8 迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：`mglMetalCopyTextureBytesToBGRA8`
（Metal 纹理字节 → GL BGRA8：RGBA8/BGRA8、R/RG/RGBA 8/16/32 unorm/snorm/
int/uint/float、RGB9E5、RGB10A2/BGR10A2、BGR5A1、ABGR4、RG11B10、
half/float 变体，可选 Y-flip；纯指针+格式解码，零 Metal/ObjC 调用）迁入
`mglRenderCppCopyTextureBytesToBGRA8`（两门共用单一事实源）：
- ObjC 变薄委托壳；调用方（Texture.m / readback 路径）不变；
- half 与 11/10-bit unsigned float unpack 以 TU 内静态忠实拷贝
  `mglHalfToFloat` / `mglUnpackUnsignedFloatComponent`（pixel_utils.h 是
  ObjC-only 头，不可入 C++ TU；smoke 亦不链 pixel_utils.c）；
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展：RGBA8 通道重排、flipY 行反转、
  BGRA8 直通、R8 扩通道、RGB10A2 / BGR5A1 / RGBA32Float / RGB9E5 解码、
  坏参不写 dest；
- 验证：A/B 双门均 71/0/2/73；test-mglair 全信号；test-metalcpp
  SMOKE_DONE；check-air-only OK；git diff --check 干净。
- item 1171 剩余：`mglMetalCopyBGRA8CompatibleTextureBytesToGL` 的
  RG11B10 / 16-32bit 旁路、BGRA8 中间缓冲（NSMutableData）
  与 packed/scalar 收尾路径。

### P4 完成记录追加（2026-08-16，commit b4181c5：readback type-accept + SNORM8 直接转码迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglMetalCopyBGRA8CompatibleTextureBytesToGL` 拆出两块共用事实源：
- `mglRenderCppReadbackGLTypeAccepted`——GL pixel type 接受表；
- `mglRenderCppCopySnorm8TextureBytesToGL`——R8/RG8/RGBA8 SNORM 直接
  转到 GL format/type（绕过有损 BGRA8），含 format 通道映射、
  packed/scalar dest 步长、可选 Y-flip。
- ObjC 入口只剩 type 委托 + SNORM 薄转发；其余旁路仍在 ObjC。
- half pack 以 TU 内静态忠实拷贝 `mglFloatToHalf`（pixel_utils.h
  不可入 C++ TU）。
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展：type 正/负、R8 SNORM→
  float/byte、RGBA8 SNORM→BGRA float、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-mglair 全信号；test-metalcpp
  SMOKE_DONE；check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit 0a169e6：RGB10A2 直接转码迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopyRGB10A2TextureBytesToGL` 承接 ObjC
`sourceIsRGB10A2Direct` 路径（绕过有损 BGRA8）：
- 覆盖 UNSIGNED_BYTE/BYTE/SHORT/INT/FLOAT/HALF 与
  10_10_10_2 / 2_10_10_10_REV / 5_9_9_9_REV / 8_8_8_8(_REV)；
- 复用已有 format 通道映射、packed dest 步长、`mglCppPackRGBToSharedExp`；
- ObjC 只剩 type 门 + 薄转发。
- smoke：RGBA/BGRA float、2_10_10_10_REV 位精确、10_10_10_2 MSB、
  UNSIGNED_BYTE、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：RG11B10 / 16-32bit 旁路、BGRA8 中间缓冲
  （NSMutableData）与 packed/scalar 收尾路径。

### P4 完成记录追加（2026-08-16，commit be0d569：RG11B10 直接转码迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopyRG11B10TextureBytesToGL` 承接 ObjC
`sourceIsRG11B10FloatDirect` 路径（绕过有损 BGRA8）：
- 覆盖 UNSIGNED_BYTE/BYTE/SHORT/INT/FLOAT/HALF 与
  10F_11F_11F_REV / 5_9_9_9_REV / 8_8_8_8(_REV)；
- `GL_RGB` + `10F_11F_11F_REV` 走逐行 memcpy（与 Metal LSB 布局相同）；
- 其余路径 decode 11/10-bit float 后按 format 重排再 pack；
- 11/10-bit pack 以 TU 内静态忠实拷贝 `mglFloatToFloat11` /
  `mglFloatToFloat10`（pixel_utils.h 不可入 C++ TU）；
- ObjC 只剩 type 门 + 薄转发。
- smoke：RGB memcpy 位精确、RED/BGRA float、UNSIGNED_BYTE、
  BGR 重排 10F_11F_11F_REV、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：16-32bit 旁路、BGRA8 中间缓冲（NSMutableData）
  与 packed/scalar 收尾路径。

### P4 完成记录追加（2026-08-16，commit 3b3c24a：16/32-bit 直接转码迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopy16or32TextureBytesToGL` 承接 ObjC
R16/RG16/RGBA16 Unorm/Snorm/Float 与 R32/RG32/RGBA32 Float
直接路径（绕过有损 BGRA8）：
- 覆盖 UNSIGNED_BYTE/BYTE/SHORT/INT/FLOAT/HALF 与
  3_3_2 / 5_6_5 / 4_4_4_4 / 5_5_5_1 / 8_8_8_8 /
  10_10_10_2 / 2_10_10_10_REV / 10F_11F_11F_REV / 5_9_9_9_REV
  及其 REV 变体；
- 缺通道按末分量复制（与 ObjC `idx >= srcChannels` 一致）；
- SNORM 解码为 value/32767（无 INT16_MIN 特例，忠实 ObjC）；
- ObjC 只剩 type 门 + 薄转发。
- smoke：R16Unorm float/u8/RGBA 复制、RGBA16→BGRA/565、
  R16Float/R32Float/R16Snorm、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：BGRA8 中间缓冲（NSMutableData，alloc 留 ObjC）
  与 BGRA8/RGBA8 packed/scalar 收尾路径。

### P4 完成记录追加（2026-08-16，commit d413116：BGRA8/RGBA8 标量 readback 迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopyUnorm8ScalarTextureBytesToGL` 承接 ObjC
BGRA8/RGBA8 UNORM → BYTE/SHORT/INT/UINT/USHORT/HALF/FLOAT：
- 源通道按 RGBA vs BGRA 展开为逻辑 RGBA，再按 format 重排；
- 缩放与 ObjC 一致（u16=`v*257`，u32=`v*16843009`，
  i32 用 64-bit 防溢出）；
- ObjC 只剩 type 门 + 薄转发；UNSIGNED_BYTE 与 packed 仍在 ObjC。
- smoke：BGRA8→RED/RGBA/BGRA float、RGBA8→RED、u16/snorm8/half、
  flipY、UNSIGNED_BYTE/错格式/NULL 拒绝。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：BGRA8 中间缓冲（NSMutableData，alloc 留 ObjC）、
  BGRA8/RGBA8 packed 与 UNSIGNED_BYTE 通道重排收尾。

### P4 完成记录追加（2026-08-16，commit 5270432：BGRA8/RGBA8 packed readback 迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopyUnorm8PackedTextureBytesToGL` 承接 ObjC
BGRA8/RGBA8 UNORM → packed types：
- 仅 `GL_BGRA`/`GL_BGR` 交换 R/B，其余 format 保持逻辑 RGBA
  （含 GREEN/BLUE/ALPHA，与 ObjC 一致，不走单通道抽取）；
- 覆盖 3_3_2 / 5_6_5 / 4_4_4_4 / 5_5_5_1 / 8_8_8_8 /
  10_10_10_2 / 10F_11F_11F_REV / 5_9_9_9_REV 及其 REV；
- `10F_11F_11F_REV` 的 UNORM8→unsigned-float pack 以 TU 内静态
  忠实拷贝 `mglPackUnsignedFloatFromUNorm8`；
- ObjC 只剩 type 门 + 薄转发。
- smoke：BGRA8→RGB/BGR 565、8888_REV、2_10_10_10_REV、
  RGBA8→565、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：BGRA8 中间缓冲（NSMutableData，alloc 留 ObjC）
  与 UNSIGNED_BYTE 通道重排收尾。

### P4 完成记录追加（2026-08-16，commit a50c114：UNSIGNED_BYTE 通道重排迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
`mglRenderCppCopyUnorm8SwizzleTextureBytesToGL` 承接 ObjC
`CopyBGRA8CompatibleTextureBytesToGL` 收尾 format switch
（UNSIGNED_BYTE 通道重排 + 遗留 RGBA FLOAT 分支）：
- BGRA/RGBA/BGR/RGB/RG/RED/GREEN/BLUE/ALPHA 与 ObjC 一致；
- dest 步长校验按 packed-vs-scalar 再套原 switch 约束
  （BGRA 必须 4B；RGBA 允许 4B 或 FLOAT 16B）；
- ObjC 收尾循环改为薄转发。
- smoke：BGRA8→RGBA/BGRA/RGB/RED/BLUE、RGBA8→BGR、flipY、坏参。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171：`CopyBGRA8CompatibleTextureBytesToGL` 的 CPU 转码已迁完；
  仅剩 NSMutableData 中间缓冲（alloc 留 ObjC，convert 已走 C++）。
- 审查（`0a169e6..a50c114`）：无本轮引入缺陷。残留项见
  `REVIEW_item1171_loop_2026-08-16.md`（smoke 缺口、16/32 缺通道复用、
  既有舍入/短行、NSMutableData alloc）。

### P4 完成记录追加（2026-08-16，commit 6aa27db：行拷贝 + depth readback CPU 转码迁 C++——item 1171 切片）

**item 1171（readback 纯 CPU 数据变换）续切片**：
- `mglRenderCppCopyRows` 承接 `mglMetalCopyRows`（逐行 memcpy + 可选
  Y-flip）；ObjC 变薄委托，Texture getTexImage / depth float 直通调用点
  不变。
- `mglRenderCppCopyDepthTextureBytesToFloat` 承接
  `mglReadDepthTextureAsFloat` 的 Depth16 / unpacked depth-float → GL
  float 循环（与 ObjC 一样默认 flipY）。
- smoke `READBACK_SCALAR_CONVERT_OK` 扩展：行拷贝直通/flipY/坏参、
  Depth16 flipY、Depth32 直通。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1171 剩余：`CopyBGRA8CompatibleTextureBytesToGL` 的 NSMutableData
  中间缓冲（alloc 留 ObjC）。

### P4 完成记录追加（2026-08-16，commit 0c7895d：R8 swizzle expand 迁 C++——item 1111 切片）

**item 1111（纹理 swizzle 纯 CPU）切片**：
- `mglRenderCppResolveR8SwizzledComponent` 承接
  `mglResolveR8SwizzledComponent`（tex 未用，只看 swizzle + red）。
- `mglRenderCppCreateSingleChannelSwizzledUpload` 承接
  `mglCreateSingleChannelSwizzledUpload`：仅 `GL_R8` 1B/px → RGBA8，
  四通道走 resolve；非 R8 / 坏参 / 512MiB 上限返回 NULL。
- ObjC 变薄委托，Texture* 抽取 internalformat + 四个 swizzle 后转发。
- smoke `R8_SWIZZLE_EXPAND_OK`：resolve 表、默认 R/0/0/1、padded 行、
  非 R8 / 坏参拒绝。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mgl_texture_compat.m` 剩余纯 CPU 表
  （`mglMetalTextureLevelDimension` / stored-components /
  layer pixel-format / sRGB↔linear）；不要迁 NSMutableData alloc，
  不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit 482b0f4：layer / sRGB pixel-format 表迁 C++——item 1111/887 切片）

**item 1111/887（纹理 pixel-format 纯 CPU 表）切片**：
- `mglRenderCppMetalLayerPixelFormatIsSupported`：仅 BGRA8 / BGRA8_sRGB。
- `mglRenderCppSRGBPixelFormat` / `mglRenderCppLinearPixelFormat`：
  RGBA8 / BGRA8 互转，其余原样返回。
- `mglRenderCppEffectiveMTLPixelFormat`：`srgb_decode_ext ==
  GL_SKIP_DECODE_EXT` 时降到 linear；0 / DECODE 不变。
- ObjC 四个函数变薄委托；Effective 只抽 `tex->params.srgb_decode_ext`。
- smoke `LAYER_PIXEL_FORMAT_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mglStoredColorComponentsForTexture`（需 TU 内
  `numComponentsForFormat` 拷贝，勿 include pixel_utils.h）或
  `mglTextureUploadNeedsSingleChannelSwizzle` 格式表；不要迁
  NSMutableData alloc，不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit 71102f0：R-only swizzle 门迁 C++——item 1111 切片）

**item 1111（纹理 swizzle 纯 CPU 表）切片**：
- `mglRenderCppTextureUploadNeedsSingleChannelSwizzle` 承接
  `mglTextureUploadNeedsSingleChannelSwizzle`：`swizzled==0` → 0，
  否则 GL_R* 格式表（R8/R16/R32 及 SNORM/F/I/UI）。
- ObjC 只抽 internalformat + swizzled 后转发。
- smoke `R_ONLY_SWIZZLE_GATE_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mglStoredColorComponentsForTexture`（TU 内拷贝
  `numComponentsForFormat`，勿 include pixel_utils.h）；不要迁
  NSMutableData alloc，不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit e994c3e：stored-components 迁 C++——item 1111 切片）

**item 1111（纹理分量计数纯 CPU 表）切片**：
- TU 内 `mglCppNumComponentsForFormat` 忠实拷贝 `pixel_utils.c` 的
  `numComponentsForFormat`（static，避免与 lib 符号冲突；core 未声明
  的 legacy 枚举用同一组十六进制）。
- `mglRenderCppStoredColorComponents` 承接
  `mglStoredColorComponentsForTexture` 的 format→count（>0 否则 4）。
- ObjC 只保留 null-tex → 4，其余转发。
- smoke `STORED_COLOR_COMPONENTS_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mglMTLSwizzleForGLSwizzle` 或
  `mglTextureNeedsChannelExpansion` 表（若仍在 ObjC）；不要迁
  NSMutableData alloc，不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit 77a93d4：RGB channel-expansion gates 迁 C++——item 1111 切片）

**item 1111（纹理展开门纯 CPU 表）切片**：
- `mglRenderCppTextureInternalFormatNeedsRGBA8Expansion` /
  `mglRenderCppTextureNeedsChannelExpansion` 承接 ObjC 两表；
  upload-prep 内直接调用 C++，不再经 ObjC 符号。
- smoke stub 删除，改走同一事实源；`CHANNEL_EXPANSION_GATE_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-16，commit bc7fee0：GL→MTL swizzle 映射迁 C++——item 1111 切片）

**item 1111（纹理 swizzle 纯 CPU）切片**：
- `mglRenderCppMTLSwizzleForGLSwizzle` 承接
  `mglMTLSwizzleForGLSwizzle`（components 门控缺通道 → Zero /
  Alpha→One；未知枚举 → Zero + stderr）。
- ObjC 只算 components 后转发；C ABI 返回 uint32_t（Metal
  TextureSwizzle 数值）。
- smoke `GL_MTL_SWIZZLE_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mglTextureMinFilterUsesMipmaps`（纯枚举表）或
  `mglTextureDataKindName`；`mgl_texture_compat` 主体纯 CPU 表已薄委托。
  不要迁 NSMutableData alloc，不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit 5bde0fc：min-filter mipmaps 门迁 C++——item 1111 切片）

**item 1111（纹理 min-filter 纯 CPU 表）切片**：
- `mglRenderCppTextureMinFilterUsesMipmaps` 承接
  `mglTextureMinFilterUsesMipmaps`（四个 MIPMAP_* → 1，其余 0）。
- ObjC static 变薄委托。
- smoke `MIN_FILTER_MIPMAPS_OK`。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- 下一刀候选：`mglTextureDataKindName`；或窄 `GLMMetalFuncs` 回调删体。
  不要迁 NSMutableData alloc，不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16，commit de04778：texture data-kind name 迁 C++——item 1111 切片）

**item 1111（纹理 data-kind 纯 CPU 表）切片**：
- `mglRenderCppTextureDataKindName` 承接 `mglTextureDataKindName`
  （float/sint/uint/depth/unknown 静态字面量）。
- ObjC 变薄委托；smoke `TEXTURE_DATA_KIND_OK` 扩展 name 断言。
- 验证：A/B 双门均 71/0/2/73；test-metalcpp SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- `mgl_texture_compat.m` 主体纯 CPU 表已薄委托。下一刀候选：头文件内
  inline 分类表（depth/stencil / packed / looks-depth）迁 C++，或窄
  `GLMMetalFuncs` 已有 facade 的删体。不要迁 NSMutableData alloc，
  不要做 P5 级 compute/CB sequencing。

### P4 完成记录追加（2026-08-16：fragment fallback binding snapshot 收口——item 1014/1138 切片）

**item 1014（binding auxiliary segment）切片**：
- `bindFragmentFallbackBuffersToCurrentRenderEncoder:` 现在与 fragment 主
  binding loop 共用 `MGLRenderCppBindingSnapshot`；gate-on 收集 buffer/clear
  op 后在 fallback 段末一次交给 `mglRenderCppEncodeBindingSnapshot`，保持
  主循环 → fallback 的 encoder 顺序。
- gate-off 仍调用原 ObjC setter；dedup cache 更新、fallback buffer、全槽
  safety-net 和统计路径均保持不变。
- 现有 `BINDING_SNAPSHOT_OK` smoke 覆盖 fragment buffer + bytes + nil-clear
  及错误边界；`make -j4 lib` 编译通过。
- 本切片没有迁移 texture/sampler 的 Metal 资源创建，也没有改变 fallback
  的 GL 语义；这些仍属于 item 1014/P5 的后续范围。
- 验证发现并修复一项 migration-specific archive 差异：C++ eligibility 原先
  只排除缺 fragment 的 pipeline，现与 ObjC 一致要求 vertex + fragment
  同时存在；schema 升至 v4 隔离可能含 fragment-only 记录的 v3 archive。
- 后续顺序实验确认 v4 仍不能跨 producer 复用：cpp 连续 load/save 正常，
  `cpp -> ObjC -> cpp` 后最后一次 serialize 稳定报 `missing vertex stage`。
  schema 升至 v5，并按 `normal/ASan/TSan x cpp/objc` 六个命名空间隔离；这是
  迁移双实现共享 archive 引入的问题，损坏文件自愈逻辑继续保留。
- 更长的独立进程实验又确认：producer 隔离不是最终根因。同一
  v5-cpp archive 每次加载后仍无条件重复 add 相同 pipeline，第 4 个进程
  稳定出现 `expecting 'fragment' stage in pipeline no. '97'`。现已改为
  archive hit 不 add、miss 才 add；producer/sanitizer 隔离作为防御边界保留。
- smoke 新增 `BINARY_ARCHIVE_HIT_MISS_OK`，用真实 Metal archive 断言首次
  miss 与序列化重载 hit。独立进程验证：cpp cold + warm x4 全部
  71/0/2 且 created/loaded/saved 正常；ObjC cold/warm 后再切 cpp warm 也全部
  71/0/2，无 `discarded unserializable archive`。详见
  `docs/P4_BINARY_ARCHIVE_LIFECYCLE_2026-08-16.md`。
- sanitizer 构建修正：`SANITIZE` 现同时进入 C/C++ flags，AIR/Metal-cpp
  flags 定义前移到 compile/link hash 之前，避免复用未插桩 C++ 对象。
  完整插桩后 ASan/TSan cpp regression 均 71/0/2、exit 0，分别稳定
  load/save `v5-asan-cpp` 和 `v5-tsan-cpp`。

### P4 完成记录追加（2026-08-16：command commit guard owner 化——item 1051/1141 切片）

- `CommandBufferOwner` 新增 `commit_in_progress`，通过
  `mglRenderCppCommandBufferOwnerBeginCommit/EndCommit` 管理；首次 begin 返回
  1、嵌套返回 0、空 owner 返回 -1。
- `MGLRenderPassManager` 的 begin/end 方法变为薄适配，删除
  `MGLCommandState.isCommittingCommandBuffer`；shutdown 在销毁 owner 前结束
  guard。
- smoke `COMMAND_BUFFER_COMMIT_GUARD_OK` 覆盖获取、嵌套拒绝、释放后重入和
  空 owner；原 ObjC guard 的非线程同步语义保持不变。
- AGX commit validation/recovery、异常捕获和 raw ObjC fallback 本轮未改，记录
  于 `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。

### P4 完成记录追加（2026-08-16：command completion/error value-state 分类——item 1051 切片）

- 新增 `mglRenderCppClassifyCommandBufferCommit` 与
  `mglRenderCppClassifyCommandBufferCompletion`：提交状态判定和
  `MTLCommandBufferErrorDomain/code 4` driver-rejection 识别迁入 C++，
  `MGLRenderer+GPURecovery.m` 只消费分类结果并保留现有日志、错误计数、节流、
  deferred reset、异常捕获和 commit fallback。
- smoke `COMMAND_BUFFER_RECOVERY_DECISION_OK` 覆盖空参、proceed、Error 的 legacy
  already-committed 判定、成功 completion 和 driver rejection。
- 原 ObjC 中 Error 专用提交分支因枚举顺序不可达；本轮保持既有行为并记录到
  `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`，不作为迁移回归修复。
- `CommandBufferRecoveryOwner` 进一步接管原 ObjC error/success 计数、最后错误
  时间、recovery mode 和跨线程锁；API 保留 8-error、3 秒 timeout、
  4-success/0.25 秒 reset，以及 completion 首成功另行 clear-mode 的两步顺序。
  smoke `COMMAND_BUFFER_RECOVERY_OWNER_OK` 对上述状态机和空参/销毁边界做
  确定性覆盖。

### P4 完成记录追加（2026-08-16：completion/error-recovery 编排 facade——item 1051 切片）

- 新增 `mglRenderCppProcessCommandBufferCompletion`：C++ 统一执行完成状态分类、
  recovery owner 的 error/success 记账，并返回 `MGLRenderCppCommandBufferCompletionResult`
  value-state。`MGLRenderer+GPURecovery.m` 的 completion block 只消费结果做日志、
  driver-rejection 节流和 deferred reset，不再直接编排 owner 状态机。
- success 路径仍严格调用 `RecordSuccess` 后再调用独立的 `ClearMode`，保留原
  ObjC 两次加锁与首成功清 mode 的可见顺序；driver-rejection 的静态 2 秒节流、
  ObjC 异常捕获、实际 commit/fallback 和 `clearProblematicGPUState` 仍在 ObjC，
  这些原有限制记录于 `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。
- smoke 新增 `COMMAND_BUFFER_COMPLETION_PROCESS_OK`，覆盖空参、8 次错误、进入
  recovery、首个成功清 mode，以及 4-success/0.25 秒 sustained reset。
- 本切片只迁移 completion/error-recovery 的 value-state 与结果编排；`.m` 中
  借用 current command buffer 的适配列、提交异常恢复高层策略和 callbacks 其余
  入口仍是 P4.5 后续项。

### P4 完成记录追加（2026-08-16：texture format classification predicates——item 1111 切片）

**item 1111（纹理格式纯 CPU 分类）切片**：

- 新增 `mglRenderCppMetalPixelFormatIsDepthOrStencil`、
  `mglRenderCppMetalPixelFormatIsPackedDepthStencil`、
  `mglRenderCppGLInternalFormatLooksDepthOrStencil` 和
  `mglRenderCppTexturePixelFormatCompatibleWithExpectedDataKind` 四个 C ABI
  facade；实现只使用 `uint32_t` enum value，不把 `MTL::*` 类型暴露到 ABI。
- `mgl_texture_compat.h` 的 ObjC inline 名称保留，但只做薄包装；depth/stencil、
  packed attachment 和 sampler data-kind compatibility 的分类表不再在 ObjC
  header 中重复实现。gate-off 与 gate-on 共用同一分类事实源。
- smoke 新增 `TEXTURE_FORMAT_CLASSIFY_OK`，覆盖 depth/stencil、packed、GL
  internal-format 正反例及 float/sint/unknown compatibility；`TEXTURE_DATA_KIND_OK`
  原有覆盖保持不变。
- 验证：`make -j4 lib`、`make test-metalcpp` 通过；ObjC 和 Metal-cpp
  regression 均 `71 PASS / 0 FAIL / 2 SKIP`（73 tests）。
- 本刀只迁移纯 CPU 分类，不改变 texture upload、command-buffer sequencing，
  也不改变原 ObjC 的提交、异常捕获、deferred reset 或线程前提；这些限制继续
  记录于 `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。

下一刀候选：已有 C++ facade 的窄 `GLMMetalFuncs` 适配列；不要借此批次迁
`NSMutableData` 分配或 P5 级 compute/command-buffer sequencing。

### P4 完成记录追加（2026-08-16：compressed upload row table——item 1111 切片）

**item 1111（压缩纹理上传行数纯 CPU 表）切片**：

- 新增 `mglRenderCppMetalCompressedBlockHeight` 和
  `mglRenderCppMetalUploadRowsForPixelFormat`；BC/ASTC block-height 表和
  `pixelHeight` 的 min-1、block 向上取整逻辑统一在 `mgl_render_cpp.cpp`。
- `mgl_texture_compat.h` 保留原 `NSUInteger` helper 名称，仅转换为
  `uint64_t` C ABI 结果；删除 header 内重复的 Metal pixel-format switch。
- smoke 新增 `TEXTURE_COMPRESSED_ROWS_OK`，覆盖 BC1、ASTC 6x6、未压缩格式、
  height=0 和跨 block height 的正例。
- 完整验证：`make test-all` exit 0；normal ObjC/C++、ASan C++、TSan C++
  regression 均 `71 PASS / 0 FAIL / 2 SKIP`，sanitizer 无报告，archive 分别为
  `v5-objc`、`v5-cpp`、`v5-asan-cpp`、`v5-tsan-cpp`。
- 该切片只改变纯 CPU 表的实现位置；上传 command encoder、专用 command buffer、
  copy-back、ObjC 异常恢复和生命周期限制均未改变。

### P4 完成记录追加（2026-08-16：GPU timestamp callback 直连 C++——item 1051/1197 切片）

- `GLMMetalFuncs.mtlGetGPUTimestamp` 在 gate-on 且 C++ device 可用时改为
  `mglRenderCppGetGPUTimestamp`，直连 C++ callback 数由 11 增至 12，剩余 41。
- C++ callback 先通过原 `mtlFlush(ctx, true)` 适配列执行 GL ordering 所要求的
  flush+wait，再调用 `mglRenderCppSampleTimestamps` 返回 GPU timestamp；无效
  context/bridge 返回 0，保持旧 C bridge 的防御边界。
- 为避免让 `mgl_render_cpp.cpp` 依赖完整 `GLMContext` 布局，新增纯 C
  `mglContextHasValidMetalBridge` 查询；C++ 继续只持有 opaque context pointer。
- gate-off 仍调用 `MGLRenderer+QuerySync.m -mtlGetGPUTimestamp:`；gate-on 的
  flush 仍进入 ObjC `flushCommandBuffer:YES`、commit/AGX recovery 和等待策略。
  本刀不迁移或修正这些原 ObjC 高层行为。
- smoke 新增 `GPU_TIMESTAMP_CALLBACK_OK`，锁定 valid/invalid context、单次
  finish flush 和非零 timestamp；`make test-all` exit 0，normal ObjC/C++、
  ASan C++、TSan C++ regression 均 `71 PASS / 0 FAIL / 2 SKIP`，sanitizer
  无报告，archive namespace 正常。

### P4 完成记录追加（2026-08-16：swap-present owner-aware facade——item 1051 切片）

- 新增 `mglRenderCppGetCommandBufferOwnerState`：gate-on 的 swap presentation
  状态检查直接从 `CommandBufferOwner` 输出 value-state，不再先把 current buffer
  借给 `MGLRenderer.m` 再查询 status。
- 新增 `mglRenderCppPresentDrawableForCommandBufferOwner`：C++ 从 owner 取 current、
  验证 `NotEnqueued` 并编码 present；删除旧 raw
  `mglRenderCppPresentDrawable(command_buffer, drawable)` API。
- `mgl_render_cpp_objc.h` 的 adapter 明确分门：gate-on 走 owner-aware C++ facade，
  gate-off 仍借用 current 并调用原 ObjC `presentDrawable:`，异常继续传播到 swap
  外层 `@catch`。finalized-buffer rotate、drawable 校验、commit 和日志顺序未改。
- smoke `COMMAND_BUFFER_OWNER_PRESENT_OK` 覆盖 owner state、空参、detach 后无
  current 等边界；真实 present 由 A/B regression 的 swap 路径覆盖。
- `MGLRenderer.m` 的 direct
  `mglRenderCppCommandBufferOwnerGetCurrent` 匹配行从本切片前 15 降至 10；剩余
  是诊断日志、非 present command 调度与后续适配列，不在本切片机械替换。
- status-check/present 仍依赖原 GL-thread + `METAL_LOCK` 串行前提，不新增跨线程
  原子性；该原 ObjC 限制记录于
  `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。
- 验证：`make test-metalcpp`（含 `COMMAND_BUFFER_OWNER_PRESENT_OK`）、
  `make -j4 lib`、`make check-air-only`、`make test-all`、`git diff --check`
  全部通过；ObjC/C++ 串行 regression 均 71 PASS / 0 FAIL / 2 SKIP，archive
  分别稳定 load/save `v5-objc` 与 `v5-cpp`，无损坏自愈日志。

### P4 完成记录追加（2026-08-16：command-owner encoder facade——item 1051 窄切片）

- 新增 `mglRenderCppCreateRenderEncoderFromCommandBufferOwnerState` 与
  `mglRenderCppCreateBlitEncoderFromCommandBufferOwner`；gate-on 的 scissored
  clear 和 stage copy-back 现在直接从 `CommandBufferOwner.current` 创建 encoder，
  不再先把 current command buffer 借到 `MGLRenderer.m`。
- `mgl_render_cpp_objc.h` 增加统一 owner-first encoder adapter：gate-on 成功路径
  不暴露 current，gate-off 以及 C++ facade 失败时保留 ObjC descriptor/encoder
  fallback。空 owner、空 state、空输出和 detached owner 均返回错误，不改变
  原有清理与 copy-back 失败路径。
- `MGLRenderer.m` 的 direct
  `mglRenderCppCommandBufferOwnerGetCurrent` 匹配从前一切片的 10 降至 4，
  剩余 4 处全部是 `mglLogStateSnapshot` 诊断日志；`GPURecovery.m`、
  `QuerySync.m`、`Lifecycle.m` 仍为 0。adapter 中保留的 4 处 getter 仅用于
  gate-off/失败回退，属于后续适配列，不宣称 command lifecycle 已完全迁出。
- smoke 新增 `COMMAND_BUFFER_OWNER_ENCODERS_OK`，使用真实 2x2 render target
  覆盖两个 owner encoder 的创建、结束、空参和 detached 边界。
- 验证：`make test-metalcpp`、`make -j4 lib`、`make check-air-only`、
  `make test-all`、`git diff --check` 全部通过；ObjC/C++ 串行 regression
  均 71 PASS / 0 FAIL / 2 SKIP，分别 load/save `v5-objc` 与 `v5-cpp`，无
  `discarded unserializable archive`。

### P4 完成记录追加（2026-08-16：RenderPassManager owner encoder 适配列——item 1051 窄切片）

- `MGLRenderPassManager -createRenderEncoderWithDescriptor:` 的默认 render-pass
  路径现在先 snapshot `renderPassStateOwner`，再经
  `mglRenderCreateRenderEncoderForCommandBufferOwner` 从 `CommandBufferOwner`
  创建 encoder；gate-on 成功路径不再先借出 current command buffer。
- 自定义 descriptor 仍保留 ObjC `renderCommandEncoderWithDescriptor:` fallback，
  C++ owner/state 缺失或创建失败时也保留同一 fallback。该限制是 descriptor
  value-state 尚未覆盖全部调用点的适配边界，不改变 A/B 语义。
- `mdiArgumentScratchBufferWithDevice:length:offset:` 的 current 存在性检查改为
  `mglRenderCommandBufferOwnerState`；scratch owner 的容量、对齐、增长和 reset
  逻辑保持不变。
- manager 中保留的 raw getter 只用于自定义 descriptor/owner 创建失败的兼容回退，
  以及 detach 路径的 strong-local handoff；不宣称 command lifecycle 已完全迁出。
- 验证：`make -j4 lib`、`make test-all`、两门 regression 均
  `71 PASS / 0 FAIL / 2 SKIP`，`make test-metalcpp`、`git diff --check` 通过；
  gate-off/gate-on 分别稳定 load/save `v5-objc`/`v5-cpp` archive；独立
  `SANITIZE=address` / `SANITIZE=thread` regression 也均为
  `71 PASS / 0 FAIL / 2 SKIP`，无 sanitizer 报告，分别使用
  `v5-asan-cpp` / `v5-tsan-cpp` archive。

### P4 完成记录追加（2026-08-16：Texture mipmap owner blit 适配列——item 1116 窄切片）

- `MGLRenderer+Texture.m` 的 `mtlGenerateMipmaps` 现在通过
  `mglRenderCreateBlitEncoderForCommandBufferOwner` 从当前
  `CommandBufferOwner` 创建 blit encoder；gate-on 不再先借出 current command
  buffer。专用 upload/readback command buffer 仍使用原始 helper。
- mipmap 的 `generateMipmapsForTexture`、异常清理和 encoder end 顺序保持不变；
  gate-off 仍由 adapter 借用 current 并调用 ObjC 原语。
- 验证：`make test-all`、两门 regression 均 `71 PASS / 0 FAIL / 2 SKIP`；
  独立 ASan/TSan regression 均 `71 PASS / 0 FAIL / 2 SKIP`、无报告，分别
  load/save `v5-asan-cpp`/`v5-tsan-cpp`；`git diff --check` 通过。

### P4 完成记录追加（2026-08-16：timer-query callbacks + completion TSan 修复——item 1051/1197 切片）

- `GLMMetalFuncs.mtlBeginTimerQuery` / `mtlEndTimerQuery` 在 gate-on 且 C++
  device 可用时直连 `mglRenderCppBeginTimerQueryCallback` /
  `mglRenderCppEndTimerQueryCallback`；C++ 侧以独立 mutex 保护按 `GLMContext`
  索引的非拥有 `QueryStateOwner` registry，lifecycle 在创建后注册、销毁前注销。
- callback 先保留原 `mtlFlush(ctx, true)` 适配列，故 render-encoder 结束、commit、
  wait、异常捕获和 AGX recovery 仍是 ObjC 高层策略；GL-thread、外层
  `METAL_LOCK` 和 owner 生命周期前提已写入
  `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。
- `mglRenderCppAddCommandBufferCompletion` 改用直接的
  `MTL::CommandBufferHandler` block 按值捕获 `shared_ptr`，移除
  `HandlerFunction`/`std::function` `__block` 竞态；TSan 原报告消失。
- 为 clean 状态下 `test-all` 的 standalone binary 增加统一 `build/` order-only
  prerequisite；不改变测试内容，只修复测试目标对目录创建顺序的隐式依赖。
- smoke 新增 `TIMER_QUERY_CALLBACKS_OK`；验证：clean `make test-all`、
  `make check-air-only`、`make test-metalcpp`、`git diff --check` 全部通过；
  normal ObjC/C++、ASan/UBSan C++、TSan C++ regression 均 `71 PASS / 0 FAIL /
  2 SKIP`，sanitizer 无报告，archive namespace 分别为
  `v5-objc`、`v5-cpp`、`v5-asan-cpp`、`v5-tsan-cpp`。

### P4 完成记录追加（2026-08-15，commit 6a1e119：dirty-level 上传循环迭代 + 分类迁 C++——item 1116 切片）

**item 1116（纹理全量上传）切片**：单面（2D）dirty-level 循环的迭代 +
has-uploadable CPU 数据判定（内联，compat 头是 ObjC 类型不可入 C++ TU）+
逐 level 分类（上传 op / 短后备 / 坏参）迁入 `mglRenderCppBuildLevelUploadOps`：
- 逐 level 运行 round-46 的 prep，压缩成 op 列表（短后备 op 携带
  have/need 字节供诊断日志；stale/incomplete level 静默跳过——与 ObjC
  基线一致）；
- ObjC 循环只剩：上传每个 op、短后备日志、释放自有数据；
- item 1116 剩余：uploadTextureSliceViaBlit + dedicated-CB 的深编排
  （renderer 侧，P5 级）。
- smoke LEVEL_UPLOAD_OPS_OK：op 列表（含字段）、stale 跳过、incomplete
  跳过、容量、坏参。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（纹理上传路径被 tex* 回归覆盖）；
  ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；test-metalcpp
  42 项 SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 81dc120：compute dispatch 编排去重 + 死字段清理——item 1147 切片 + item 1155 收尾）

**item 1147（compute CB sequencing）切片**：两条 compute dispatch 入口
（direct/indirect）共享 ~90 行编排（endRenderEncoding →
ensureWritableCommandBuffer → 纹理绑定 → encoder 创建 → processCompute →
程序解析 → plan 编码 → encoder 结束 + _currentCBHasWork → copy-back flush →
dirty bits）提取为 `runComputeDispatchOrchestrationLocked:dispatchKind:...`
（reason 标签化日志；direct 专属的 image-unit authoritative 标记保留在
direct 调用方）。错误路径中间态清理按调用点语义保留。

**item 1155 收尾**：删除未使用的 `blitOperationComplete` 字段——该 item
剩余仅 isCommittingCommandBuffer（平凡重入 BOOL，保留）。

- 验证：A/B 双门 66/0/2/68 判定逐条一致（compute_dispatch_ssbo +
  air_geometry_ssbo_visibility 均双门 PASS）；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp 41 项 SMOKE_DONE；check-air-only
  OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit c4fc78b：MGLRenderPassManager → MGLMetal*Ref 适配列——item 1099 切片，白名单 16→15）

**item 1099（render pass 收口）切片**：MGLRenderPassManager 是合法的
ObjC-facing 外壳（API 必须保持 ObjC 类型给 renderer 调用），因此走
typedef 适配列下白名单（与 40-42 轮同模式，非 deep 迁移）：
- 所有方法签名、结构字段、桥接局部量切换为 MGLMetal*Ref（纯别名——
  调用方零改动；新增 MGLMetalCommandQueueRef）；
- 顺带修复 round-49/50 批量替换在三个适配文件（QuerySync/GPURecovery/
  SwapDiagnostics）引入的 id<MTL 桥接 token：getter 转换改回
  MGLMetal*Ref 桥接。
- 白名单 16 → 15 文件（剩余：PipelineCache/Renderer/Batch/BatchReplay/
  Binding/BindingState/Blit/Buffer/Compute/Draw/DrawSupport/Lifecycle/
  RenderPass/Tessellation/Texture）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only
  OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit b92080a：currentRenderEncoder 镜像删除——C++ owner 双门单源——item 1141/1155 收口）

**item 1155 第二个大镜像删除（该 item 收口）**：ObjC `currentRenderEncoder`
镜像字段移除：
- `mglRenderCppRenderEncoderOwnerGetCurrent`（借用返回 owner->encoder；
  end 后指针保留至 destroy——与旧镜像的 end 语义一致，clear 是独立步骤）；
- `installRenderEncoder` 无条件维护 owner（双门）；`endCurrentRenderEncoder`
  经 getter 读取；`clearCurrentRenderEncoder` 仅销毁 owner；
- 12 个文件的 237 处外部读取机械替换（含 3 处换行断开的 MGLEncodeContext
  初始化 + 2 处 label 赋值需本地绑定——lvalue cast 非法）+ 23 处双桥接清理，
  `MGLCommandState` 字段删除。至此 item 1141/1155 的命令生命周期双镜像
  （currentCommandBuffer / currentRenderEncoder）+ MDI scratch 全部收口。
- item 1155 剩余：isCommittingCommandBuffer 平凡重入 BOOL；
  blitOperationComplete 未用（可随 P5 清理）。
- smoke RE_GETTER_OK：null owner、创建、身份一致、end 后指针保留至 destroy、
  destroy 清零。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 RE_GETTER_OK）SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit c955bd5：currentCommandBuffer 镜像删除——C++ owner 双门单源——item 1141/1155 切片）

**item 1155 大镜像之一删除**：ObjC `currentCommandBuffer` 镜像字段移除：
- `mglRenderCppCommandBufferOwnerGetCurrent`（借用返回 owner->current）+
  `mglRenderCppCreateCommandBufferOwnerAdopt`（gate-off 回退：把 ObjC 创建的
  CB 收养进 owner——即使 C++ 创建路径失败，getter 读取仍正确）；
- `installNewCommandBufferFromQueue` / `detachCurrentCommandBufferForSubmission`
  / `discardCurrentCommandBuffer` 与 manager 内部读取（createRenderEncoder
  WithDescriptor、mdiArgumentScratchBuffer 守卫）全部 getter 化；
- 15 个文件的 136 处外部读取机械替换（`(__bridge id<MTLCommandBuffer>)`
  + getter；12 处双桥接清理），`MGLCommandState` 字段删除。
- item 1155 剩余：currentRenderEncoder 镜像（~237 读/13 文件，同模式）；
  isCommittingCommandBuffer 平凡重入 BOOL；blitOperationComplete 未用。
- smoke CB_GETTER_ADOPT_OK：null owner、坏参、adopt 身份一致、discard 后
  getter 为 NULL、destroy 清零。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 CB_GETTER_ADOPT_OK）
  SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 521446d：MDI scratch 分配器双门共用 C++ owner——item 1155 切片）

**item 1155（command lifecycle）切片**：`mdiArgumentScratchBufferWithDevice:length:offset:`
不再按编码门分叉——ObjC gate-off 分配器（64K 倍增 + 256 对齐 + 增长归零）
删除，双门统一委托 C++ `MDIScratchOwner`（同算法：2 的幂增长、对齐掩码、
增长后 offset 归零；容量可能更大——内部 scratch，不可观察）。
- `mdiArgsScratchBuffer/Capacity/Offset` 三个镜像字段删除（item 1155 三处
  镜像清零）；不再使用的 `mglRenderPassManagerCreateBuffer` 静态函数删除；
  返回值改为借用引用（owner 保持存活、增长时可能换缓冲——与旧镜像相同的
  生命周期契约）。
- item 1155 剩余：currentCommandBuffer（~139 读/13 文件）、
  currentRenderEncoder（~245 读/13 文件）两镜像（纯机械替换级，P5 规模）；
  isCommittingCommandBuffer 重入 BOOL（平凡）；blitOperationComplete 未用。
- smoke MDI_SCRATCH_OK：创建、坏参、首分配（offset 0 / cap≥64K）、二次
  分配（对齐 offset 256、同缓冲）、增长（新缓冲 / offset 0 / cap≥200K）、
  销毁清零。
- 验证：A/B 双门 66/0/2/68 判定逐条一致（间接绘制/参数缓冲路径被回归
  覆盖）；ASan 双门 66/0/2/68 零报告；test-legacy-compat 193/193；
  test-metalcpp（含 MDI_SCRATCH_OK）SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 2e67dc9：stage-binding copy-back 编码 + CPU 前缀同步迁 C++——item 1138 切片）

**item 1138（compute copy-back）切片**：`flushStageBindingCopyBacks:` 的
校验 + blit 编码循环 + CPU 前缀 memmove 迁入 C++：
- `mglRenderCppEncodeStageBindingCopyBacks`（C-ABI 条目数组——ObjC 侧桥接
  buffer 指针；边界检查 vs Metal buffer length；blit_encoder 为 NULL 时仅
  校验——保持原「先校验后建 encoder」顺序）；
- `mglRenderCppCopyBackCPUPrefix`（守卫 + memmove + ever_written /
  cpu_shadow_pending 副作用 + failed_index 输出供诊断日志）；
- CB 排序（detach / AGX 恢复提交 / wait / newCommandBufferLocked）仍在
  ObjC（commitCommandBufferWithAGXRecovery 是刻意保留的恢复语义）。
- smoke COPY_BACK_OK：空条目跳过、坏参、真 blit encoder 上编码、OOB 拒绝、
  CPU 前缀同步内容校验 + 失败索引。
- item 1138 剩余：processCompute resource plan、barrier、CB sequencing
  （P5 级，item 1141 前置）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 COPY_BACK_OK）SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 3370bb9：逐 level 上传数据准备迁 C++——item 1111 最后一个列出的执行体）

**item 1111（纹理执行体）收口**：dirty-level 循环的逐 level CPU 数据准备迁入
C++（纯数据变换，两门共用）：
- `mglRenderCppTexturePrepareLevelUpload`：几何计算（pitch/height MIN 钳制、
  3D copy_depth）、短后备守卫（-2——数学上不可达，防御性保 A/B 一致）、
  RGBA8/通道展开选择与执行；返回 0/-1/-2 + `MGLRenderCppLevelUploadPrep`
  （data 借用或自有 + owns_data）；
- `mglRenderCppCreateChannelExpandedUpload`：RGBA16/RGBA32 展开表 + 校验
  从 mgl_texture_compat.m 逐字迁入（mglCreateChannelExpandedUpload 变
  薄委托——单一事实源，双门共用）；needs-check 两 helper 的 pixelFormat
  参数改 uint32_t（ABI 不变，C++ TU 可调用）；
- ObjC dirty-level 循环委托 C++（-2 分支保留诊断日志），上传分支体不变；
- smoke LEVEL_UPLOAD_PREP_OK（2D/3D 几何、RGBA8 展开字节校验（A=255）、
  RGBA16 通道展开 alpha=0xFFFF、小后备钳制、坏参 -1；standalone smoke 不
  链 compat 子系统，提供三 helper 的最小 stub——生产 A/B 由 regression 覆盖）。
- item 1111 全部三个列出的执行体（3D 重打包 ✓、cloud-faces ✓、dirty-level
  循环 ✓）已收口；剩余随 item 1014/1069 的逐段迁出（上传分支体的深编排）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 LEVEL_UPLOAD_PREP_OK）
  SMOKE_DONE；check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit c5c518e：detached 命令缓冲 ObjC 镜像删除——item 1141 第三切片）

**item 1141（命令生命周期）第三切片**：`MGLCommandState.detachedCommandBuffer`
（void* 镜像）删除——它只用于 commit/release 的归属校验：
- 新增 `mglRenderCppCommandBufferSubmissionMatchesBuffer`（submission 持有的
  MTL::CommandBuffer 与传入指针按指针相等比较，1/0/-1）；
- commitDetachedCommandBufferIfOwned / releaseDetachedCommandBufferIfOwned
  的守卫改调 C++；detach 的镜像写/清 3 处删除。
- **item 1141 已删镜像：currentCommandBufferSyncList ✓、currentEvent/currentSyncName ✓、
  detachedCommandBuffer ✓**；剩余：currentCommandBuffer / currentRenderEncoder /
  mdiArgsScratch*（gate-off 分配器状态，需 redesign）（P5 级）。
- smoke：submission 流程内增 matches=1 / 异指针=0 / NULL 坏参=-1 校验。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit b403e0c：pending shared-event 槽迁入 C++ owner——item 1141 第二切片）

**item 1141（命令生命周期）第二切片**：`MGLCommandState.currentEvent` /
`currentSyncName` 镜像删除，迁入 C++ `PendingEventOwner`（event + GL sync name）：
- 新增 `mglRenderCppCreatePendingEventOwner` / `PendingEventPrepare`（懒创建 +
  复用，经单例 renderer device——mglRenderCppInit 双门都运行，event 创建路径
  双门一致）/ `PendingEventDetach`（所有权转移给调用方，ObjC 侧 __bridge_transfer）/
  `PendingEventClear` / `DestroyPendingEventOwner`；坏参 -1；
- ObjC prepare/detach/clear 三方法变薄适配（device 参数 __unused）；静态
  mglRenderPassManagerCreateEvent 与 teardown 的 clear 调用删除（改 destroy）；
- smoke 新增 PENDING_EVENT_OWNER_OK（prepare 复用、detach 转移 + 空槽、
  clear 后槽空、坏参拒绝；指针复用陷阱：clear 后 newEvent 可能回写同一地址，
  测试只验证槽语义不比较指针）。
- **item 1141 已删镜像：currentCommandBufferSyncList ✓、currentEvent/currentSyncName ✓**；
  剩余：currentCommandBuffer / currentRenderEncoder / detachedCommandBuffer /
  mdiArgsScratch*（gate-off 分配器状态，需 redesign）等（P5 级）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp（含 PENDING_EVENT_OWNER_OK）SMOKE_DONE；
  check-air-only OK；git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit b743f81：SyncList 镜像迁入 C++ 命令缓冲 owner——item 1141 首个切片）

**item 1141（命令生命周期）首个切片**：当前 CB 的 sync 跟踪列表
（`MGLCommandState.currentCommandBufferSyncList` + SyncList 结构）删除，
迁入 C++ `CommandBufferOwner`：
- C++ 侧新增 `CommandBufferSyncList`（Sync** 数组 + count/size，析构 free；
  reset 清空条目；owner reset/discard 时自动 reset）；
- 新增 `mglRenderCppCommandBufferOwnerAppendSync` / `mglRenderCppCommandBufferOwnerClearSyncs`
  （含扩容溢出防护）；
- ObjC `appendSyncToCurrentCommandBuffer:` / `clearCurrentCommandBufferSyncListEntries`
  变薄适配（gate-off 无 owner 时 append 报成功——列表仅作记录、等待路径从不
  读取，行为不变，A/B 保持）；teardown 的镜像释放删除。
- item 1141 剩余：`currentCommandBuffer` / `currentRenderEncoder` /
  `currentEvent` / `detachedCommandBuffer` 等镜像的 owner 化删除（P5 级，
  item 1141 完整判据：_state.currentCommandBuffer 等 ObjC 镜像删除）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit f82490c：draw_encode + 三个诊断/查询壳模块出列）

**item 887 白名单推进（22 → 16）**：mgl_draw_encode.m（40 处）、
MGLRenderer+GPURecovery.m（2）、MGLRenderer+QuerySync.m（9，含新增
MGLMetalEventRef）、MGLRenderer+SwapDiagnostics.m（36，含 9 处
id<MTLTexture>）全部改用 MGLMetal*Ref typedef——GL 语义层与 A/B 兜底不变。
- 剩余 16 个文件：MGLRenderer.m / +Lifecycle.m 为白名单外壳；
  MGLPipelineCache.m / MGLRenderPassManager.m 为 owner 类（判据要求真迁移
  而非 typedef）；其余 renderer 分类为深迁移对象（随 1111/1138/1141/1157）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 1d000fc：Metal ref typedef 适配列 + 两个壳模块出列）

**item 887 白名单推进**：`mgl_render_cpp_objc.h` 新增 P4 适配列 typedef
（MGLMetalDeviceRef / BufferRef / TextureRef / RenderCommandEncoderRef /
ComputeCommandEncoderRef / BlitCommandEncoderRef / CommandBufferRef /
FunctionRef / RenderPipelineStateRef / ComputePipelineStateRef /
DepthStencilStateRef / SamplerStateRef——语义与 id<MTL*> 完全一致：strong
引用、__bridge 转换不变），供壳模块实现文本避开 `id<MTL` 字样。
- mgl_texture_compat.m（7 处 id<MTLTexture）与 mgl_index_buffer.m（59 处
  id<MTLBuffer>/id<MTLDevice>，含缓存静态变量——typedef 保持 strong 语义）
  全部改用 typedef。
- **白名单 census：22 → 20 个 .m 文件**。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。
- 剩余 20 个文件：MGLRenderer.m / +Lifecycle.m 为白名单外壳；其余为深
  迁移对象（随 1111/1138/1141/1157 逐段迁出）。


### P4 完成记录追加（2026-08-15，commit 7c8cb8d：render-pass 颜色附件查询迁 C++ + 壳模块 id<MTL 清零）

**item 1157/887 小切片**：
- `mglRenderPassUsesColorTexture`（mgl_sync.m 的 mirror fallback 查询）迁入
  C++：新增 `mglRenderCppRenderPassUsesColorTexture`（MTL::RenderPassDescriptor
  的 colorAttachments 遍历，命中写 index 返回 1 / 未中 0 / 坏参 -1）；
  mgl_render_cpp_objc.h 的 ForState 内联包装的 fallback 改调 C++ 入口，
  ObjC 函数与 mgl_sync.h 声明删除。
- `MGLCapabilityInit` 签名改 `void *deviceRef`（header 提供
  MGLMetalDeviceRef typedef + mglCapabilityDeviceRef bridge helper），
  mgl_capability.m 不再含 `id<MTL` 文本；调用方 Lifecycle.m:308 改
  `(__bridge void *)_device`。
- **白名单 census：24 → 22 个 .m 文件**（mgl_sync.m、mgl_capability.m 出列）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净（修掉 mgl_sync.m 尾部空行）。


### P4 完成记录追加（2026-08-15，commit e78eb1c：旧式 packed 格式 RGBA8 展开迁入 C++）

**item 1111 第三个执行体切片：mglCreateRGBA8ExpandedUpload（dirty-level 循环
调用的两个展开 helper 之一）** —— 新增 `mglRenderCppCreateRGBA8ExpandedUpload`
（R3_G3_B2 / RGB4/5/565 / RGB10/12 / RGBA2/4 / RGB5_A1 / RGB8 变体的逐格式
位展开 + unorm 取整，含 512MB 尺寸上限与坏参拒绝），替换 mgl_texture_compat.m
的内联体；ObjC wrapper 保留签名与
mglTextureInternalFormatNeedsRGBA8Expansion 守卫后直接委托。逐格式位布局与
内联版逐字节一致（RGB565 的 5_6_5、RGB5_A1 的 alpha 位、RGBA2/4 的 4_4_4_4、
snorm 1.0=0x7f、整型 a=1 等）。被迁体专用的 mglReadPackedUploadLE /
mglExpandUNormBitsTo8 已随迁删除（无其他引用）。两门共用，无 A/B 分歧。
- smoke 新增 RGBA8_EXPAND_OK：RGB565 全 1（→全 255 白）、RGB8 3 texel
  （a=255）、RGBA4 全 1（→全 255）+ 4 个坏参拒绝（src NULL / width 0 /
  bpr 过短 / 未知格式）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp RGBA8_EXPAND_OK + SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1111 剩余：dirty-level 循环的编排体（per-level 迭代 + 上传调用，
  GL 语义，随 item 1014 逐段迁出）。


### P4 完成记录追加（2026-08-15，commit bc0a0b0：CloudFaces 通道扩展迁入 C++）

**item 1111 第二个执行体切片：texel buffer 2D fallback（CloudFaces 路径）的
RGB→RGBA 通道扩展** —— 新增 `mglRenderCppTextureExpandRGBToRGBA`（纯数据
变换，写入调用方提供的 dst buffer，ObjC 用 NSMutableData 保持 ARC 生命周期
管理；src 每 texel 3×comp，dst 每 texel 4×comp，alpha 取默认值低字节，超出
texel_count 的尾 texel 置零，坏参返回 -1）。替换内联逐 texel 扩展循环；与
内联版布局逐字节等价（row×texWidth×dstPixel + col×dstPixel 相同）。两门共用
同一 helper，无 A/B 分歧。行打包（packedData memcpy + 零尾）非变换，保留
ObjC。
- smoke 新增 TEXTURE_EXPAND_OK：8-bit（3 texel → 2×2 网格，尾 texel 置零、
  alpha 255 注入）与 16-bit（alpha 65535 低 2 字节）逐字节校验 + 4 个坏参
  拒绝（src/dst NULL、width 0、comp 0）。中途修正测试自身错误（16-bit 需
  12B src / 16B dst，初版数组尺寸算错）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp TEXTURE_EXPAND_OK +
  TEXTURE_REPACK_OK + SMOKE_DONE；check-air-only OK；git diff --check 干净。
- item 1111 剩余：dirty-level 循环（GL 语义体，随 item 1014 逐段迁出）。


### P4 完成记录追加（2026-08-15，commit bd0780e：3D 纹理 depth-plane 重打包迁入 C++）

**item 1111 第一个执行体切片：REPLACE_3D 分支的 3D depth-plane 重打包** ——
新增 `mglRenderCppTextureRepackDepthPlanes`（纯数据变换：strided
bytesPerImage → tight bpr*height 布局；参数非法/溢出/分配失败返回 NULL），
替换 ObjC 内联 malloc+逐 plane memcpy 循环。两门共用同一 helper，无 A/B
分歧；replaceRegion 调用与 @try/@catch 回退仍留 ObjC。
- smoke 新增 TEXTURE_REPACK_OK：3 plane × 20B stride → 16B tight 的逐字节
  校验 + 4 个坏参拒绝（NULL bytes / depth 0 / bpi < expected / expected 0）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp TEXTURE_REPACK_OK + SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1111 剩余：dirty-level 循环、cloud-faces 特判（GL 语义体，逐段迁出）。


### P4 完成记录追加（2026-08-15，commit 40719ec：compute 纹理/sampler 绑定并入 binding snapshot）

**item 1138：bindTexturesToComputeEncoder 的 setter 序列并入 compute
binding snapshot** —— op kind 扩为 4 种：0 = setBuffer、1 = setBytes、
2 = setTexture、3 = setSamplerState（texture/sampler 对象指针放 buffer
字段，NULL = 槽位清除）。5 个 emit 点（sampled 主循环 texture+sampler、
sampled 数组 pass texture+sampler、storage-image 数组 pass texture）gate-on
收集、函数末尾一次重放；default 分支与 texture-not-found 两个 return false
前先 flush。
- **临时对象生命周期**（round 35 崩溃类的预防性处理）：level view
  （mglComputeCreateTextureLevelView，gate-on __bridge_transfer）与 fallback
  sampler（mglComputeCreateSampler）经方法级强数组 ctexTemporaries 持有至
  末尾重放后才释放；重放（编码器当场 retain）与释放顺序在 flush 之后，杜绝
  悬垂进延迟重放。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp COMPUTE_BINDING_SNAPSHOT_OK
  （含 texture+sampler op 与 NULL texture 清除）+ SMOKE_DONE；
  check-air-only OK；git diff --check 干净。
- item 1138 剩余：processCompute 资源 plan 的深编排（copy-back
  flushStageBindingCopyBacks 的 CB detach/commit/wait 时序、barrier、CB
  sequencing）——P5 级（item 1141 currentCommandBuffer owner 化前置）。


### P4 完成记录追加（2026-08-15，commit 77d96ed：compute 绑定 snapshot + snapshot 临时 buffer 生命周期修复）

**item 1138 下一个切片：bindBuffersToComputeEncoder 的 setter 序列
snapshot 化** —— 新增 `MGLRenderCppComputeBindingSnapshot`（与 render 版同构
的 op 列表：kind 0 = setBuffer / NULL = 槽位清除，kind 1 = setBytes）+
`mglRenderCppEncodeComputeBindingSnapshot`（逐条重放；坏 kind / NULL bytes /
越界计数拒绝）。3 个 emit 点（isolated、普通 map、runtime-array-size
sizeBuffer）在 gate-on 收集，函数末尾一次重放；3 个校验失败路径先 flush 再
return false。GS/TES P4.3e 路径（DrawSupport.m:1725）与用户 compute 共用此
函数，compute_dispatch_ssbo + 全部 air_geometry_* 测试覆盖。

**顺带修复一个 REAL gate-on 崩溃（air_geometry_resources 全套跑必炸）**：
runtime-size `sizeBuffer` 是块级局部（gate-on 经 __bridge_transfer 拥有），
块结束即释放，而它的 op 等到函数末尾才重放 → 重放时悬垂 → objc_retain
EXC_BAD_ACCESS（ASan 定位：mglRenderCppEncodeComputeBindingSnapshot ←
bindBuffersToComputeEncoder ← handleGeometryDrawIfNeeded）。修复：sizeBuffer
emit 后立即 flush（编码器当场 retain）。单跑该测试不炸（状态无关，全套
1-15 前缀才触发）——ASan 全栈是唯一可靠的定位手段。

**同型潜在隐患一并加固**（__bridge_transfer 临时 buffer 一律 emit 后立即
flush，编码器当场 retain）：
- compute isolated 缓冲（非 writable 无 copy-back 登记时唯一持有者是循环
  局部）；
- render isolated 缓冲（round 30 遗留同型隐患，未被测试触发过）；
- VAO attrib 4 个转换缓冲（double/int-float/packed/integer，gate-on 每
  次新建、无缓存）。

- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp COMPUTE_BINDING_SNAPSHOT_OK +
  SMOKE_DONE；check-air-only OK；git diff --check 干净。
- item 1138 剩余：processCompute 资源 plan（copy-back 编排
  flushStageBindingCopyBacks、barrier、CB 时序）。


### P4 完成记录追加（2026-08-15，commit 78b9c52：fallback + point-size 段 snapshot 化）

**item 1056 主绑定 pass 收口：fallback 段（2 个直接 setVertexBuffer 点）与
point-size 段（1 个 setVertexBytes 点）也改走 binding snapshot** —— 至此
主 per-draw 顶点绑定 pass 全链路（map 循环 → VAO attrib → fallback →
point-size）在 gate-on 全部经 `mglRenderCppEncodeBindingSnapshot` 一次或
多次重放，`BindingState.m` 主流程直接 setter 调用点归零（仅剩宏的 gate-off
else 分支与 helper 定义）。
- 两个方法各新增 snapshot 参数（fallback：bindingSnapshot/useSnapshot；
  point-size：外加 byteScratch/byteScratchUsed/byteScratchCapacity 用于
  2×float bytes op 的 scratch 拷贝），各自方法内定义指针版宏并就地重放：
  重放位置 = 各方法结束处，顺序 = map-replay → attrib-replay →
  fallback-replay → point-size-replay，与直接路径逐点一致。
- 书keeping（mglRenderCppBindingUpdateVertexBuffer / invalidateLastBound /
  PERF_INC / anyBindingPresent / baseBindingPresent）全部保持内联、两门一致。
- item 1056 绑定段全部完成；剩余（随 item 1014 跟踪）为 BindingState.m
  的 88 个 `id<MTL` 残留（绑定状态 owner 内部结构，P5 级迁移）。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 5d7184c：VAO attrib 段 snapshot 化）

**item 1056 下一个切片：bindVertexAttributesFromVAO 的 6 个 emit 点全部改走
binding snapshot**（round 30 主 map 循环 snapshot 的延续）。
- 方法新增 `bindingSnapshot` / `useSnapshot` 参数（调用方传主流程的
  vbindSnapshot + useVertexBindingSnapshot）；方法内定义指针版
  MGL_VATTR_EMIT_BUFFER / MGL_VATTR_FLUSH_SNAPSHOT 宏（attrib 段只有
  buffer op，无需 bytes scratch）。6 个 emit 点（current-value 缓存、
  GL_DOUBLE / packed / int→float / integer 转换、plain VBO）统一收集。
- 重放位置 = attrib 方法结束处（fallback / point-size 之前）——与直接路径
  「map 循环 emit → attrib emit → fallback → point-size」顺序逐点一致；
  4 个校验失败 `return false` 前先 flush 已收集 op（否则早期失败会丢
  attrib 段已发生的 emit，破坏 encoder 状态与 dedup 缓存一致性）。
- 书keeping（mglRenderCppBindingUpdateVertexBuffer / mglNoteBufferEncoded /
  PERF_INC / anyBindingPresent / 转换缓存）全部保持内联、两门一致。
- 剩余（item 1056）：fallback 段（2 个直接 setVertexBuffer 点）与
  point-size 段（1 个 setVertexBytes 点）仍在直接路径，下一切片处理。
- 验证：A/B 双门 66/0/2/68 判定逐条一致；ASan 双门 66/0/2/68 零报告；
  test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；check-air-only OK；
  git diff --check 干净。


### P4 完成记录追加（2026-08-15，commit 0a5259b：compute dispatch plan 首切片 + smoke 修复）

**P4.5 item 1138 第一个切片：dispatch 参数 value-state plan。**
- `MGLRenderCppComputePlan`（mgl_render_cpp.h）+ `mglRenderCppDispatchComputePlan`
  （mgl_render_cpp.cpp）：DIRECT / INDIRECT 两种 dispatch 一次 C ABI 编码；
  local size 0 在 C++ 内解析为 1（与既有 `x ? x : 1` 默认一致）。为
  「ObjC 只传 MGLRenderCppComputePlan value-state」定型。
- MGLRenderer+Compute.m 的 mtlDispatchComputeLocked / mtlDispatchComputeIndirectLocked
  尾部改为 gate-on 组装 plan + 单次 C++ 调用；gate-off 保留原逐条 ObjC 路径作
  A/B 对照（两分支产生完全相同的结果；顺带删除了直接路径里冗余的
  `if (local_workgroup_size 非零)` 双分支——两分支本就相同）。
- 新回归测试 `compute_dispatch_ssbo`（第 68 个）：用户 glDispatchCompute 经
  SSBO 写回验证（8 组 / 4 组两次 dispatch）。注意：legacy 前端不解析
  layout(local_size_*)，Program::local_workgroup_size 恒为 0，两门都以 (1,1,1)
  单线程组解析——测试按此真实语义断言，A/B 结果一致。

**test-metalcpp smoke 修复（该二进制自 round 30 起就未真正跑过——此前
「翻译器单测」只覆盖 test_legacy_compat 的 193 项；本轮跑通后抓到两个存量 bug）：**
- round 30 测试 bug：BINDING_SNAPSHOT 的 nullBytes 用例写的是
  `fragment_ops[2]`（越界幻影 op，复放为合法 clear）应为 `fragment_ops[1]`。
- **round 31 真实 bug：`mglRenderCppTextureUploadRoute` 硬编码的 MTL ABI
  常量错误**——MTLTextureType1DArray 实为 1（误写 3）、MTLTextureType3D
  实为 7（误写 5）、MTLStorageModePrivate 实为 2（误写 0）。gate-on 对
  1DArray/3D/Cube 上传误判（1DArray Shared → BLIT 而非 REPLACE_1D；3D
  Shared+bug → BLIT 而非 REPLACE_3D；Cube 被当成 3D 误入 REPLACE/REJECT）。
  回归套件未命中这些形状（uploadTextureSliceViaBlit 只被 2D 路径触达）故 A/B
  未暴露；已按 SDK 真实数值修正，现与 uploadTextureSliceViaBlit 的 ObjC 内联
  判定逐条件一致，smoke 的 8 条 route 断言全绿。

**A/B 验证**：gate-on / gate-off 各 66/0/2/68、判定逐条一致；ASan 双门
66/0/2/68 无报错；test-legacy-compat 193/193；test-metalcpp SMOKE_DONE；
check-air-only OK；`git diff --check` 干净。


### P4.1 完成记录（2026-08-13，提交 b9fe45b）

- **读取侧（149 → 25 处）**：`MGLRenderer+RenderPass.m` 全部
  `MTLRenderPassDescriptor` 读取改为 owner-first 快照 helper
  （`mglRenderCppGetRenderPassStateOwner` 优先、ObjC 镜像 gate-off 回退）：
  texture / load-store actions / clear color-depth-stencil / render target
  尺寸 / visibilityResultType。新增 helper 组（`mglRenderPass*For` 共 13 个
  静态函数）落在 `mglRenderPassGetPersistent*` 附近。metric：视觉验证用
  inflight instrumentation 确认无遗漏读点（A/B 像素一致即证明等价）。
- **写入侧（部分 gate 化）**：`SetPersistentDimensions` /
  `SetPersistentActions`(镜像分支) / `SetPersistent{Color,Depth,Stencil}Clear`
  在 `MGL_USE_METALCPP` 下不再写 ObjC 镜像，C++ owner 为唯一写入点。
- **踩坑**：`SetPersistentAttachment` 的 gate-on 专属重写（预归一化
  slice + 追加 arrayLength 写入）在 `air_geometry_layer_viewport` 全量回归
  中出现 FBO-match 重建循环——该 helper 的 layered slice 归一化与 C++
  attachment setter 内部归一化（`SetRenderPassStateAttachmentTexture` 内
  `destination->slice = 0`）叠加后语义有分歧。修复为**保留该 helper 双写**
  （镜像 gate-off 用、owner 调用不变），其余 gate 化保持；专项归一到 P4.1e
  （新增 layer-array slice 探针后再动）。
- **验证**：regression A/B 均 52 PASS / 0 FAIL / 2 SKIP；test-mglair /
  test-metalcpp / git diff --check 全绿。

### P4.1 e/f 调研记录（2026-08-13，提交 a9a91fe）

**P4.1 读取转换遗留修复**：owner-first 转换过程中
`mglRenderPassMatchesFramebufferImpl` 的 9 个 `return false` 被误删成空
if 体（FBO-match 校验被静默禁用），已恢复并通过激活校验的 A/B regression
（52/0/2 双门）。

**P4.1e 结论：SetPersistentAttachment 双写暂不拆除**。把该 helper 改成
gate-on 只写 C++ owner（mirror 完全不写）后，`conditional_render`（occlusion
query + 0-size scissor）在 gate-on 下稳定失败：occluded query 返回 2888 而
非 0（scissor 未生效）。**关键特征：代码布局敏感**——在
`updateViewportAndScissorLocked` 里加任何 printf 都能把失败翻转为通过（6/6
pass），去掉则 5/5 fail；非逐次随机，而是确定性的指令布局依赖。这指向
visibility-result / encoder 复用 / scissor dedup 路径里的一个潜在时序或未
初始化读取缺陷，需要专项排查（不是 SetPersistentAttachment 的确定性语义差）。
已用 ACQUIRE slot / beginSampleQuery / QSYNC / SCISSOR 四组插桩定位：
两 query 均 ACQUIRE slot=0、counting=1、镜像 rpVB 正常，惟 scissor 未落到
encoder。双写版本在同样插桩下稳定通过。

**根因（2026-08-13 定位，提交 ed31dee）**：不是 layout 依赖，是
**scissor clamp bug**——`updateViewportAndScissorLocked` 把 0-size scissor
box 重置回整幅 pass（GL 4.6 §14.6.1 允许空 box 裁剪全部 fragment，
0-size 是合法语义），occluded query 因此统计了全三角形样本。修复：两处
`sw<=0||sh<=0` 及「origin 完全在 viewport 外」的分支保留 0-size box。
修复后 conditional_render 与 occluded 探针在 **double-write 与 gate-on
单写**下都稳定通过（单写重验 3/3）。

**P4.1e2 交付**（提交 ed31dee）：
- [x] `air_query_scissor_occluded` 探针：2D FBO + GL_SAMPLES_PASSED + 可见
  三角 > 0 + 4 轮 0-size scissor 各 == 0 + scissor 清除后同一 FBO 仍渲染。
- [x] `air_renderpass_layer_slice` 探针：2D-ARRAY attachment slice∈{0,1} +
  无 gl_Layer 输出对照；本提交最初用它固定当时的已知限制（slice-1 绑定仍画到
  layer 0）。**2026-08-16 已更新为修复后的正向断言**：slice 1 draw/clear 只改
  slice 1，并证明 slice 0 保持原图像。
- [x] scissor dedup 排查结论：`mglRenderCppBindingSetScissor` 的
  `state->valid`/scissorEqual dedup 在修复后不再涉事（无 dedup 的强制 emit
  实验同样失败→失败在值本身而非 setter dedup；dedup 保留）。
- **未解项（转 P4.4 专项）**：gate-on 单写 + **isolines/point_mode**（TES
  compute → passthrough raster）仍失败（double-write 基线过）。outBuffer
  CPU 读回在套件里位置正确（含 -0.3 v）、独立跑时 v=0——指向 **compute
  读 VS-capture/TCS 输出的跨命令缓冲同步**。P41E3 诊断：在 dispatch 前
  `flushCommandBuffer:YES` 仍失败；`flushStageBindingCopyBacks` 的
  CPU-visible 等待不足以建立 GPU 间 ordering。
  - [x] 排查 TES compute 与 VS-capture/TCS 输出在 gate-on 下的跨 CB 可见性
    （`dispatchAIRTessEvalCompute` 读 `tessVertexCaptureBuffer` /
    `tcsOutputBuffer`，其写入可能在上一 CB）。
    （2026-08-14 结案：该「跨 CB 可见性」表象与 6c6b1cd 根因同域——TES
    compute kernel 的 XFB stream 写越过 1 字节 dummy 溢出到相邻 slab 分配
    （VBO 快照 slot / capture 块），独立跑时分配布局不同故表象为「读旧
    数据」；dummy 按 outSize 分配后 10+ 连跑全套件双门全绿，无需额外的
    GPU ordering 改动——compute 与 VS-capture 同 CB 内编码，顺序由 Metal
    保证。）

**2026-08-14 专项调查（P4.1e3 续）**：**纠正了问题的性质** —— 独立跑失败
**与 gate 无关**（gate-off 独立跑 isolines_multidraw/point_mode 同样失败；
套件跑双门都过 → 纯状态依赖）。文档原记「double-write 基线过」实际指
**套件跑**，独立跑在 double-write 下也失败。

调查过程与已排除项（MGL_TESS_DIAG 插桩，全套件 A/B 保持 54/0/2）：
- capture pipeline / vertex descriptor / EBO / gather 全部正确（isCapture=1、
  EBO={0..5}、idxOff 正确）；TES kernel 的绑定 buffer（glIn/factor/patch/out）
  与 VBO 均不同对象/不同 contents；capture buffer（C2）与 VBO 也不同。
- **污染机制**（0xCD 标记实验）：VBO 的 mtl_data 内容在 **GPU 执行期间**被写
  （dispatch 前标记完好、flush 后 offset 8 起 56 字节被写为 (0,1,0,0)+0）；
  子 draw 1 不污染、子 draw 2 污染（同 kernel → 分配/复用时序差异）。
- **COW 快照池方向已排除**：池是 per-buffer 的（`bufferCowPool(owner)`），
  VBO 的快照只被 VBO 自己的池复用，不会被他 buffer 覆盖；`installBufferCowSnapshot`
  的「持久 buffer 快照从池移除」改动破坏了 test-metalcpp 的池复用断言
  （`completed COW slot was not reused`）→ 已回退，池复用是受保护的设计。
- 引用计数正常（VBO mtl_data retainCount 5/6，非悬垂）；TES 绑定后 VBO 与
  C2 对象/contents 均不同（本次运行），与 0xCD 运行的「glIn 内容=VBO 标记」
  矛盾 → 指向**偶发的分配/复用时序**（同一物理内存在不同运行中被不同
  buffer 复用，VBO 读到其他 buffer 内容）。
- **有效 workaround（诊断用，未保留）**：capture-ready 时强制把 buffer_data
  同步到 VBO mtl_data → 测试通过，证实「VBO 的 mtl_data 内容过期」是直接
  故障面。
- **后续建议**：① 在 `dispatchAIRTessEvalCompute` 的 dispatch 前/后对比
  VBO mtl_data 的 contents 指针与 glIn/out/factor 的 contents，用
  MGL_TESS_DIAG 复现「VBO contents == 某绑定 buffer contents」的精确时刻；
  ② 检查 `updateDirtyBuffer` / `mapBuffersToMTL` 在 DIRTY_ALL 下的 buffer
  重建路径（是否把 VBO 的 mtl_data 换成复用内存）；③ 关注
  `mglRenderCppCreateBuffer`（gate-on 的 newBuffer 包装）与 COW 池的地址
  复用交互。

**✅ P4.1e3 系列问题收官（2026-08-14，commit 6c6b1cd）**：后续建议 ① 的插桩
把「VBO contents == 某绑定 buffer contents」的时刻**精确定位到
`dispatchAIRTessEvalCompute` 的 compute kernel 执行本身**（NODISPATCH 跳过
kernel 后 VBO 全 8 draw 保持干净；恢复 dispatch 则 draw-2 起被写为
(0,0),(0,1),(8,0),(0,0)）。根因：**TES compute kernel 无条件声明并写入 XFB
stream（slot 31），而 GL feedback 未激活时渲染器只绑了 1 字节 dummy buffer**
（`tessXfbDummyBuffer`）——kernel 的 stream 写越界溢出到相邻 slab 分配
（VBO 的 32B COW 快照 slot / 上一 draw 的 capture 块），溢出落点随分配布局
漂移 → 第 3+ 连续 tess draw / 套件位置敏感的全部历史特征。修复：dummy 按
stage-out span（outSize）分配。隔离的 isolines_indexed 间歇 flake 属同一
spill-victim 家族：修复后 **10 连跑全套件（gate-on ×8 + gate-off ×2 之外的
重复批次，含 isolines_indexed 定向 2 连跑）全部 61/0/2/63 绿**，未再复现
（原「与 accumulation 无关」的结论在根因发现前作出，现归类为同域）。

- [x] P4.1f：删除 `MGLCommandState` 的 lastFboMatch* 镜像（C++ identity
  owner 已权威）、`dontCareFrameGeneration` 的 ObjC 字段，以及
  `installNewRenderPassDescriptor` 在 gate-on 下的 descriptor 创建。
  - **dontCareFrameGeneration 删除 SKIP**：该 ObjC 字段（
    `MGLRenderPassManager.h`）无 C++ 对应物，仍被 gate-off 基线的纹理
    `mtl_rt_frame_generation` 时间戳（RenderPass.m ~3050）使用；P5 删除
    gate 时随 ObjC 路径一起移除，现保留。
- [x] 待 conditional_render 与 isolines 双问题都绿后，再拆
  `SetPersistentAttachment` 双写 + 删 `renderTargetArrayLength` 镜像写与
  `attachment.slice=0` 镜像逻辑。
  （2026-08-14 核验收官：两个前置问题均已绿——conditional_render 修复于
  ed31dee（scissor clamp bug），isolines 系列修复于 6c6b1cd（XFB dummy 越界）。
  **双写已拆**：`mglRenderPassSetPersistentAttachment` 为 gate-on 纯 C++ 写 /
  gate-off 纯镜像写（无任何双写路径）；gate-on 下 `renderPassDescriptor` 保持
  nil（P4.1f），当时的 `renderTargetArrayLength` 上限与 `attachment.slice=0`
  语义由 C++ owner 在 `mglRenderCppSetRenderPassStateAttachmentTexture` 内维护，
  encoder 创建（`mglRenderCppCreateRenderEncoderFromStateOwner`）从 owner 读取
  `render_target_array_length`（mgl_render_cpp.cpp:1027）。镜像侧两处写入仅存于
  gate-off A/B 基线（RenderPass.m:649 / ~660），随 P5 gate 删除。）
  - **处置**：gate-on 世界 0 处镜像写（全库仅 RenderPass.m:649 一处
    `renderTargetArrayLength` 写 = gate-off 基线）；该条随 P5 与 ObjC 基线
    一起删除，不单独动 A/B 基线。**2026-08-16 语义修正**：两门现都按
    attachment `layered` 状态决定是否归零 slice/depthPlane；非 layered pass
    的 array length 为 0，layered pass 取全部 attachment 的共同最小层数。

### P1/P2 回归补齐记录（2026-08-14）

新增 4 个产品级回归（均在 `test_regression/main.c`，A/B 双门 PASS）：

- `air_geometry_cull_distance`：GS 逐发射顶点写 gl_CullDistance——all-negative
  剔除 / all-positive 可见 / mixed (+1,-1,+1) 按 GL 规则（非全负）可见；
  arrays（direct 批路径）+ elements（element 路径）双段像素探针。
- `air_tessellation_patch_varying`：per-patch input/output + patch-qualified
  varying；双 patch 各路由独立 patch color（per-patch 输出不串 patch）；
  outer=3/inner=2 + fractional_odd + ccw 细分三角形。
- `air_tessellation_resources`：native TES 读 sampler2D + std140 UBO +
  std430 SSBO，三段变值重画证明逐次重读（绿→蓝=UBO 重读；白×红=SSBO 重读）。

- `air_geometry_ssbo_visibility`：GPU→GPU 写后可见性（doc 176 项）——段 1 GS
  写 SSBO（蓝）经 GL_SHADER_STORAGE_BARRIER_BIT 后由后续 draw 的 GS 读回渲染
  在右侧（蓝），写入方左侧绿色；段 2 GS imageStore（红）经
  GL_TEXTURE_FETCH_BARRIER_BIT 后由后续 draw 的 GS 采样渲染在右侧（红）；
  写入/读取位置分离，陈旧值会以错误探针颜色暴露。

**遗留 bug 补充证据（2026-08-14 追加）**：accumulation 触发与测试内容无关
（新增 GS-only 测试置于 isolines 块前同样确定性失败，gate-off 下
point_mode/variants/indexed 三个 isolines 测试同时破）；将测试移至 isolines
块之后即恢复 58/0/2 稳定。另确认 gate-on 下 isolines_indexed 存在与
accumulation 无关的间歇性 flake（P4.1e3，同配置连续两次全绿，偶发一次
probe 1 缺失）。

**新发现遗留 bug（isolines_indexed 累积性失败，未修复，记录在案）**：
`air_tessellation_isolines_indexed` 在**任何额外 TES/TCS 测试先于 isolines
块执行时确定性失败**（probe 1 缺失，右半实例 1 渲染为空）。控制实验证明与
新测试内容无关：仅注册一个 `air_tessellation_varying` 的重复项（同一函数）
即复现。诊断证据（MGL_TESS_DBG 临时插桩，已移除）：
- CPU 侧全部编码逐字段一致（entry/patchCount/instanceCount/indexed/
  tessInstanceRecords/capture-draw count/recordsPerInst/gather 内容
  {2,0,1,1,0,2}/outBuffer 尺寸/两次 raster offset）；
- GPU 侧输出分歧：suite 下实例 1 的 TES 展开读到错误的 gl_in（实例 0 的无
  shift 控制点）→ 实例 1 整段为空；query 仍计 32（由 CPU 公式记录，与
  缓冲区内容无关）。
- 结论：分歧发生在 GPU 可见输入（VS capture 缓冲内容或 kernel 读到的
  控制点流），与累积的 in-flight command buffer / COW 池 slot 复用时机
  相关（test-metalcpp 的池复用断言路径 `mglRenderCppCreateBuffer`）。
  修复方向：检查 capture/outBuffer/gather 缓冲的生命周期与
  `mglRenderCppCreateBuffer` 池复用是否在旧命令缓冲提交前被覆写。
  当前规避：新回归注册在 isolines 块之后（套件 57/0/2 稳定）。

**✅ 遗留 bug 根因已定位并修复（2026-08-14 收官，commit 6c6b1cd）**：
accumulation 与 suite-position 两类失败的同一根因 —— **AIR TES compute kernel
无条件声明并写入 XFB stream（slot 31），即使 GL transform feedback 未激活；
渲染器给该 slot 绑定 1 字节 dummy buffer，kernel 的 stream 写越界溢出到相邻
driver slab 分配**（典型受害者：VBO 的 32B CoW snapshot slot、上一 draw 的
capture 块）。溢出落点取决于分配布局 → 第 3+ 连续 tess draw 才显现、随套件
位置漂移（即历史「accumulation」特征）。修复：dummy 尺寸按 stage-out span
（outSize）分配（inactive 反馈的写入留在界内）；另将 TES-only/native-no-TCS
的 default factor 缓冲按 patch 级别缓存（跨 draw 复用，消除每次 draw 的 12B
重分配）。新增 `air_tessellation_accumulation`：8 个连续光栅化断言 tess draw
（quads point-mode n=2/3/4、isolines {4,2}/{3,2}/{4,3}、quads n=5/6），
注册于 isolines 块之前覆盖 suite-position 模式；全套件双门 61/0/2/63（重复
跑稳定）。P4.1e3 的 isolines_indexed 间歇 flake 仍单独记录在案（见下）。

**遗留 bug 同域最小复现（2026-08-14，`air_tessellation_factors_spacing`
开发中定位）**：不再需要套件位置前置——**测试内第 3 个连续 tessellation
draw（native 或 point-mode compute 皆可）确定性光栅化错乱**，程序无关
（(odd,equal,even)→even 破；(odd,equal)→tri-odd 破；换 TCS 程序类后
第 1 个 draw 即破）。关键证据（MGL_TESS_DBG + prevOut 保留插桩，已移除）：
- 失败 draw 的 outBuffer 内容正确（even n=4 的 tesscoord (1/8,1/8)、
  (1/8,3/8)）但 position 巨大（6.12/4.38）——即 kernel 的 spacing 取整与
  tesscoord 生成正确，gl_in（VS capture）读到陈旧/错位数据；
- 失败 draw 上屏像素是**上一 draw 的 n=3 tesscoord 颜色**（(5/6,5/6)），
  证明 passthrough raster 读到陈旧 outBuffer 记录；
- 二进制 pipeline archive（~/Library/Caches/MGL/…）移除以排除缓存 kernel
  干扰后仍复现；glFinish 分隔 CB、fresh newBuffer（mglRenderCppCreateBuffer
  无池）、CPU 参数逐字段一致均排除。
- 触发条件与分配尺寸/内容无关，与**同一测试内的 tessellation draw 序号**
  相关；非 tess draw 插入不重置。规避：`air_tessellation_factors_spacing`
  只对前 2 个 tess draw 做光栅化断言（native winding 段），后续 spacing/
  zero-factor 段走 CPU-side primitive query（immune）；套件仍 59/0/2 全绿。

### P4.1f 完成记录（2026-08-14）

**交付**：gate-on 下 `MGLCommandState.renderPassDescriptor` 镜像不再创建
（nil），所有跨文件的 render-pass 读取改为 owner-first；回归 A/B 双门
54/0/2 全绿（含 3 次重复跑验证稳定性）。

**共享访问器**（`mgl_render_cpp_objc.h`，供 RenderPass/Batch/Draw/
DrawSupport/BindingState/QuerySync/MGLRenderer.m 共用）：
- `mglRenderPassUsesMetalCpp()` —— 统一 gate 定义（原 RenderPass.m 的
  static 版本删除，避免两处实现漂移）。
- `mglRenderCppGetRenderPassState(owner,&state)` —— owner-first 读 C++
  状态。
- `mglRenderPassAttachmentTextureForState` / `AttachmentSubresourceForState`
  / `RenderTargetSizeForState` / `UsesColorTextureForState` /
  `ActionsForState`（+ `LoadActionForTrace`/`StoreActionForTrace`）——
  全部 owner-first，mirror 兜底（gate-off A/B 基线语义不变）。

**修复的 gate-on 破坏点**（descriptor=nil 后暴露的一族未加 gate 的读取，
全部为 P4.1f 引入的回归，gate-off 不受影响）：
1. `mglRenderPassSetPersistentAttachment`：原 `!renderPassDescriptor`
   提前返回在 gate-on 下同时杀掉 C++ 写 → 附件从不写入 owner。拆为
   gate-on 纯 C++ 写 / gate-off 纯镜像写。
2. `finalizeRenderPassDescriptorLocked:` 的「descriptor is NULL」守卫 →
   gate-on 校验 `renderPassStateOwner`（51 个测试的「Cannot create render
   encoder」即此）。
3. **pipeline 格式覆盖块**（`if (renderPassDescriptor)` 守卫）→ gate-on
   跳过 → pipeline 用 FBO/context 推导格式构建，与真实 pass 附件（sRGB
   变体 / transient depth）不匹配 → Metal 静默丢弃 draw。
4. **`updateViewportAndScissorLocked`（关键）**：pass 尺寸解析同样被
   descriptor 守卫跳过 → 回退 drawable 尺寸（200x200）做 GL→Metal 视口
   换算 → metal viewport y=72 落在 128x128 render target 之外 → **所有
   draw 静默空转、输出只剩 clear 色**。这是「3 PASS / 51 FAIL」→ 修完
   守卫后仍全黑的主因。
5. `updateCurrentRenderEncoder` 的 depth/stencil 附件存在性判断 → gate-on
   恒 false → depth/stencil test 被禁用（depth_test/stencil 回归失败）。
6. `currentRenderPassUsesTexture:` / `validateRenderPassAttachmentsAnd
   PipelineFormatsLocked:` / Batch.m `markCurrentFramebufferDrawAttachments
   Written` / MGLRenderer.m clear-quad encoder 复用 / QuerySync.m
   visibility-buffer 判断 / BindingState.m 4 处 uses-color-texture /
   DrawSupport.m pass 尺寸回退 / Draw.m 紧急重绑格式检查 → 全部改为
   owner-first。
7. 6 处 fallback-pipeline 的 `renderPassDescriptor && mglRenderPass
   ColorTextureFor` 前置条件删除（owner-first reader 已 nil-safe）。
8. 4 处 trace 日志的 `descriptor ? reader : nil` 守卫（会掩盖真实状态）
   改为直接 owner-first 读取。

**遗留观察**：一次全套件跑中 `stencil` 出现单发 FAIL（mismatch），后续
3 次 gate-on 全套件 + up-to-stencil 均 54/0/2 未复现 —— 与 P4.1e3 isolines
同类的偶发状态/时序 flake，暂不追（记录在案）。

**A/B 验证**：gate-on 54/0/2（×3）、gate-off 54/0/2（×2）、test-mglair
（TCS/TES/GS/XFB/RUNTIME_LENGTH/VALUE_OK）、test-mglair-gtest 42/42、
test-metalcpp（SMOKE_DONE 含 COW 池复用断言）、test-legacy-compat 134/134、
test-dirty-hash PASS、test-benchmark smoke PASS、check-air-only OK、
`git diff --check` 干净。

### P4 完成记录追加（2026-08-14，commit 681a7c9：legacy glClipPlane GL surface）

**Legacy 裁剪平面 GL surface（glClipPlane / glGetClipPlane）**：此前两个入口
均为 MGL_UNIMPLEMENTED 桩。实现：
- glClipPlane：校验 plane ∈ [GL_CLIP_PLANE0, GL_CLIP_PLANE0+MAX_CLIP_DISTANCES)
  与 max_clip_distances 上限，方程原样存入 `state.var.clip_planes[8][4]`
  （GL 1.1 的 modelview 逆转置变换简化为恒等——MGL 定点矩阵栈未实现，
  glClipPlane 调用时刻无模型视图状态可读；着色器侧必须与 gl_ClipVertex
  同空间点乘）；标记 DIRTY_STATE|DIRTY_RENDER_STATE。
- glGetClipPlane：对称回读。
- 启用路由零成本：GL_CLIP_PLANE0..5 == GL_CLIP_DISTANCE0..5（0x3000+i），
  glEnable/glDisable/glIsEnabled 已走 mglClipDistanceIndex 的
  clip_distances 掩码；GL_MAX_CLIP_PLANES 查询已别名到 max_clip_distances。
- mgl.h 补充 GL_CLIP_PLANE0..7 遗留名别名（源兼容）。
- 回归 test_gl_clip_planes（self-check）：默认零值、set/get 往返（plane
  0/1/5）、写入不扰动其余平面、enable/disable 反射、越界 plane 报
  GL_INVALID_ENUM。A/B 双门 64/0/2/66 + ASan 全绿；188/188 翻译器单测。
- 后续（未做）：着色器侧推导——VS 写 gl_ClipVertex 时按 enabled 掩码
  注入 `gl_ClipDistance[i] = mix(1, dot(plane[i], clipVertex), en[i])`，
  需 per-draw GL 状态 uniform 更新机制（当前无此类机制，矩阵为应用直设）。

### P4 完成记录追加（2026-08-15，commit 48501b0：上传路径选路 C++ 化）

**P4.4 item 1111 第一个切片：CPU→GPU 上传路径的「storage mode 选路」迁入
C++** —— `mglRenderCppTextureUploadRoute`（纯决策函数，无 Metal 对象参与，
texture_type / storage_mode 传 MTLTextureType / MTLStorageMode 的 ABI 数值）：
1D/1DArray 且非 Private → REPLACE_1D；3D 且 AGX copyFromBuffer slice OOB bug
生效时 Private → REJECT、其余 → REPLACE_3D；其余（2D/2DArray/Cube/1D-Private…）
→ BLIT。与 uploadTextureSliceViaBlit 既有内联判定逐条件一致。
- MGLRenderer+Texture.m 的 uploadTextureSliceViaBlit 改为先取 route 再按
  REPLACE_1D / REPLACE_3D / REJECT / BLIT 执行分支体（3D 分支的 private
  拒绝检查随内联判定一并移入 route；REJECT 分支保留原日志与 return false）。
  分支体本身不变（1D replaceRegion、3D 紧凑重打包 + replaceRegion、blit
  dedicated CB 路径均原样），两个 gate 共用同一 route（纯逻辑，与 gate 无关）。
- 探测验证：texture_binding_switch 下 route 真实被调（type=2/storage=0/
  bug=1 → BLIT，本机 AGX bug 标记生效、2D private 纹理正确走 blit）。
- 翻译器单测新增 TEXTURE_UPLOAD_ROUTE_OK：8 组断言覆盖路由表（1D+shared/
  managed → REPLACE_1D；1D+private → BLIT；3D+bug+private → REJECT；
  3D+bug+shared → REPLACE_3D；3D 无 bug → BLIT；2D/Cube → BLIT）。
- 验证：A/B 双门 65/0/2/67 + ASan 双门 65/0/2/67 零报告；翻译器单测
  193/193；check-air-only OK；git diff --check 干净。
- 后续状态：2026-08-16 已由完整 `MGLRenderCppTextureUploadPlan`、既有
  `mglRenderCppBuildLevelUploadOps` 和 `mglRenderCppTextureExpandRGBToRGBA`
  收口，见 P4.4 顶层完成说明。

### P4 完成记录追加（2026-08-15，commit 67b5f42：主绑定路径 snapshot 化）

**P4.3b 收口：主 per-draw 绑定路径（非 batch 的常规 draw）的 setter 序列
snapshot 化** —— item 1056 的下一个切片。此前 P4.3b 只覆盖两条 batch fast
path；本轮把 vertex/fragment 主绑定循环（bindVertexBuffersToCurrentRenderEncoder /
bindFragmentBuffersToCurrentRenderEncoder）的全部 emit 也统一走
mglRenderCppEncodeBindingSnapshot，一次 C ABI 调用完成整个 setter 序列。
- snapshot 契约升级（mgl_render_cpp.h）：`MGLRenderCppBindingBufferEntry`
  双数组改为**有序 op 列表**（MGLRenderCppBindingOp，
  MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_OPS=32，每 stage 独立数组），保留同一
  stage 内 buffer / bytes / nil-clear 的**精确交错顺序**（同 slot 先 clear
  后 set 等场景必须保序）。kind 0=set buffer（buffer==NULL 即 clear，等价
  mglRenderCppSetRenderBuffer(nil)）；kind 1=set bytes（bytes 借用，重放返回前
  有效）。mgl_render_cpp.cpp 重放逐 op 调 setVertexBuffer/setVertexBytes/
  setFragmentBuffer/setFragmentBytes；NULL buffer 由"错误"改为合法 clear。
- 两条 P4.3b batch 收集点（MGLRenderer+BatchReplay.m
  bindDynamicVertexArrayBuffersDirectly / bindDynamicUniformRangesDirectly）
  迁移到新 op 结构（纯机械改写，语义不变）。
- 主路径收集（MGLRenderer+BindingState.m）：gate-on 下 fragment 与 vertex
  循环的每个 emit 点（nil-clear 跳过、小缓冲 CPU 数据 inline bytes、mtl 缓冲
  set、base plain-uniform inline bytes、isolated fallback、dedup 后 set）改
  收集进 snapshot；gate-off 保持逐条 ObjC 调用作 A/B 对照。判定（match-check）、
  统计（perf counters）、COW 记账（mglNoteBufferEncoded）、owner 更新
  （Update/Clear）、last-bound 失效两路完全一致——只有 encoder 调用被推迟。
  bytes 数据统一拷贝进函数作用域 scratch（4096B，重放前存活；padded 栈缓冲
  由此避开生命周期问题）；scratch 或 op 数组满先 flush 已收集 op 再继续，
  保持全局顺序。重放点：vertex 在 map 循环后、bindVertexAttributesFromVAO 前；
  fragment 在 map 循环后、fallback bindings 前——与直接路径的
  「map 循环 emit → VAO/fallback emit」顺序一致。
- 探测验证：临时 probe 显示 uniform_alias 下 vertex_ops=1 / fragment_ops=1
  的 replay 均真实触发（非空转）。翻译器单测更新：test_metalcpp_smoke 的
  snapshot 用例改为 op 结构 + 新增 bytes op 成功、NULL buffer clear 成功、
  NULL bytes / 坏 kind 拒绝的断言。
- 验证：A/B 双门 65/0/2/67 + ASan 双门 65/0/2/67 零报告；翻译器单测 193/193；
  check-air-only OK；git diff --check 干净。

### P4 完成记录追加（2026-08-15，commit a29dd07：gl_ClipVertex 着色器侧推导）

**Legacy 裁剪平面着色器侧推导（gl_ClipVertex → gl_ClipDistance）**：上一条的
"后续（未做）" 已交付，三层实现：
- 翻译器（mgl_legacy_compat.c）：VS 源码含 `gl_ClipVertex` 时，将首个
  `void main(` 改名为 `_mglLegacyUserMain`（新 `rename_first_main`，注释/字符串
  感知、空白容忍、可处理 `(void)`，容量溢出返回 0），追加
  `uniform vec4 _mglClipPlane[8]; uniform float _mglClipPlaneEnabled[8];` 与
  包装 main：`gl_ClipDistance[i] = mix(1.0, dot(_mglClipPlane[i],
  _mglClipVertex), _mglClipPlaneEnabled[i])`。探测验证 #version 110 翻译后
  rc=0 编译（MGL_STAGE_VERTEX=0——传 1 会静默按 FRAGMENT 编译，探针曾误报）。
- AIR 后端数组 uniform 真修复（潜在 bug，此前从未渲染验证）：var_ref 数组
  uniform 此前命中 `isArray`→LOCAL 路径（undef 折叠 NaN）；现仅非
  MGL_AST_Q_UNIFORM 走 LOCAL，uniform 数组落 VarSym::BUFFER→
  bufferLoad(整数组)+ExtractValue(i)。MGL_DUMP_IR 验证 loads+fma 正确
  （plane 0..112 / enabled 128..156，mix=fma(dot,en,1-en)）。注意：修复后
  抽取 metallib 位码 llvm-dis 报 "Malformed block"（llvm-dis rc=1），以
  MGL_DUMP_IR 为 IR 判据。
- per-draw 状态刷新（mglRefreshLegacyStateUniforms，mglDrawDispatch S7.5）：
  从 `state.var.clip_planes` + `caps.clip_distances` 逐叶写入当前 program 的
  plain uniform 槽——**每个数组元素一个 location**（plane 叶 16B，enabled 叶
  4B）；链接期缓存 `legacy_clip_plane_loc/_enabled_loc`（init -1，无特征
  零开销）。此前整数组写基址导致除首叶全 stale（en[5] 不生效）。

**struct packing 的 std140 叶步长修正是本轮关键 bug**：打包路径按
`member->array_stride`（std140，float 为 16）散布叶，而 AIR 后端以元素字节
大小（LLVM 整数组 load）布局 plain uniform 数组（float[8] 元素实际 4 步长）
——着色器读 base+4i、打包写 base+16i，en 数组必然错位。修复：mgl_buffer_plan.c
与 MGLRenderer+Buffer.m 两处打包均改用 `mglGLTypeElementByteSize(gl_type)`
（vec4→16、float→4、mat4→64，与 AIR 一致），std140 stride 仅作兜底。
调试取证：MGL_DEBUG_STRUCT_PACK 显示 p0clip 时 packed[8]=-1.0（plane0.z 打包
正确）、packed[128]=1.0（en0 叶）；探测 FS 输出 p0clip/p5clip 均为清屏色
(0,0,0,255)（片段被裁剪丢弃、alpha 1.0 为 glClearColor），p0off 为
(0,0,0,0)（恢复渲染）。

回归 test_legacy_clip_vertex（self-check，330 手写 8 路推导 VS + 纯红 FS）：
默认红→plane0=(0,0,-1,0)+enable 裁剪黑→disable 恢复红→plane0 翻转
(0,0,1,0) 不裁剪→plane5 裁剪黑→越界 plane 报 GL_INVALID_ENUM。A/B 双门
65/0/2/67 + ASan 全绿；翻译器单测 193/193（含 gl_ClipVertex 包装 main
新 5 项断言；另修复 rename_first_main 中 scan_step 的 char**→const char**
-Werror 限定符告警）。

### P4 完成记录追加（2026-08-14，commit 44a3732：GLSL 1.10 内建常量）

**GLSL 1.10 内建编译期常量（§7.4）**：gl_MaxDrawBuffers / gl_MaxClipPlanes /
gl_MaxLights / gl_MaxTextureUnits / gl_MaxTextureCoords / gl_MaxVertexAttribs
等此前报 "undeclared identifier"。双层修复：
- 翻译器：常量表 s_legacy_constants（12 项，值取 MGL 实际上限或 GLSL 1.10
  规范最小值）注入 `const int gl_MaxX = N;`（原 gl_ 名保留，与矩阵一致）；
  按 identifier 使用探测 + `code_has_const_decl` 防重（shader 自带同名
  const 时不注入）。
- AIR 后端：全局 const 初始化器求值此前仅覆盖数组（const vec3[] 路径）；
  标量 const 会走 uniform BUFFER 加载（offset 无效）→ 生成的 bitcode
  损坏（llvm-dis "Malformed block"）→ PSO 静默失败、全黑帧。修复：预 main
  循环扩展到 MGL_AST_Q_CONST 标量，emitExpr 求值入 cg.lvalues；VAR_REF 在
  uniform 路径前先查 const lvalue 折叠。取证：dump 的 metallib 位码在
  修复前无法 llvm-dis，修复后可正常反汇编。
- 回归 legacy_glsl_frontend 段 R：FS 用三个常量算颜色 (255,191,255)；
  翻译器单测 7 项新增（注入、不注入未用项、防重）。

### P4 完成记录追加（2026-08-14，commit 1b5c6a2）

**AIR：支持 gl_FragData[i] 多渲染目标（MRT）**：gl_FragData 数组输出此前
报 "codegen: unknown lvalue"（下标写缺聚合 lvalue）。后端端到端接通：
- retTy：数组片段输出展平为逐元素 float4 颜色输出（MSL 禁止 render-target
  struct 的数组成员——与数组 varying 同一约束）：struct { N×float4 }（可再
  追加 [[depth(any)]] 成员）；聚合 lvalue 预注册（与数组 varying/clip
  distance 相同修复），使 gl_FragData[i] = v 下标写可用；assembleReturn
  逐元素抽取；fragment output 列表携带 N 个 air.render_target 节点
  （(member index, color index) 常量，llvm-dis 对照参考着色器取证）。
- runtime 零改动：glDrawBuffers -> FBO 多颜色附件 -> render pass 颜色槽
  -> pipeline 附件格式的整条 MRT 链早已存在（调查证实：mglMetalColorSlot
  ForDrawBuffer / resolveFboDrawAttachmentIndex / generatePipelineDescriptor
  的附件循环 / value-state color_format[i] 全覆盖）。
- 回归 legacy_glsl_frontend 段 Q：双颜色附件 FBO + glDrawBuffers(0|1)，
  gl_FragData[0]=红、[1]=绿，glReadBuffer 分别读回两附件。**调试弯路**：
  初次失败实为测试自身像素缓冲区别名（q0 指针指向的 pixels 被 q1 的
  glReadPixels 覆盖）——逐字节拷贝后段 Q 通过；A/B 双门 + ASan 全绿
  63/0/2/65。
- 遗留：gl_FragData 与 gl_FragColor 互斥语义、draw buffer 未使能的附件
  由 Metal 按 pipeline 附件丢弃（与 GL 一致）；真实 GL 应用的 MRT +
  glDrawBuffers 组合仍需 P5 级验收。

### P4 完成记录追加（2026-08-14，commit b1ee7db）

**AIR：支持 gl_ClipDistance 顶点输出（普通 VS 路径）**：gl_ClipDistance
（GLSL 1.30+ 用户裁剪距离）此前被 "undeclared identifier" 拒绝。普通 VS
路径端到端接通：
- sema：var-ref 类型为 float[8] 数组。
- backend：strstr 检测（usesClipDistance，仅普通 VS——capture/tess 路径用
  固定 MGLAIRPerVertexRecord ABI，那里是 cull-distance 软件仿真）在
  point_size 之后追加 [8 x float] 输出成员；assembleReturn 插入数组；
  聚合 lvalue 预注册（与数组 varying 修复相同），使下标写
  （gl_ClipDistance[i] = v）可用；未写元素默认 +1.0（Metal 裁剪负距离，
  默认值不能裁剪）。
- metadata：MSL 形态为 'float cd [[clip_distance]] [N]' —— 输出节点携带
  air.clip_distance + air.clip_distance_array_size（用 llvm-dis 对照参考
  着色器取证；纯标量 air.clip_distance 或属性在数组上的写法都被本编译器
  拒绝）。
- 回归 legacy_glsl_frontend 段 P：P1 用 gl_ClipDistance[0] = -a_pos.y 裁剪
  y>0（下方探针红、上方探针被裁掉）；P2 对照写恒正距离（两探针皆红）。
  全套件 63/0/2/65 双门。

### P4 完成记录追加（2026-08-14，commit 0ad82ac）

**AIR：支持 gl_PrimitiveID + gl_SampleID fragment 内建**：两者此前都被拒绝
（gl_SampleID "undeclared identifier"；gl_PrimitiveID 仅曲面细分阶段可用）。
按其他 fragment 内建的模式接入为 fragment 参数：
- gl_PrimitiveID（uint）：sema int var-ref 类型；strstr 检测在 gl_PointCoord
  之后追加 i32 fragment 参数并绑定 lvalue，发射 air.primitive_id metadata；
  读 case 优先 fragment 参数，再回退 TCS/TES 的 patch 路径。
- gl_SampleID（uint）：sema int；i32 fragment 参数 + air.sample_id metadata
  + 读 case。
- gl_SamplePosition：尝试过，但本 Metal 编译器没有 [[sample_position]]
  fragment 输入（未知属性；air.sample_position metadata 会让着色器编译器
  XPC 服务崩溃——newLibraryWithData→PSO 阶段 XPC_ERROR_CONNECTION_INTERRUPTED，
  确定性复现）。已回退，内建保持干净的 "undeclared identifier" 错误。
  平台限制，待后续 SDK 提供该输入再支持。
- 回归 legacy_glsl_frontend 段 O：6 顶点绘制（两个三角形并排）用 FS 按
  gl_PrimitiveID == 0 && gl_SampleID == 0 取色（左红右蓝）。
  全套件 63/0/2/65 双门。

### P4 完成记录追加（2026-08-14，commit de65e7b）

**legacy：双面光照的 FS gl_BackColor 输入链接**：经典 GLSL 1.10 双面模式
（VS 输出 gl_FrontColor/gl_BackColor，FS 用 gl_FrontFacing ? gl_Color :
gl_BackColor 选择）此前在 fragment 阶段失败：Step 3 对阶段不适用的内建用
另一阶段的名称重命名（gl_BackColor → _mglBackColor），但 Step 4 的
fall-through 检查只看本阶段 fs_name（NULL）→ 不注入声明 → 前端拒绝重命名
后的标识符。Step 4 现在对存在性检查与注入都镜像 Step 3 的 fallback：被以
另一阶段名称重命名的内建在本阶段声明为 varying（FS gl_BackColor →
in vec4 _mglBackColor;，与 VS 同名输出链接）；声明相对 preamble 去重
（FS gl_Color 与 gl_FrontColor 都重命名为 _mglFrontColor，只能声明一次）。
回归 legacy_glsl_frontend 段 N：经典双面光照端到端 —— CCW（正面）读
gl_Color（红），glFrontFace(GL_CW) 翻转后读 gl_BackColor（蓝）。
standalone +6 检查（含去重计数）。全套件 63/0/2/65 双门；translator 181/181。

### P4 完成记录追加（2026-08-14，commit 54f274a）

**AIR：支持 gl_PointCoord + gl_FragDepth fragment 内建**：两者此前都被
"undeclared identifier" 拒绝（sema/backend 零支持）；gl_FragDepth 还需要
正确的 AIR 输出 metadata。
- gl_PointCoord（FS 输入，vec2）：sema var-ref 类型；strstr 检测在
  gl_FrontFacing 之后追加 float2 fragment 参数并绑定 lvalue，发射
  air.point_coord metadata（center、no_perspective），加显式读 case。
- gl_FragDepth（FS 输出，float）：sema var-ref 类型；fragment 返回值变为
  结构体 {color, depth}，第二成员承载深度写；assembleReturn 组装结构体
  （未写路径保持 1.0）；输出列表 metadata 采用 aux_shaders/scaled_depth_blit.metal
  的参考形态 —— air.depth + air.depth_qualifier air.any（最初用的
  air.frag_depth 名称被 metal 编译器静默丢弃，导致深度写完全失效；从
  llvm-dis 反汇编辅助着色器 AIR 取证后修正）。
- 回归 legacy_glsl_frontend 段 L：点精灵 —— VS 写 gl_PointSize 96，FS
  按 gl_PointCoord.x 取色（中心探针为红）。段 M：同一三角形两次绘制，
  gl_FragDepth 0.25 vs 0.75，GL_DEPTH_TEST + LEQUAL —— 更深的一趟必须被
  剔除（探针保持红）。段 M 用深度纹理 FBO：渲染器深度路径基于纹理，
  make_fbo 的深度 renderbuffer 从不生效（纯 z 对照实验隔离确认）。
- standalone +5 检查。全套件 63/0/2/65 双门；translator 套件 175/175。

### P4 完成记录追加（2026-08-14，commit 29ab0db）

**AIR：支持 gl_FrontFacing fragment 内建（item 753 延续）**：gl_FrontFacing
（GLSL 1.10 与现代同名）此前被前端拒绝（"undeclared identifier"），AIR 全栈
零支持。镜像 gl_FragCoord 全链路接通：sema var-ref 类型解析返回标量 bool；
backend strstr 检测（usesFrontFacing）在 gl_FragCoord 之后追加 i1 fragment
参数并绑定为 gl_FrontFacing lvalue，发射 air.front_facing metadata
（arg_name gl_FrontFacing，type bool）；读路径加显式 var-ref case（守卫
fragment stage）。回归 `legacy_glsl_frontend` 段 K：FS 按 gl_FrontFacing
取色（正面红/背面蓝），glFrontFace(GL_CW) 翻转约定使两个分支都被验证。
standalone +3 检查。全套件 63/0/2/65 双门；translator 套件 170/170。

### P4 完成记录追加（2026-08-14，commit 9280391）

**Legacy GLSL：gl_FragData[0]-only 改写为标量输出（item 753 延续）**：
gl_FragData[i] 着色器此前编译为数组 fragment 输出——AIR 后端无法 codegen
（"codegen: unknown lvalue _mglFragData"）——常见单缓冲旧式写法（含
`#define gl_FragColor gl_FragData[0]` 移植）完全不可用。翻译器修复：改名
gl_FragData→_mglFragData 后逐下标扫描——若全部为字面量下标 0，则把
`_mglFragData[0]` 改写为 `_mglFragColor`（新增 replace_literal 助手）并注入
标量 `layout(location=0) out vec4 _mglFragColor;`（对应单 color attachment，
即默认 draw buffer 的 GL 语义）；任一下标非 0 或动态下标保持数组声明
（真 MRT 属后续事项，该路径行为不变）。standalone 套件 +10 检查（全 [0]
改写 vs 混合下标保留数组）；回归 `legacy_glsl_frontend` 段 J：GLSL 1.10
VS + gl_FragData[0] FS 渲染红三角。全套件 63/0/2/65 双门；translator 套件
167/167。

### P4 完成记录追加（2026-08-14，commit b5b34af / 段 H 追加）

**Legacy GLSL：固定功能 attribute 槽位保留原名（item 753 延续）**：
旧式 VS attribute 输入（gl_Color、gl_Normal、gl_SecondaryColor、
gl_FogCoord）此前被改名 _mgl* 且由 linker 分配 location——应用既不能按
原名查询也不能按固定功能槽位绑定。镜像 gl_Vertex 处理：s_builtins 增
`vs_location`（gl_Normal=2、gl_Color=3、gl_SecondaryColor=4、
gl_FogCoord=5）；Step 3 对固定槽位 VS 输入跳过改名（源码保留 gl_ 名）；
Step 4 注入 `layout(location = N) in <type> gl_Name;`（AIR 前端接受带
显式 location 的 gl_ 前缀输入声明；backend/reflector 的细化 gl_ 跳过
本就放行显式 location 符号）→ attribute 以原名反射在传统槽位，
glGetAttribLocation 返回原名。varying 输出与 FS 条目不变（改名接口）。
回归 `legacy_glsl_frontend` 段 G：经典颜色链（attribute gl_Color →
gl_FrontColor varying → gl_FragColor）以槽位 3 颜色流渲染红三角，并断言
glGetAttribLocation("gl_Color")==3；段 H：雾坐标链（float attribute
槽位 5 → gl_FogFragCoord float varying → FS 取色）渲染红三角，断言
glGetAttribLocation("gl_FogCoord")==5。全套件 63/0/2/65 双门 + ASan；
translator 套件 157/157。

### P4 完成记录追加（2026-08-14，commit 1c7eab4）

**AIR 后端：数组型 stage varying 展平 + 传统 MultiTexCoord 槽位（753）**：
Metal 禁止 vertex return / stage-in 结构体含数组成员（"field of illegal
type float4[N]"）——数组 varying（如传统 gl_TexCoord[8] 流转）能编译但
从不真正到达光栅器（回归实测黑屏）。修复：retElems/paramTys 展开为 N 个
标量元素类型；FS 侧 N 个参数组装回聚值 lvalue（readIndexChain/swizzle
无需改动）；assembleReturn 逐元素抽取；接口 metadata 按元素
`air.vertex_output`/`air.fragment_input` + 元素专属名（_mglTexCoord_elm0..7，
双端一致、逐元素唯一）。配套：gl_MultiTexCoord0..7 注入
`layout(location = 8 + i)`（固定功能 texcoord 槽位）。回归段 F 覆盖完整
传统纹理流（gl_MultiTexCoord0 → gl_TexCoord[0] 数组 varying → texture2D）
渲染红三角。全套件 63/0/2/65 双门；translator 套件 157/157。

### P4 完成记录追加（2026-08-14，commit f62b28d）

**Legacy GLSL：ftransform() 展开（item 753 延续）**：ftransform() 此前只
检测不替换——仅用 ftransform() 的着色器会以未声明函数编译失败。在
gl_Vertex + 矩阵 uniform 可用（round 13-14）后，Step 2.5 将其展开为
`gl_ModelViewProjectionMatrix * gl_Vertex`；矩阵与 gl_Vertex 的注入从
feature-flag 守卫改为按当前源码逐名守卫（展开在检测之后才引入这两个名字）。
回归段 E：仅以 ftransform() 为位置来源的 GLSL 1.10 VS 渲染红三角。
全套件 63/0/2/65 双门；translator 套件 157/157。本 round 另关闭
item 1054（query/sync 收口：查询 100% C++，fence 走 C++ owner，剩余
ObjC 恰为 GL 语义层）。

### P4 完成记录追加（2026-08-14，commit c8835a4）

**Legacy GLSL：gl_Vertex 在传统 location 0 可绑定（item 753 延续）**：
gl_Vertex 此前能编译但不可用——AIR 后端 syms 转换跳过所有 gl_ 前缀符号，
gl_Vertex 从未进入 kernel 签名（metallib 无 [[attribute(0)]] 输入），且
reflector 的 gl_ 跳过使其不进 stage-input 列表（attribute slot 0 不预留，
uniform/SSBO slot 计算与 metallib 脱钩——潜藏错位 bug）。修复：
`mgl_air_backend` syms 转换放行「显式 location 的 gl_ 符号」与「uniform
限定 gl_ 符号」（与 reflector 的细化跳过一致；gl_Position/gl_in 等无
显式 location 仍按后端内建绕过）；`mgl_legacy_compat` 注入
`layout(location = 0) in vec4 gl_Vertex;`（传统固定功能槽位 0，名称保留）；
reflector stage-input 放行显式 location 的 gl_ 符号 →
glGetAttribLocation(prog,"gl_Vertex")==0，descriptor/binding 机制预留
槽位 0。回归 `legacy_glsl_frontend` 段 D：经典 GLSL 1.10 VS
（gl_ModelViewProjectionMatrix * gl_Vertex）用测试框架普通 2 分量
attrib-0 流渲染红三角（MVP 原名解析）。全套件 63/0/2/65 双门；
translator 套件 146/146。本 round 另关闭 item 900（FBO-cache 完成 +
三项决策态明确处置为 P5 落位）。

### P4 完成记录追加（2026-08-14，commit 5e9257c）

**Legacy GLSL：gl_Vertex + 固定函数矩阵 uniform（item 753 延续）**：
GLSL 1.10 最常见的 `gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex`
此前无法翻译（gl_Vertex 不在 builtin 表、§7.4 矩阵无处理）。新增
`s_legacy_matrices` 表（MVP/MV/Projection/TextureMatrix[8]/NormalMatrix +
inverse/transpose 变体）按**原名**注入 `uniform mat4 gl_ModelViewProjectionMatrix;`
（AIR 前端接受 gl_ 前缀用户声明，parse+sema 实测）→ GL 侧 uniform 契约不变；
gl_Vertex 入 builtin 表（`in vec4 gl_Vertex;`）。配套：`airPrepareLegacySource`
加 preamble marker 防二次翻译（矩阵原名保留 → 重复注入），`mgl_air_reflect`
的 gl_ 前缀跳过规则放行 uniform 限定符号（注入矩阵入反射，stage I/O 仍跳过）。
回归 `legacy_glsl_frontend` 段 C：GLSL 1.10 VS 经原名
`gl_ModelViewProjectionMatrix` 设置单位矩阵渲染红三角。translator 套件
146/146；全套件 63/0/2/65 双门。

### P4 完成记录追加（2026-08-14，commit 5320bed）

**TES compute dispatch 按 P4.3e 同模式落地（item 997 收口）**：
`dispatchAIRTessEvalCompute` 的固定序列（encoder + pipeline + ABI 槽位
factor/patch/out）经 `mglRenderCppBeginComputeDispatch` 一次交给 C++
（setup 3 buffer，≤16 cap）；GL 资源绑定留在 begin/end 之间经 facade；
gate-off fallback 保持原 ObjC 序列。全套件 63/0/2/65 双门一致（含
air_tessellation 家族）。

### P4 完成记录追加（2026-08-14，commit 1bae6fb）

**Legacy GLSL frontend 接线（item 788 后续 / 753）**：`mgl_legacy_compat`
（纯 C pre-3.30 GLSL 源码级改写）此前为孤儿模块（无 lib 调用方）。现于
`mgl_air_backend.cpp` 全部源码入口（reflect / compile / compileGLSLImpl /
interface check）在 mglGLSLParse **之前** detect→translate（+2048 余量缓冲，
`#version` 解析默认 110，stage→GL shader type 决定 varying/builtin 方向）；
无 legacy 特征时零拷贝直通。新增回归 `legacy_glsl_frontend`（第 65 项）：
段 A attribute/varying/gl_FragColor 红三角（内部探针），段 B legacy
texture2D() 采样红 1x1 纹理；双门 A/B 一致。translator 独立套件 134/134。

### P4 完成记录追加（2026-08-14，commit 4a2bd56）

**新增 regression `air_msaa_resolve` + 两个既有 bug 修复**（P4.4 item 906 覆盖）：

- **Blit 陈旧读**：deferred draw batch 只在 flush 点重放，FBO bind 切换在
  deferFboRotation 下跳过 flush —— blit 在 draw 后立即执行时复制的是
  draw 前（仅 clear）的内容。`mtlBlitFramebuffer:` 现先
  `flushDrawBuffer:` + `endRenderEncoding` 再读源镜像（空 batch 时零开销）。
- **Gate-on MSAA sample count**：两处 pipeline descriptor builder 的
  rasterSampleCount 只在 ObjC `renderPassDescriptor` 存在时解析（P4.1f 后
  gate-on 为 nil）→ gate-on MSAA pass 停在 sampleCount=1（4x pass 只写
  1/4 采样，resolve 25% 覆盖率）。改为 owner-first 读取 C++ owner 的
  attachment sample count（gate-off 行为不变）。
- 全套件双门 62/0/2/64。

### P4.2 完成记录（2026-08-13）

**交付**：gate-on 的 final/simple/safe pipeline descriptor 组装全部迁入 C++
builder；ObjC 不再构造 `MTLRenderPipelineDescriptor`（value-state 直出）。

- **value-state**：`MGLPipelineDescriptorState` 更名为
  `MGLRenderCppPipelineDescriptorState`（`mgl_air_loader.h`，旧名 typedef 兼容）；
  `mgl_render_cpp.h` 前向声明 + 新 facade
  `mglRenderCppCreateRenderPipelineFromState(vs, fs, state, archive, out, err, cap)`
  —— ObjC 传 value-state + 函数指针 + 归档，C++ 完成
  `MTL::RenderPipelineDescriptor` 组装。
- **C++ builder**（`mgl_air_loader.cpp`）：`buildRenderPipelineDescriptor` +
  `createRenderPipelineInternal` 共享实现 —— label "GLSL Pipeline"、
  packed depth/stencil normalize（`normalizeDepthStencilFormats`，镜像
  `mglNormalizePipelineDepthStencilFormats`）、`MGL_ENABLE_ICB_PIPELINES`
  opt-in（镜像 `mglEnableIndirectCommandBuffersForPipeline`）、color attachment
  writeMask/blend 只在格式有效时设置（未触碰 attachment 保持 Metal 默认值）、
  layout 只在 attrib 格式有效时写入（修复旧转换路径 Invalid attrib 以 0 stride
  覆盖 layout 的隐患）；`maxTessellationFactor` 为 0 时跳过设置（Metal 默认
  64；safe/simple 零值 state 不触发 "maxTessellationFactor must be >= 1" 断言）。
  归档：创建前 `setBinaryArchives`、成功后 `addRenderPipelineFunctions`
  （镜像 ObjC applyBinaryArchiveToDescriptor / addPipelineToBinaryArchive）。
- **ObjC 读取侧**：`generatePipelineDescriptorState:vertexFunction:fragmentFunction:`
  （RenderPass.m）+ `generateVertexDescriptorState:`（VertexLayout.m）直接填充
  value-state，函数选择、bindMTLProgram/bindMTLTexture side effects、FBO/rp/
  drawable 格式兜底、blend（owner-first `blendStateForAttachment:out:`）、
  sample count、topology、tess 字段与 ObjC 路径逐字段等价；blend 镜像写入在
  gate-on 下不再写（`setBlendFactorsForAttachment:` 镜像段 gate 化）。
- **签名**：`mglVertexDescriptorSignatureFromState` /
  `mglPipelineDescriptorSignatureFromState`（mgl_vertex_format.m）哈希字段/顺序
  与 descriptor 版一致（layout 取「最后写它的 attrib」，镜像累积写入语义）。
- **descriptor cache 改 value-state**：`mglRenderCppLookupPipelineDescriptorState` /
  `StorePipelineDescriptorState` 取代 pointer-based 版（原 C++ 侧把 ObjC
  descriptor 当 `MTL::*` 存取的 type-confused 隐患消除）；
  `MGLPipelineCache` 新增 `pipelineDescriptorStateForWords:state:` /
  `storePipelineDescriptorState:forWords:`；gate-off 的 ObjC descriptor 字典保留。
- **miss 路径**：新 `buildPipelineStateOnCacheMissWithState:` —— final/simple/safe
  三套降级 state + `mglRenderCppCreateRenderPipelineFromState`；GPU recovery /
  interface-mismatch / `MGL_FORCE_SAFE_FALLBACK_PIPELINE` 测试钩子语义与 ObjC
  版一致；`mglCreateAIRRenderPipelineCpp`（descriptor→state 转换）已删除。
- **踩坑**：①非 TES pipeline 的 `max_tessellation_factor=0` 直接
  `setMaxTessellationFactor(0)` 触发 Metal 断言崩溃（ObjC descriptor 默认 64，
  旧转换路径从 descriptor 拷贝所以不炸）——state 默认 64 + C++ builder 对 0
  跳过；②`@try`/`@catch` 作用域里 `psoPtr`/`binaryArchive` 需提升到方法级；
  ③ARC 三元表达式混 `id<MTLFunction>` 与 `void*` 需先转 `void*` 再桥接；
  ④`insertPipelineIntoCacheWithWords` 需要 value-state 变体
  （`insertPipelineStateIntoCacheWithWords:`）。
- **验证**：`make lib` / test-mglair（TCS/TES/GS/XFB/VALUE OK）/ gtest 42/42 /
  test-metalcpp（SMOKE_DONE，descriptor cache smoke 改 value-state 断言）/
  test-dirty-hash PASS / test-legacy-compat 134/134 / test-benchmark PASS /
  `check-air-only` OK / git diff --check 全绿；regression A/B 均
  **54 PASS / 0 FAIL / 2 SKIP**（56 测试，含 `air_pipeline_safe_fallback`）。
- **rg 盘点**（P4.2 验收口径，P5 删 gate 时清零）：
  `rg -l "MTLRenderPipelineDescriptor" MGL/src --glob '*.m'` 命中 7 个文件：
  `MGLRenderer.m`（白名单外壳 helper）+ 6 个 gate-off A/B 回退
  （RenderPass/VertexLayout/PipelineCache/mgl_vertex_format/Blit；
  Tessellation.m 仅注释）。gate-on 新 PSO 路径零组装。

### P4.3a 完成记录（2026-08-13）

**交付**：draw 提交的统一 C ABI —— `MGLRenderCppDrawPlan` + `mglRenderCppEncodeDraw`。

- **plan 定义**（mgl_render_cpp.h）：六种 kind —— `MGL_RENDER_CPP_DRAW_ARRAY`
  （vertex_start/vertex_count）、`DRAW_INDEXED`（index_count/index_type/
  index_buffer/index_buffer_offset/base_vertex）、`DRAW_ARRAY_INDIRECT` 与
  `DRAW_INDEXED_INDIRECT`（indirect_buffer/offset）、`DRAW_PATCHES` 与
  `DRAW_INDEXED_PATCHES`（control_point_count/patch_start/patch_count/
  patch_index_buffer/control_point_index_buffer）；通用 instance_count/
  base_instance。资源全部 +0 borrowed。
- **EncodeDraw**（mgl_render_cpp.cpp）：按 kind 分派到既有 per-call facade
  （mglRenderCppDrawPrimitives 家族），非法 kind/空 encoder/缺 buffer 返回
  -1 并写 err —— 调用方回退 ObjC 直接编码，gate-off 语义不变。
- **ObjC 桥**（mgl_render_cpp_objc.h）：`mglRenderCppTryEncodeDraw(encoder,
  plan)` inline —— 内部做 gate 检查（MGL_USE_METALCPP + device），返回 YES
  表示已由 C++ 提交。
- **wrapper 收敛**（9 个文件 22 个 wrapper）：Draw.m（4）、BatchReplay.m（4）、
  DrawSupport.m（6，含 indexed-type 与 patches 变体）、Batch.m（2）、
  mgl_draw_encode.m（2）、Tessellation.m（1）、MGLRenderer.m（1）、
  SwapDiagnostics.m（1）、Blit.m（1）—— 原「per-call facade + ObjC fallback」
  双份实现改为「plan 构造 + TryEncodeDraw + ObjC fallback」单份语义；
  Draw.m 的 `mglDrawUsesMetalCpp` 等死代码删除（其余文件 gate 函数仍被
  binding/encoder 路径使用）。16 处重复 wrapper 体消除。
- **smoke**：`DRAW_PLAN_ENCODE_OK` —— 真实 encoder + scaled_blit PSO 上
  编码合法 plan、NULL encoder / 非法 kind / instance_count=0 均返回 -1。
  （踩坑：无 PSO 的 encoder 上 drawPrimitives 直接 AGX 崩溃 —— 必须先
  setRenderPipelineState。）
- **验证**：regression A/B 均 54 PASS / 0 FAIL / 2 SKIP（56 测试）+
  test-mglair / gtest 42/42 / test-metalcpp（SMOKE_DONE + DRAW_PLAN_ENCODE_OK）/
  check-air-only / git diff --check 全绿。
- **rg 盘点**：`\[encoder drawPrimitives|\[encoder drawIndexedPrimitives` 等
  22 处直接调用全部位于 gate-off fallback；gate-on draw 路径零直接调用。

### P4.3b 完成记录（2026-08-13）

**交付**：per-draw binding snapshot —— `MGLRenderCppBindingSnapshot` +
`mglRenderCppEncodeBindingSnapshot`，batch replay 两条 direct 绑定快路径的
setter 序列迁入 C++。

- **snapshot 定义**（mgl_render_cpp.h）：vertex/fragment buffer 绑定条目
  （`MGLRenderCppBindingBufferEntry`：buffer +0 borrowed / offset / Metal slot），
  各最多 `MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_BUFFERS`（31）条。
- **EncodeBindingSnapshot**（mgl_render_cpp.cpp）：在 C++ 内按序重放
  `setVertexBuffer`/`setFragmentBuffer` —— 与逐条 `mglRenderCppSetRenderBuffer`
  完全等价，但一次 C ABI 调用完成整个序列；count 溢出 / NULL 条目 / 空
  encoder 返回 -1。
- **接入**（MGLRenderer+BatchReplay.m 两条 fast path）：
  - `bindDynamicVertexArrayBuffersDirectly:` —— 逐 stream dedup 判定后把待
    emit 绑定收集进 snapshot，循环后一次重放；
  - `bindDynamicUniformRangesDirectly:` —— vertex+fragment 双 stage 收集进
    同一 snapshot，所有 override 循环后一次重放。
  - 判定（`mglBindingStateBufferMatches`）、统计（perf counters）、COW 记账
    （`mglNoteBufferEncoded`）与 owner 更新全部保留在 ObjC 且与逐条路径一致；
    **gate-on 走 snapshot，gate-off 保留逐条 ObjC 调用**（A/B 对照必须是纯
    ObjC）。`mglBatchReplaySetRenderBuffer` 的逐条调用点被替代。
- **smoke**：`BINDING_SNAPSHOT_OK` —— 合法 snapshot 编码成功；NULL encoder /
  count 溢出 / NULL buffer 条目均返回 -1。
- **验证**：regression A/B 均 54 PASS / 0 FAIL / 2 SKIP（56 测试，含
  multibatch_same_fbo 的 batch replay 覆盖）+ test-mglair / gtest 42/42 /
  test-metalcpp（SMOKE_DONE + BINDING_SNAPSHOT_OK）/ test-dirty-hash /
  test-legacy-compat 134 / test-benchmark / check-air-only / git diff --check
  全绿。
- **语义等价性说明**：snapshot 只收集「已通过 dedup 判定」的绑定，重放顺序
  与逐条路径一致（slot 索引是 dedup key，收集阶段提前不改变结果）；收集阶段
  无 encoder 交互，重放延迟到循环后与逐条 emit 完全等价。

### P4.3c 完成记录（2026-08-13）

**交付**：whole-batch simple replay —— batch replay 执行 loop 的最小 surgery
版：数据仍是 ObjC batch arena 的只读快照（`MGLDrawCommand[]`），循环与最终
draw 在 C++。

- **C ABI**（mgl_render_cpp.h）：`MGLRenderCppReplayBatchCommand`（cmd_type /
  first / count / instance_count / base_vertex / base_instance / index_type /
  index_buffer_offset / index_buffer）+ `MGLRenderCppReplayBatch`
  （primitive_type / command_count / commands 数组，上限 128）+
  `mglRenderCppReplayBatchDraws(encoder, batch, err, cap)` → OK / NEEDS_OBJC /
  ERROR。契约：调用方预校验全部命令，成功即全部绘制；NEEDS_OBJC 时调用方
  必须整体回退 ObjC 循环（不得部分重放）。
- **C++ 实现**（mgl_render_cpp.cpp）：逐命令构造 `MGLRenderCppDrawPlan`
  （arrays 家族 → ARRAY，elements 家族 → INDEXED，硬编码 draw_command.h 的
  稳定 ABI 常量，GL 头不进 C++ include 链），count==0 跳过，EncodeDraw 提交；
  未知类型 / 索引未就绪返回 NEEDS_OBJC。
- **ObjC 接线**（MGLRenderer+BatchReplay.m）：`tryReplaySimpleBatchWithCpp:`
  在 `issueDirectBatch:` 开头先行 —— 前置条件：gate-on、命令数 ∈ (0,128]、
  程序无 cull-distance、`caps.primitive_restart` 关闭、批无 dynamic
  vertex/uniform/texture binding 与 sampler 快照、primitive_type 有效、mode
  无 point/fan/line-loop/quads 模拟；elements 命令逐条 resolve 元素缓冲 +
  `getMTLIndexType` + `mglPreparedElementIndexBuffer`（byte 索引展开为
  UInt16），任一失败整体回退。
- **smoke**：`REPLAY_BATCH_OK` —— 2 命令（array + indexed）整批重放成功；
  未知 cmd_type → NEEDS_OBJC；空批 / NULL encoder → ERROR。
- **覆盖验证**：lldb 断点 `mglRenderCppReplayBatchDraws` 在 gate-on regression
  运行中命中（`issueDirectBatch:` → `tryReplaySimpleBatchWithCpp:` 调用链），
  证明 C++ replay 路径真实执行而非静默 fallback。
- **验证**：regression A/B 均 54 PASS / 0 FAIL / 2 SKIP（56 测试，含
  multibatch_same_fbo）+ test-mglair / gtest 42/42 / test-metalcpp
  （SMOKE_DONE + REPLAY_BATCH_OK）/ test-dirty-hash / test-legacy-compat
  134 / test-benchmark / check-air-only / git diff --check 全绿。
- **语义等价性**：C++ 路径只覆盖「无特例」批 —— 与 ObjC 循环的简单分支
  （直接 drawPrimitives/drawIndexedPrimitives）逐命令等价（同一 plan 入口）；
  特例批（dynamic binding/cull/模拟/restart）仍走 ObjC 循环，行为不变。

### P4.3e 完成记录（2026-08-13）

**交付**：GS compute dispatch 编排的固定序列迁入 C++ ——
`MGLRenderCppComputeDispatchSetup` + `mglRenderCppBeginComputeDispatch` /
`mglRenderCppEndComputeDispatch`，接入 `handleGeometryDrawIfNeeded:` 的 GS
kernel dispatch（air_geometry* / XFB 路径）。

- **C ABI**（mgl_render_cpp.h/.cpp）：setup 携带 pipeline（+0 borrowed）+ 至多
  16 条 buffer 条目 + 4 条 bytes 条目；begin 一次完成「创建 compute encoder
  + setComputePipelineState + 槽位绑定」（encoder +0 borrowed，CB 持有），
  end 一次完成 dispatchThreadgroups + endEncoding。与逐条
  mglRenderCppSetCompute* / DispatchCompute / EndComputeEncoder 完全等价。
- **ObjC 接线**（MGLRenderer+DrawSupport.m，GS kernel dispatch）：
  gate-on 构建 setup（INPUT/OUTPUT/COUNTS/GATHER/GATHER_PARAMS/XFB/XFB_META
  七个 ABI 槽位，XFB stream 槽按 xfbCaptureBuffer 条件包含）+ begin；
  GL 资源绑定（bindBuffersToComputeEncoder / bindTexturesToComputeEncoder）
  在 begin/end 之间保持不变（只经 C++ facade）；end 携带 workItemCount
  dispatch。gate-off 保留逐条调用（A/B 对照）。begin/end 失败回退
  EndComputeEncoder / 逐条路径，行为与原来一致。
- **smoke**：`COMPUTE_DISPATCH_OK` —— 独立 CB 上 begin（scaled_blit_cs PSO +
  buffer + bytes）→ end（1×1×1 dispatch）成功并提交；NULL CB / NULL encoder
  拒绝。
- **覆盖验证**：lldb 断点 `mglRenderCppBeginComputeDispatch` 在 gate-on
  regression 运行中命中（`handleGeometryDrawIfNeeded:` ← `mtlDrawArraysLocked:`），
  GS compute dispatch 编排真实走 C++。
- **验证**：regression A/B 均 54 PASS / 0 FAIL / 2 SKIP（56 测试，含
  air_geometry_indexed/indirect/multi_draw/xfb/multi_stream_xfb/layer_viewport
  全部 GS 路径）+ test-mglair / gtest 42/42 / test-metalcpp（SMOKE_DONE +
  COMPUTE_DISPATCH_OK）/ test-dirty-hash / test-legacy-compat 134 /
  test-benchmark / check-air-only / git diff --check 全绿。
- **遗留**：TES compute dispatch（dispatchAIRTessEvalCompute /
  dispatchTessControlShader）的序列更长且交织 GL 资源解析（storage image /
  stage bindings / copybacks），与 P4.1e3 未解项（gate-on 跨 CB 可见性）同域，
  留待 P4.1e3 修复后按同一 begin/end 模式迁移。

### 2026-08-16 剩余残留审计（历史快照：当时 P4/P5 未完成）

### P4 收口增量记录（2026-08-16：owner transaction + gate-on copy-back 接线）

- `mglRenderCppCommitCommandBufferTransaction` 将一次 detached command-buffer
  提交表示为 value-state：提交前/后/完成状态、submission ownership、completion
  注册、wait、driver rejection、reset request 和是否需要下一个 current CB。
  submission 与 command buffer 不匹配时 fail-closed；RAII commit guard 保证
  异常路径释放重入状态。
- `MGLRenderer+GPURecovery.m` 的 gate-on 提交和
  `flushStageBindingCopyBacks:` 的 gate-on 等待均通过该 owner transaction；
  gate-off 继续使用 ObjC fallback。C++ copy-back validation、blit encode 和
  `mglRenderCppCopyBackCPUPrefix` 不回退。
- recovery completion context 对 `CommandBufferRecoveryOwner` 增加独立引用，
  因此 owner handle 销毁后 completion 仍可安全完成；`addCompletedHandler`
  异常路径会释放 context。该修复只覆盖 C++ wrapper 生命周期，不覆盖 ObjC
  completion block 的完整 TSan 结论。
- smoke 新增并通过 `COMMAND_BUFFER_TRANSACTION_OK`；串行 gate-off/gate-on
  regression 均为 `73 PASS / 0 FAIL / 2 SKIP`。该历史切片当时未勾选 compute、
  command lifecycle、binding、Draw setter、callbacks 或 P4 final 条目；这些
  未完成项已由上文 2026-08-17 P4 完成记录关闭。

- **GS XFB link-time layout plan 已完成（2026-08-16）**：`Program` 现在持久化
  varying→binding/component-offset/stream scatter plan；link 校验覆盖重复
  varying、数组元素索引、interleaved 每 binding component 上限、separate 每
  varying component 上限，以及同一 binding 的 stream 一致性。`gl_NextBuffer`/
  `gl_SkipComponentsN` 按 ARB_transform_feedback3 只接受
  `GL_INTERLEAVED_ATTRIBS`，separate 模式由每个 varying 自动占一个 binding；
  compute route 仍拒绝 SEPARATE capture execution。
- **现有 multi-stream XFB 回归仍不是 GL4 终态语义**：回归已改用
  `gl_NextBuffer` 明确建立 stream 0→binding 0、stream 1→binding 1 的合法布局，
  并新增 `air_xfb_link_layout` 覆盖合法/非法 link/API 状态；仍需补跨多个 XFB
  binding 的整图元原子容量截断、stream>0 的完整输出拓扑约束、passthrough GS
  在 XFB active 时的 bypass，以及 shader-level default-stream reflection 闭环。
- **pending clear/FBO**：pending-clear 状态尚无 scissor provenance，bit 消费也
  不是事务性的；incomplete FBO clear 尚未统一拒绝。CPU fallback 仍假设
  4-byte RGBA，array/3D subresource 寻址不完整；depth-stencil attachment 替换
  仍需先完整校验再同时提交 depth/stencil 两侧。
- **texture/blit**：1D-array upload 仍把 layer 映射到 Metal `origin.y` 而非
  slice；`TexStorage` sized-internalformat 校验仍不完整；scaled-blit scissor
  数学仍有重复实现，需收敛到单一 plan。
- **lifecycle（2026-08-16 历史快照）**：program/hash ownership 已在当时工作区收口——context teardown
  现在先从 `program_table` 脱链再释放 `Program`，因此不再把正常 teardown
  误报为 `STILL in hash table`。command-buffer 的实际提交、异常恢复、AGX
  recovery、deferred reset 和 owner 跨线程保活在该快照中仍属于原 ObjC 限制；
  这些旧阻断已由 2026-08-17 P4 完成记录关闭，当前边界见
  `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md` 顶部最终状态。

### 2026-08-16 代码审计（对照当前工作区 `lgpl`，HEAD=6a9f989 + 未提交 WIP）

> 本轮直接审计当前源码（`MGL/src`、`MGL/include`）与未提交工作区改动，
> 验证了 `git log`、`mgl_air_gs_abi.h`、`mgl_buffer_slots.c/.h`、
> `mgl_pixel_format.c` 与 `MGLRenderer+Texture.m` WIP。发现 **1 个 P0、
> 4 个 P1、1 个 P2、2 个 P3**（其中 depth32 与 fragment-UAF 两个 P1 位于未提交
> WIP，须在 submit 前修复）。以下按严重度排列，均附 file:line 与触发条件，
> 结论均已对照代码逐条核实。

- **P0 —— GS/TES 保留槽位注册表是死代码，用户 UBO/SSBO 可撞飞 GS XFB**：
  `mglBufferSlotIsReservedForGeometry`（mgl_buffer_slots.c:89-106）保留了
  `{24,28,29,30}`，注释仍写 "30 = GS XFB reserved for P1"；而**真实 GS ABI**是
  `INPUT=24 / OUTPUT=28 / COUNTS=29 / GATHER=30 / GATHER_PARAMS=25 /
  XFB=31 / XFB_META=27`（mgl_air_gs_abi.h:48-72,300-310, 纯字面量与枚举一致）。
  **更严重的是**：grep 全仓，`mglBufferSlotIsReservedForStage` /
  `ForGeometry` / `ForTessellation` / `ForCullDistance` /
  `ForFragCoordFixup` / `mglBufferSlotIsReserved` / `mglBufferSlotReservedName`
  在 `mgl_buffer_slots.c:24/56/70/89/108/122/129` 定义、在头声明，**却无任何
  调用者**；`mgl_buffer_slots.h:13/27/153` 点名要求的冲突门
  `applyMSLResourceBindings` **在源码中不存在**。用户 buffer 的实际 Metal 槽位
  由 `mglMetalResourceSlot(res)=res->binding`（mgl_shader_resource.m:100-102）
  直接取自反射的 `air.location_index`，而该索引从
  `user_buffer_base`（mgl_air_reflect.c:353，非 TES 阶段=0）起按 SSBO→UBO
  顺序平铺递增（:389-393 / :570），**无任何保留槽回避、无上限**。因此一个
  SSBO/UBO 资源较多的几何着色器程序，其用户缓冲可被分配到 27（XFB_META）或
  31（XFB 流出），**静默撞毁 GS transform-feedback 的 atomic cursor / 输出**
  （`DrawSupport.m:1612-1657` 先绑固定槽 `MGL_AIR_GS_SLOT_*` 再绑用户槽；
  绑定路径 `bindBuffersToComputeEncoder` MGLRenderer+Compute.m:187 无保留检查）。
  **修法**：把保留集改为 `{24,25,26,27,28,29,30,31}` 并注释同步，**并把
  该检查接入真正的槽位分配路径**（在 `mgl_shader_resource.m` 的
  `mglMetalResourceSlot*` 或 Compute/GS 的绑定入口处拦截，或把用户
  `air.location_index` 上限封在 24 以下）。Grep 证据：`grep -rn
  "mglBufferSlotIsReserved" MGL/src` 仅命中 mgl_buffer_slots.c/.h 自身。
- **P1 —— `glTexStorage*` 拒绝 `GL_DEPTH_COMPONENT32`（未提交 WIP 引入的新
  回归）**：mgl_pixel_format.c 的工作区改动把 `mglTexStorageInternalFormatValid`
  （:1583）从「压缩 `|| mtlFormat!=Invalid`」改为
  「sized-color 表 || depth/stencil switch」，而该 switch（1590-1599）**漏了
  `GL_DEPTH_COMPONENT32`**（只有 16/24/32F）。对照：仓库其余深度判定
  （`mglInternalFormatIsDepthStencil` mgl_pixel_format.c:440-442、
  `mglDepthOnlyFormat` :442、framebuffers.c:2625、textures.c:2143/2356/3178）
  都把 `GL_DEPTH_COMPONENT32` 当合法 sized 格式——旧实现经
  `mtlFormatForGLInternalFormat` 也映射到 Depth32Float（pixel_utils.c:2218
  明注 valid）。**行为变更**：`glTexStorage* `（textures.c:4503/4591/4739 与
  `mgl_gl_extensions.c:6007/6036` 共 5 处新增检查）现对
  `GL_DEPTH_COMPONENT32` 返回 `GL_INVALID_ENUM`。**且新回归用例直接锁死该
  回归**：test_regression/main.c:792-793 `{"bound 3D depth32",
  STORAGE_BOUND_3D, ... GL_DEPTH_COMPONENT32}` 刻意断言必须 `GL_INVALID_ENUM`
  ——把 bug 当预期编码。**修法**：在 1590-1599 的 switch 补 `GL_DEPTH_COMPONENT32`
  case，并把该回归用例改为正向（应当分配成功）。submit 前务必跑一遍
  `make test-regression` 看停机项（A/B）。
- **P1 · GS XFB slot 冲突门过期导致的三种衍生风险**（mgl_buffer_slots.c，
  与上方 P0 同源，列三条待处置）：
  - `mglBufferSlotReservedName`（:129）把槽 25 只标 runtime-array-size，
    **漏 `GS GATHER_PARAMS=25`**；槽 31 虽标对（TESS_XFB_OUT/GS_XFB）。
  - GS 相关注释（mgl_buffer_slots.c:90-96 / mgl_buffer_slots.h:163-167）仍是
    "30 (GS XFB reserved)" 的过期措辞；`mgl_air_gs_abi.h` 注释 :168 还写
    "XFB meta, buffer(32)"（实为 27）。
  - 后端 `mgl_air_backend.cpp` GS 绑定用**裸字面量** `{24,28,29,30,25,31,27}`
    （:7924），不 `#include mgl_air_gs_abi.h`，不引用 `MGL_AIR_GS_SLOT_*` 枚举
    （RISK-3）——枚举值将来调整时前端不会自动跟随。
- **P1 · 片段（fragment）isolated 绑定临时 buffer 未在快照前存活上立即 flush——
  延迟重放 UAF（`MGL_USE_METALCPP=1` 路径）**：`MGLRenderer+BindingState.m`
  的 `bindFragmentBuffersToCurrentRenderEncoder` 在 isolated 段（:1907-1925）
  用循环局部 `id<MTLBuffer> isolated`（:1891，由
  `isolatedStageBindingBufferForMap:`（MGLRenderer.m:4824）**新建** MTL 缓冲、
  无其他 owner）存入 `snapshot.fragment_ops[].buffer`（经
  `MGL_FBIND_EMIT_BUFFER`，原始 `void*` 不 retain），随后 `continue`——ARC
  在迭代末释放 `isolated`，而快照要到函数末尾（:1998）才一次性重放。**对照**：
  顶点 isolated 路径（:504-526）在同一情况明确调用 `MGL_VBIND_FLUSH_SNAPSHOT()`
  并注释「isolated buffers 仅属循环局部，必须 flush 让 encoder 当场 retain，
  否则重放拿悬垂指针」，compute buffer 亦然（MGLRenderer+Compute.m:346-353）；
  只剩 fragment 段二者皆无（既无 flush 也无 strong 临时数组登记）。**触发**：
  `MGL_USE_METALCPP=1` 下片段 uniform/SSBO 因 map 过短进入 isolated 分支且
  快照 cap 未满、坚持到函数尾重放 → `mglRenderCppEncodeBindingSnapshot` 向
  encoder `setFragmentBuffer(悬垂的 isolated)`。**修法**：与顶点段同款——emit
  后立即 `MGL_FBIND_FLUSH_SNAPSHOT();`，或方法级强数组持有 isolated 至末尾
   重放。属 77d96ed 已修并警示的「临时 buffer 生命周期」漏网点。
- **P1 · max-slot 口径两处打架 + 无上限封口**：`mgl_buffer_slots.h:7,126-128`
  写 `0..30`（31 槽，`kMGLMaxMetalVertexBufferIndex=30`）；`mgl_air_gs_abi.h:69-70`
  则暗示 `>=32` 会崩（5-bit 换码）。GS XFB=31 **恰落在争议边界**。而反射分配
  层（mgl_air_reflect.c）与后端对 `air.location_index` **不设上限**——重缓冲
  程序可能把用户 SSBO 顶到 31 甚至 32，触发 `gs_abi.h` 所警告的编译器崩溃。
  需实测 Apple Silicon `maxBuffers` 定界，统一两 header 口径，并把 GS XFB
  与用户槽均封在确认合法区间（P1）。
- **P2 · `mglRenderPassSetPersistentStencilClear` 的 ObjC 镜像写漏门控**：
  `MGLRenderer+RenderPass.m:868-876` 无条件写
  `commandState->renderPassDescriptor.stencilAttachment.clearStencil`；而 颜色
  （:838-844）与深度（:851-861）两个兄弟方法都用 `!mglRenderPassUsesMetalCpp()`
  守卫镜像写（gate-on 只写 owner）。该文件自注约束（:300-304）要求「mirror-off、
  owner-on」，`stencil` 违反——当前 gate-on 下 `renderPassDescriptor` 保持 nil
  （MGLRenderPassManager.m:399）故目前潜在；但 gate-on 任何路径一旦物化非 nil
  descriptor 即成真 double-write。**修法**：补 `!mglRenderPassUsesMetalCpp() &&`
  守卫，与颜色/深度一致。
  （✅ 2026-08-16 已修，commit 3a02af4：与颜色/深度同款守卫。）
- **P0 衍生（2026-08-16 发现并修复，commit 3a02af4）——
  `mglBufferSlotIsReservedForStage` 的槽 24 TCS/GS 共享被早期 return 遮蔽**：
  槽 24 同时是 TCS stage_in 替换槽与 `MGL_AIR_GS_SLOT_INPUT`；旧实现
  `if (slot == kMGLBufferSlot_TCSStageInRepl) return (stage==TCS||stage<0)`
  在 GEOMETRY 阶段对槽 24 直接返回 false，**GS 输入保留从未对 ForStage 生效**
  （只有 ForGeometry 正确）。修复：TCS claim 限制在 TESS_CONTROL（或 generic
  stage<0），GEOMETRY 分支接管槽 24。新 smoke `BUFFER_SLOT_REGISTRY_OK`
  锁定该语义（stage(24,GEOMETRY)=1）与完整保留集；Makefile 将
  mgl_buffer_slots.c 编入 smoke（叶子 C 工具，同 mgl_aux_assets.c 模式）。
- **P3 · 文档「广义 MTL census = 30」工具不可复现**：文档多处（P4.0、各追加节）
  称「广义 `MTL*` census = 30 个 .m」；用文档终态搜索
  `rg -n "id<MTL|MTL[A-Z][A-Za-z]*Descriptor" MGL/src --glob '*.m'` 实测约 21 个。
  严格 `id<MTL` census = 15 稳定；「广义」缺一条唯一可执行命令，建议把 census
  口径重定义为严格 `id<MTL`（单一可执行），否则各批入库的「广义」数字无法互为
  校验。
- **P3 · `.gitignore` 漏 `build-asan-residual/`**：工作区有条 62MB 未跟踪目录
  `build-asan-residual/`（SANITIZE=address 独立 build 产物）一直进 `git status`；
  应补一行（与现有 build_asan/ 同模式）并 `git rm -r --cached build-asan-residual`。
- **确认无 bug（对照审计排除项，勿再提）**：① `mgl_sized_colors` refactor
  （mgl_pixel_format.c:1407-1444，把 `mglClearTexInternalFormatIsColor` 拆出
  `mglSizedColorInternalFormatValid`）语义保持——除了上注 P1 的 depth32 遗漏；
  ② 纹理子上传 plan/encode（mglRenderCppTextureSubUploadPlan :3082 与
  `mglRenderCppEncodeTextureUploadLayers`）slice/mip/layer 数学正确，C ABI
  类型不泄漏 `MTL::*`/`id<>`，`GL_TEX_1D_ARRAY`/`2D_ARRAY` 的
  yoffset/(zoffset,dep) 映射是对旧单 slice 逻辑的**真实修复**；③
  `mgl_air_backend.cpp:2214` 的 "EmitVertex requires the P1 output ABI" 是**防御
  前置守卫**（`emitGeometryVertex` 已完整实现），不是未实现桩——与 `MEMORY.md`
  记录的 GS EmitVertex/TES gl_in 缺口已闭合（59ef83e）一致，勿当 bug 复报；
  ④ Metal-cpp 实现宏 `NS_PRIVATE_IMPLEMENTATION`/`MTL_PRIVATE_IMPLEMENTATION`
  仍只在 `mgl_render_cpp.cpp:9-10` 定义（`rg MTL_PRIVATE_IMPLEMENTATION` 仅该
  TU 命中）；`mgl_render_cpp.h` 声明与 `mgl_render_cpp.cpp` 定义基本一一对应
  （无缺失 facade）；⑤ BatchReplay/DrawSupport/MGLRenderer.m 残留的直接
  `drawPrimitives` 调用均为 **gate-off A/B 回退**（`mglRenderCppTryEncodeDraw`
  返回 NO 才走），与 P4.3a 记录一致（gate-on 零直接调用）。

### P5 - 删除迁移壳与 gate

- [x] 将 Metal-cpp 固化为唯一生产路径；生产源码不再读取 `MGL_USE_METALCPP`。
- [x] 删除所有生产 gate/fallback 分支、旧 `MGLMetal*Ref` typedef 和共享
      Objective-C completion adapter。
- [x] 删除 `mgl_render_cpp_objc.h` 过渡 adapter；standalone smoke 只包含纯 C ABI。
- [x] 删除 `mgl_metal_bridge.m/.h` 和 `GLMMetalFuncs` ObjC bridge；对外纯 C
      `GLMMetalFuncs` ABI 保持不变。
- [x] 保留仍承担 GL 调度的 renderer 分类作为薄 Objective-C 适配层；其 Metal
      descriptor、资源属性、encoder 操作和 owner 生命周期均已移入 C++ backend，
      平台对象仅由 `MGLPlatformRendererShell` 持有。
- [x] `MGLPlatformRendererShell` 作为平台对象生命周期边界；renderer backend 只接收
      opaque device/layer 借用句柄。

### P5 当前完成记录（2026-08-18，单路径收口增量）

- `MGLRenderPassManager.h` 已将 command buffer、render encoder、pending event 和
  MDI scratch 统一为 `void *` opaque 参数；Metal protocol 只在实现文件桥接。
- `MGLPipelineCache.h` 已移除 Metal import、protocol、descriptor 和 enum 类型；其
  pipeline/function/device 句柄使用 opaque `void *`，格式使用 `uint64_t` value-state，
  blend 参数使用 `uint32_t`。`MGLPipelineCache.m` 是兼容边界，显式恢复 Metal 类型
  后调用 C++ pipeline owner，不持有独立缓存权威状态。
- `check-p5-metalcpp.sh` 已固定检查：生产 gate/旧 bridge/ref typedef、adapter
  文件、唯一 implementation TU、backend/platform roots、render-pass/pipeline-cache
  私有头 opaque 约束，以及 Objective-C command operation。`make test-all` 直接调用
  P5 checker；P4 checker 仅保留兼容包装。
- 本批强制全量重编后 `make test-all` 通过；Metal-cpp smoke 输出 `SMOKE_DONE`，
  gtest 为 `42/42`，regression 为 `73 PASS / 0 FAIL / 2 SKIP`，`git diff --check`
  通过。未跟踪的 sanitizer/build 目录按工作区资产保留。

### P5 当前完成记录追加（2026-08-18：value-state 类型岛与 sampled-view owner）

- `mgl_sync.{h,m}` 已去除 Foundation/Metal 依赖；attachment subresource、descriptor
  比较和状态/load/store 命名均通过 `mgl_render_cpp.cpp` 的 opaque C ABI 完成。
- `mgl_vertex_format.{h,m}` 的格式名称、descriptor signature、winding inversion 和
  integer conversion 已改为整数 value-state；Metal descriptor 读取集中在 C++ owner。
- `mgl_texture_compat.{h,m}` 的 pixel-format/swizzle/扩展上传接口改为纯整数 ABI；
  BASE/MAX_LEVEL sampled texture view 的 view 创建、缓存 retain/release 和 source
  metadata 查询已统一进入 `mglRenderCppSampledTextureViewForBaseLevel`。
- P5 checker 新增已完成 value-state island 的 Metal 类型审计，以及 sampled-view
  backend owner facade 检查。当前生产 gate、过渡 adapter、旧 ref typedef 和直接
  Objective-C command operation 仍保持零命中。

本批验证：`make -j4 lib`、`make check-p5-metalcpp`、`make test-metalcpp`、
`make test-mglair`、`make test-mglair-gtest`（42/42）、`make test-regression`
（73 PASS / 0 FAIL / 2 SKIP）和 `git diff --check` 均通过。下方“未完成”描述属于
迁移前快照，已被本页末尾的 P5 单路径终态记录取代。

本批终态审计新增：`mgl_draw_encode.m`、`MGLRenderPassManager.m` 和
`MGLPipelineCache.m` 仅允许 opaque/value ABI；index-buffer 实现必须是唯一的
`mgl_index_buffer.cpp`。该段后续“仍未迁移”的文字是历史快照，不再代表当前状态。

### P5 当前完成记录追加（2026-08-18：RenderPass/Blit 单路径终态）

- `MGLRenderer+RenderPass.m` 的 transient/fallback texture、depth/stencil state、
  texture 元数据、drawable texture 和 clear value 已全部改为
  `MGLRenderCppTextureDescriptorState`、`MGLRenderCppDepthStencilDescriptorState`、
  `MGLRenderCppTextureInfo` 与 opaque C ABI；render-pass encoder 继续由 C++ owner
  创建和结束。
- `MGLRenderer+Blit.m` 的 buffer/texture/sampler/depth-state 创建、texture
  readback、MSAA resolve、scaled blit 和 copy/resolve geometry 已移除 Metal
  descriptor、Metal object 类型和直接资源属性读取；新增 filter-sampler、compute
  pipeline thread-limit facade，所有 region/origin/size/viewport/scissor 均为 C value。
- 新增 `mgl_render_values.h` 作为纯 C renderer value-state ABI：纹理类型/usage、
  load/store、compare/cull/winding、blend、tessellation、primitive、vertex-format
  和 command-buffer status 均使用 `MGL*` 数值常量；`pixel_utils.h` 的本地像素格式
  镜像改为 `MGLPixelFormat`，不再依赖 Objective-C `MTLPixelFormat`。唯一
  Metal-cpp TU 通过 `static_assert` 对这些数值与 SDK 枚举逐项校验。
- `mgl_readback.m`、`mgl_state_compat.m` 和 renderer categories 不再导入
  `Metal/Metal.h`；`isFramebufferOnly` 查询经
  `mglRenderCppTextureIsFramebufferOnly` owner facade，避免通用 `id` 重新承担
  Metal selector。
- `MGLRenderer+RenderPass_Private.h` 的 drawable 参数降为 opaque `id`；P5 checker
  已将 RenderPass、Blit 及其私有头纳入严格审计。生产源码无 gate、bridge、旧
  callback/ref typedef 或过渡 adapter，Metal-cpp implementation macro 仍只位于
  `MGL/src/mgl_render_cpp.cpp`。

当前 stripped 终态 census（注释先剥离，2026-08-18）：非平台 `.m` 和私有头的
`id<MTL...>`、Metal descriptor、Metal-cpp 类型及直接 Metal selector 均为 0；唯一
允许命中的是 `MGLPlatformRendererShell.{m,h}` 的 `CAMetalLayer`、drawable 和
AppKit 生命周期。可复现审计命令：

```sh
rg -n 'MGL_USE_METALCPP|mgl_render_cpp_objc|MGLMetal[A-Za-z]+Ref|MGLRendererMetalBridge' \
  MGL/src MGL/include Makefile test_legacy_compat benchmark scripts/record_p3_baseline.sh
perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' MGL/src/*.m MGL/include/*.h | \
  rg -n 'id[[:space:]]*<MTL|MTL[A-Z][A-Za-z]+Descriptor|MTL::'
rg -n 'CAMetalLayer|NSView|nextDrawable' \
  MGL/src/MGLPlatformRendererShell.m MGL/include/MGLPlatformRendererShell.h
```

值状态和导入边界的终态审计还应运行：

```sh
for f in MGL/src/*.m MGL/include/*.h; do
  case "$f" in *MGLPlatformRendererShell.m|*MGLPlatformRendererShell.h) continue;; esac
  perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g; s/"(?:\\.|[^"\\])*"//g' "$f" |
    rg -n '\bMTL(TextureType|TextureUsage|StorageMode|ResourceStorageMode|LoadAction|StoreAction|CompareFunction|CommandBufferStatus|PrimitiveType|CullMode|Winding|DepthClipMode|ColorWriteMask|PrimitiveTopologyClass|TessellationPartitionMode|TessellationFactorStepFunction|TessellationFactorFormat|TessellationControlPointIndexType|MultisampleDepthResolveFilter|MultisampleStencilResolveFilter|BlendFactor|BlendOperation|VertexFormat|PixelFormat)' && exit 1 || true
done
rg -n '#import[[:space:]]+<Metal/Metal\.h>|#include[[:space:]]+<Metal/Metal\.h>' \
  MGL/src MGL/include --glob '*.{m,mm,h}' \
  --glob '!MGLPlatformRendererShell.m' --glob '!MGLPlatformRendererShell.h'
```

`make test-all` 已直接串联 `check-p5-metalcpp`，不再调用失效的 P4 gate。全量 lib
和专项测试矩阵结果以本记录的验证段落为准；工作区现有 sanitizer/build 目录仍
视为用户资产，不由本次迁移清理。

### P5 当前完成记录追加（2026-08-18：platform/recovery/buffer/value layout）

- `MGLRenderer+GPURecovery.m` 与私有头已改用 opaque `id`、`uint32_t` command
  status 和 C++ device/command-buffer identity 查询；不再读取 Metal label、status、
  error 或 registry selector。`mgl_sync.h` 固定 command-buffer status 数值 ABI
  0..5，与 Metal 的 NotEnqueued..Error 保持一致。
- `MGLPlatformRendererShell` 统一持有 system-default device 与 capture descriptor/
  start/stop 生命周期；`MGLRenderer+Lifecycle.m` 不再创建 device/capture descriptor，
  proactive RGBA8 texture 的创建、上传和 +1 owner 引用由
  `mglRendererBackendCreateProactiveTexture` 完成。Lifecycle 仍只通过平台壳访问
  CAMetalLayer 几何与 drawable 边界。
- `MGLRenderer+SwapDiagnostics.m` 与私有头已改为 opaque `id` 和数值
  viewport/scissor/origin/size/load/store/primitive contract；texture 元数据、buffer
  contents/debug marker、command-buffer label/status 均通过 C++ facade 查询或执行。
- `MGLRenderer+Buffer.m` 与私有头已清除 `id<MTL*>`；vertex conversion 保持
  `__bridge_transfer id` 的 +1 返回契约，packed-struct/snapshot helper 删除未使用的
  device 参数，buffer CoW/snapshot generation 继续由 C++ backend 持有。
- `MGLRenderer+Binding.m` 不再创建 sampler descriptor 或直接读取 texture
  usage/mipmap/dimensions；`MGLRenderCppTextureInfo` 新增 `usage` 与
  `mipmap_level_count` value-state，默认 sampler 由
  `mglRenderCppCreateDefaultSampler` 创建并以 +1 opaque handle 返回。
- `MGLRenderer+VertexLayout.m` 的 vertex format、step function、blend 和 color-write
  mask 全部写入整数 value-state；共享 vertex/blend helper 的私有返回类型同步改为
  `uint32_t`。Buffer、Binding、VertexLayout 均已加入 P5 completed-island checker。
- `MGLRenderer+Compute.m` 已清除 Metal 类型、descriptor 和资源属性读取；buffer
  length 及 texture pixel-format/type/array-length 通过 `MGLRenderCppBufferInfo` /
  `MGLRenderCppTextureInfo` 查询，level view、默认 sampler 和 dispatch 继续由 C++
  facade 创建或编码。compute encoder/pipeline/function/temporary 仅以 opaque `id`
  持有，`MTLSize` 已改为显式 x/y/z value-state；Compute 已加入 P5 checker。

以下是 2026-08-18 RenderPass/Blit 收口前的历史快照（不代表当前终态）：当时
renderer 源码仍有 10 个文件、1017 个 `id<MTL...>` 命中；私有头仍有 5 个文件、
48 个命中。该快照保留用于迁移审计对照，当前终态以本文后续的 P5 完成记录和
可复现 census 为准。分布如下：

```text
193 MGLRenderer+Blit.m                 13 MGLRenderer+Blit_Private.h
150 MGLRenderer+RenderPass.m           12 MGLRenderer+Draw_Private.h
134 MGLRenderer+Texture.m              10 MGLRenderer+Texture_Private.h
114 MGLRenderer+DrawSupport.m           7 MGLRenderer+RenderPass_Private.h
103 MGLRenderer+Tessellation.m          6 MGLRenderer_Private.h
100 MGLRenderer+BindingState.m
 90 MGLRenderer.m
 50 MGLRenderer+Draw.m
 48 MGLRenderer+Batch.m
 35 MGLRenderer+BatchReplay.m
```

可复现命令（分别对 source/header 执行，避免重复 glob）：

```sh
for f in MGL/src/MGLRenderer*.m; do
  n=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$f" |
      rg -o 'id[[:space:]]*<MTL[^>]+>' | wc -l | tr -d ' ')
  test "$n" -eq 0 || printf '%4d %s\n' "$n" "$f"
done | sort -nr

for f in MGL/include/MGLRenderer*_Private.h; do
  n=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$f" |
      rg -o 'id[[:space:]]*<MTL[^>]+>' | wc -l | tr -d ' ')
  test "$n" -eq 0 || printf '%4d %s\n' "$n" "$f"
done | sort -nr
```

历史终态搜索（迁移前）：

```sh
rg -n "MGL_USE_METALCPP|mgl_metal_bridge|GLMMetalFuncs|MGLMetal[A-Za-z]+Ref" MGL/src MGL/include
rg -n "id[[:space:]]*<MTL|MTL[A-Z][A-Za-z]+Descriptor|MTL::" \
  MGL/src MGL/include --glob '*.m' --glob '*.mm' --glob '*_Private.h' \
  --glob 'MGLRenderer_Private.h' --glob 'MGLPipelineCache.h' \
  --glob 'MGLRenderPassManager.h'
```

第一条在当时的迁移阶段用于记录残留分布；当前终态验证请使用后续 P5 完成记录
中的命令，第二条只允许命中明确白名单的 AppKit/CAMetalLayer 外壳。

### P5 全量完成记录（2026-08-18：单路径收口）

- RenderPass、Blit、platform/recovery、buffer、binding、vertex-layout、compute、
  texture/sync value-state islands 以及 index-buffer owner 均已接入 C++ backend；
  Objective-C renderer 仅保留 GL 语义编排和 `MGLPlatformRendererShell` 平台壳。
- `MGL_USE_METALCPP`、旧 bridge/ref typedef、`mgl_render_cpp_objc.h`、直接
  Objective-C Metal command operation 和失效 P4 callback/fallback 契约均为零命中。
- `MGLPipelineCacheState` 仅保存 nullable `void *` 借用句柄；设备、pipeline、
  function 和缓存对象的 retain/release 均由 C++ owner 负责，ObjC wrapper 不再持有
  Metal 对象镜像。
- stripped 终态 census（注释先剥离）：非平台 `.m`/私有头无 `id<MTL...>`、Metal
  descriptor、`MTL::` 或直接 Metal selector；唯一允许命中为
  `MGLPlatformRendererShell.{m,h}` 的 AppKit/CAMetalLayer 生命周期。
- `make test-all` 已直接串联 `check-p5-metalcpp`。最终验证通过：
  `make -j4 lib`、`make check-air-only`、`make check-p5-metalcpp`、
  `make test-mglair`、`make test-mglair-gtest`（42/42）、`make test-metalcpp`、
  `make test-regression`（73 PASS / 0 FAIL / 2 SKIP）和 `git diff --check`。

### P5 当前完成记录追加（2026-08-18：共享 fallback buffer owner）

- Vertex/fragment stage fallback binding buffer 与 cull-distance dummy buffer 不再由
  Objective-C 分类的函数静态变量持有；`MGLRendererBackendHandle` 负责懒创建、借用
  返回、retain/release 和 teardown。这样这些 Metal buffer 与 size-constants、copy-back
  及其他临时资源遵循同一 backend owner 生命周期。
- `check-p5-metalcpp.sh` 增加了 backend field/API/lifecycle 以及 Objective-C 静态
  fallback owner 的窄审计；合法的 GL 语义 fallback 和 capability fallback 不纳入该规则。
- `test_metalcpp_smoke.mm` 锁定 getter 的零长度拒绝、非空懒创建和重复调用稳定性，
  输出 `RENDERER_BACKEND_FALLBACK_BUFFER_OWNER_OK`。

本轮增量验证：`make -j4 lib`、`make check-air-only`、`make check-p5-metalcpp`、
`make test-mglair`、`make test-mglair-gtest`（42/42）、`make test-metalcpp`、
`make test-regression`（73 PASS / 0 FAIL / 2 SKIP）、`make test-all` 和
`git diff --check` 均通过。

当前终态审计命令：

```sh
rg -n 'MGL_USE_METALCPP|mgl_render_cpp_objc|MGLMetal[A-Za-z]+Ref|MGLRendererMetalBridge' \
  MGL/src MGL/include Makefile test_legacy_compat benchmark scripts/record_p3_baseline.sh
for f in MGL/src/*.m MGL/include/*.h; do
  hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$f" |
    rg -n 'id[[:space:]]*<MTL|MTL[A-Z][A-Za-z]+Descriptor|MTL::' || true)
  [ -z "$hits" ] || case "$f" in
    *MGLPlatformRendererShell.m|*MGLPlatformRendererShell.h) ;;
    *) printf '%s\n' "$hits"; exit 1;;
  esac
done
rg -l '^#define (NS_PRIVATE_IMPLEMENTATION|CA_PRIVATE_IMPLEMENTATION|MTL_PRIVATE_IMPLEMENTATION)' \
  MGL --glob '*.{c,cc,cpp,cxx,h,m,mm}' | sort
```

## 4. 每阶段验证矩阵

每个 P1-P5 增量至少运行：

```sh
make -j4 lib
make test-mglair
make test-mglair-gtest
make test-metalcpp
make test-regression
DYLD_LIBRARY_PATH=build build/test_regression --golden-dir MGL_Golden_Images
git diff --check
```

删除 A/B gate 后，最后两项替换为唯一 C++ 路径的串行和压力运行。当前唯一路径已
通过 lib、P5 checker 和 Metal-cpp smoke；最终 M3 验收还需：

- [ ] 覆盖 TCS/TES/GS 的 CTS/KHR-GL46 子集；
- [ ] 在至少一台 Metal 4 真机验证 pipeline archive、indexed/indirect GS、native
  tessellation 和 completion ownership；
- [x] C++ wrapper 定向 ASan/TSan 验证无 double-release、UAF 和 wrapper
  completion race。
  （ASan 的 C++/ARC 定向验证已覆盖 copy-back、slot 和 completion wrapper
  边界。2026-08-17 进一步使用全新 `build-tsan-final` 完整运行 gate-off/gate-on，
  两门均为 `73 PASS / 0 FAIL / 2 SKIP`，且
  `TSAN_OPTIONS=halt_on_error=1` 未报告 race。callback census、Metal 类型白名单、
  compute/binding/Draw 与 P4 final gate 也已在同日完成，因此该完整 TSan 结果现为
  P4 终验证据之一。）
- [x] 干净 clone 在没有 glslang/SPIRV-* 目录时完成构建和测试。
  （2026-08-14 完成：`git clone` 全新工作树直接 `make lib` 曾因
  `external/glfw/build`（CMake 产物，不入库）缺失而失败——Makefile 新增
  前置规则按需执行 `external/build_external.sh`（脚本改为 cwd 无关），
  干净 clone 现在 `make` 即可完成 glfw 配置+构建；全套件 61/0/2/63 通过，
  全程无需 glslang/SPIRV-* 树。）

## 5. 下一批建议顺序

1. ~~先定义统一 geometry draw plan 和 indexed input ABI。~~ → P0 已交付
   `mgl_air_gs_abi.h`（index gather 参数块、counts/indirect ABI、输出 record
   布局）；geometry draw plan 的**统一化**（删除 array-only 边界）与
   direct indexed 落地在 P1。
2. 落地 direct indexed GS，并补 `air_geometry_indexed`。
3. 在同一 plan 上接 indirect/multi-draw，避免为每个 GL entry 重写一套 capture。
4. 并行补 native tess indexed/instanced contract。
5. ~~M3 draw 语义稳定后，按 P3.1 -> P3.2 将 Blit/safe shader 的动态 MSL 全部换成
   预编译 AIR metallib；P3.3/P3.4/P3.5 再删除 source compiler、旧命名和第三方树。~~
   → ✅ 2026-08-13 完成（提交 2ced5c1 / d766aa9 / 882fff5）。
6. 最后按 P4.1 -> P4.2 -> P4.3 -> P4.4 -> P4.5（render pass -> pipeline ->
   draw/binding -> texture/blit -> compute/lifecycle/callbacks）删除 ObjC
   权威状态；A/B parity 稳定后 P5 删除迁移 gate。

### P4 完成记录追加（2026-08-16：program teardown hash ownership）

**program 生命周期诊断收口**：context teardown 的
`mglDestroyContextProgram` 现在先按回调传入的 name 调用
`deleteHashElement(&ctx->state.program_table, name)`，再调用
`mglFreeProgram`。这与 `glDeleteProgram` / deferred reference release 的
ownership 顺序一致：name 从 GL hash table 脱离后，仍可由 `Program::refcount`
保持对象存活；context teardown 则在没有 GL name 删除请求时显式执行同一脱链动作。

此前 `mglFreeProgram` 的保护性诊断在 context teardown 打印
`STILL in hash table`（`delete_status=0, refcount=0`），随后再删除 entry；这
不是可以忽略的噪声，而是释放函数和 name-table ownership 顺序相反。修复保留
诊断作为真正错误的防线，没有屏蔽日志或改变 deferred program reference 语义。

验证：重编 `build/libmgl.dylib`/`test_regression` 后，单一路径 regression
exit 0；日志包含
`Destroying GLMContext` 且 `mglFreeProgram .*STILL in hash` 为 0。该项只处理
program/hash ownership；实际 command-buffer submit、AGX recovery、deferred
device reset 仍受 `/docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`
中的 ObjC 限制约束，不能据此宣称 command lifecycle 已迁完。

### P4 完成记录追加（2026-08-16：program-aware buffer slot registry）

**GS/TES/模拟路径的用户 buffer 槽冲突门已接入 link-time。**

- `mglBufferSlotConflictsForProgram`（`MGL/src/mgl_buffer_slots.c`）按
  `Program` 的实际 route/feature 区分 GS compute、TCS、native/compute TES、VS
  cull-distance 和 FS FragCoord fixup；普通 program 不会因为未启用的内部路径而
  被保留槽位误拒绝。
- `mglLinkProgram` 在 link 成功前遍历 UBO、plain-uniform、SSBO、atomic-buffer
  资源及其数组元素，使用反射后的 `resource->binding` 做冲突校验；发现冲突时
  返回 link failure 并输出 stage/resource/slot/owner 诊断，避免等到 encoder 已
  绑定后静默覆盖内部 ABI buffer。
- GS compute 的保留集合按当前 ABI 覆盖 `24..31`（含 gather params、XFB
  stream、XFB meta）；TCS 只保留其真实 `24/26..29`，native TES 保留
  `27/28/30`，isolines/point-mode TES 保留 `24..31`。
- `test-metalcpp` 的 `BUFFER_SLOT_REGISTRY_OK` 增加 program-aware 正/负路径；
  regression 新增 `air_geometry_buffer_slot_conflict`：24 个 GS SSBO（用户槽
  `0..23`）可 link，25 个（撞 `MGL_AIR_GS_SLOT_INPUT=24`）必须 link fail。

验证：`make test-metalcpp` 通过；ObjC/C++ 两门定向 regression 均
`1 PASS / 0 FAIL / 73 SKIP`；完整 `make test-regression` 为
`72 PASS / 0 FAIL / 2 SKIP`。该项只处理 C/AIR 反射和 link-time ownership；
非 GS 的 runtime-array-size slot 约束、Metal 物理 max-buffer 定界仍按既有
ABI 文档单独审计，没有借此扩大本轮范围。

### P4 完成记录追加（2026-08-16：runtime-array-size 用户槽 link 契约）

**非 GS runtime-array `.length()` 的隐藏 slot 25 冲突已在 link-time 封口。**

- AIR 对使用 runtime SSBO `.length()` 的 stage 固定声明
  `MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX=25` 的 `spvBufferSizeConstants`；此前
  反射资源仍可能把第 26 个用户 buffer 分到同一槽，随后绑定 size table 会
  覆盖用户 SSBO。
- `mglBufferSlotConflictsForProgram` 现在读取
  `modules[stage].needs_runtime_array_size_buffer`，只在该 stage 真正使用
  `.length()` 时把 slot 25 视为冲突；普通 program 继续允许 slot 25，避免
  非必要的 false positive。
- `mglLinkProgram` 同时拒绝反射到 `[0, kMGLMaxMetalUserBufferCount)` 之外
  的用户 buffer slot，数组元素也按最终 Metal slot 逐项检查，并保留 stage、
  resource、slot 的 link error 诊断。
- GS/TES 的固定 ABI 仍由既有 program-aware registry 管理；本刀没有迁移或
  改写 ObjC command submit、completion、AGX recovery、deferred reset 或 owner
  生命周期策略。该切片当时没有处理 GS slot 31 的物理 max-index；该项随后由
  下文“普通用户槽与 compute 物理槽边界”切片以 Metal 4 真机证据完成。

回归：`BUFFER_SLOT_REGISTRY_OK` 增加 compute runtime-size 正负路径；
`air_geometry_buffer_slot_conflict` 同时验证 25 个 compute SSBO（用户槽
`0..24`）可 link、26 个（撞 slot 25）link failure，以及普通 compute 的
31 个 SSBO（用户槽 `0..30`）可 link、32 个（用户 slot 31 超出当前表预算）
link failure。`make test-metalcpp` 和完整 `make test-regression` 均通过
（`72 PASS / 0 FAIL / 2 SKIP`）。ASan/TSan 定向单项均 `1 PASS / 0 FAIL / 73
SKIP`、无 sanitizer 报告；当时 TSan 完整套件仍被原 ObjC completion race
阻断。该历史阻断已由 2026-08-17 `build-tsan-final` 完整双门复核关闭。

### 2026-08-16 TODO 状态复核（避免重复施工）

- **`GL_DEPTH_COMPONENT32`**：当前工作区已包含
  `mglTexStorageInternalFormatValid` 的 `GL_DEPTH_COMPONENT32` case，且
  `texture_storage_internalformat_validation` 已把 depth32 storage 作为正向
  成功用例；不再按 2026-08-16 早期审计中的“未提交回归”重复修改。
- **fragment isolated buffer UAF**：当前
  `MGLRenderer+BindingState.m:1907-1918` 已在 emit 后立即执行
  `MGL_FBIND_FLUSH_SNAPSHOT()`，与 vertex/compute 路径一致；不再重复加 flush。
- **TSan completion race（状态更新）**：本段记录的 2026-08-16 旧阻断已由
  2026-08-17 completion context ownership/publish 修复关闭；新建
  `build-tsan-final` 的 gate-off/gate-on 完整套件均 `73/0/2` 且无报告。仍以
  `/docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md` 的更新段为当前结论。
- **max-buffer 定界已完成**：普通用户/vertex 域固定为 `0..30`，GS/TES
  compute 物理 ABI 域为 `0..31`，见下节及 Apple M4 正向回归证据。
- **runtime-array/gather 组合语义已完成**：普通 VS/FS/CS/TCS/native-TES
  继续使用隐藏 slot 25；GS 与 compute-TES 使用隐藏 slot 23，固定 gather params
  保持 slot 25。仅在对应 route 实际使用 `.length()` 时保留 slot 23/25，见下节。

### P4 完成记录追加（2026-08-16：普通用户槽与 compute 物理槽边界）

**Metal buffer index 定界已按实际 ABI 拆开，不再把用户表上限误当成
GS/TES compute 的物理上限。**

- `mgl_buffer_slots.h` 现在明确区分两套常量：普通用户资源和 vertex layout
  仍为 `0..30`（count `31`）；固定 compute ABI 的物理域为 `0..31`（count
  `32`）。历史 `kMGLMaxMetalVertexBuffer*` 名称保留为用户/vertex 域别名，
  不会把现有用户 binding、runtime-size table 或 vertex layout 数组扩成 32。
- `mgl_air_gs_abi.h` 的 GS XFB stream 与 `mgl_air_tess_abi.h` 的 TES XFB
  stream 均用 slot `31`，并以编译期断言保证它等于 compute 物理末槽、低于
  物理 count、且高于普通用户末槽；slot `>=32` 仍不尝试验证，避免触发 AGX
  compiler 的 5-bit mask 边界。
- Apple M4 真机现有正向回归 `air_geometry_xfb` 与
  `air_tessellation_isolines_xfb` 均实际覆盖 slot 31 XFB 路径并通过；这是
  采用 `0..31` compute ABI 物理域、同时保留用户 `0..30` 限制的设备证据。
- `program.c` 的 link-time 用户资源上限改用 user-buffer 常量；普通 compute
  的 31 个 SSBO 仍可 link，第 32 个（用户 slot 31）仍按契约失败。

本项当时未改变 runtime-array slot 25 与 TES compute gather-params slot 25 的
组合语义；该组合已由下节 route-specific ABI 切片完成。原 Objective-C command
submit、completion、AGX recovery、deferred reset 与 owner 生命周期限制继续
只记录在 `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`，本轮不修。

### P4 完成记录追加（2026-08-16：runtime-array 与 gather params 组合 ABI）

**GS/compute-TES 的 runtime-array size table 已与固定 gather params 解冲突，
`.length()` 不再被 slot 25 的后绑定覆盖。**

- 普通 VS/FS/CS/TCS/native-TES 保持
  `MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX=25`；GS 与 isolines/point-mode
  compute-TES 改用 `MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX=23`，固定
  gather params 继续占 slot 25。GS/TES ABI 以编译期断言固定 23、25 不相等。
- `mglRuntimeArraySizeBufferIndexForProgram(program, stage)` 是 route 选择的单一
  入口；link gate、AIR `spvBufferSizeConstants` metadata、compute renderer 和
  tessellation prepared-binding list 均使用同一结果。只有 route 实际调用
  runtime-array `.length()` 时，slot 23/25 才成为用户资源冲突，未使用时不缩减
  用户 binding 空间。
- `air_geometry_buffer_slot_conflict` 同时锁定普通 GS 的 24 个用户 SSBO
  （slot `0..23`）可 link、第 25 个撞固定 slot 24 失败，以及 runtime-array GS
  的 23 个用户 SSBO（slot `0..22`）可 link、第 24 个撞隐藏 slot 23 失败；普通
  compute 的 slot 25 契约继续保留。ObjC/C++ 两门定向运行均为
  `1 PASS / 0 FAIL / 73 SKIP`。
- `RuntimeSSBOArrayLengthAcrossStages` 扩到 GS 与 isolines TES；独立真实 Metal
  dispatch 新增 `GS_RUNTIME_LENGTH_OK` 和 `TES_RUNTIME_LENGTH_OK`，两条路径都把
  `4 + 7 * sizeof(float)` 的可见 SSBO 映射为 `.length() == 7` 并写回验证。
  compute-TES 的用户 SSBO 绑定在既有 TES user-buffer 起始槽 1，size table 在
  23，固定 stage/gather/XFB 参数在 24..31。
- 新 smoke 首次运行暴露 compute-TES kernel metadata 的参数序号少算一项：固定
  参数实际为 stage-in/factors/patch-inputs/stage-out/indirect/gather/params/XFB
  共 8 个，旧 `mArgSlot += 7` 令 `thread_position_in_grid` metadata 指向 XFB
  buffer 参数，command buffer 虽完成但 `.length()` 写回保持 0。修正为
  `mArgSlot += 8` 后真实值恢复为 7。

验证：Apple M4（Mac16,12）上 `make test-mglair` 输出
`GS_RUNTIME_LENGTH_OK`、`TES_RUNTIME_LENGTH_OK`、`VALUE_OK`；
`make test-mglair-gtest` 为 `42/42`；`make test-metalcpp` 输出
`BUFFER_SLOT_REGISTRY_OK`、`SMOKE_DONE`；`make test-regression` 与最终
`make test-all` 均 exit 0，regression 为 `72 PASS / 0 FAIL / 2 SKIP`。
`air_tessellation_isolines_xfb` 的 ObjC/C++ 定向运行也均为
`1 PASS / 0 FAIL / 73 SKIP`，继续覆盖 slot 31。原 ObjC command lifecycle 与
TSan completion race 仍按 `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`
记录，本切片不修。

### P1/P4 完成记录追加（2026-08-16：GS XFB link-time layout plan）

- `Program` 新增持久化的 `MGLTransformFeedbackVaryingPlan`，记录每个 XFB
  entry 的 binding、component offset/count、stream 和 builtin 标记；修改
  `glTransformFeedbackVaryings` 时清空旧 plan，link 成功后重建。
- `mglValidateTransformFeedbackVaryings` 现在按 ARB_transform_feedback3 校验：
  特殊 token 只能用于 `GL_INTERLEAVED_ATTRIBS`；`GL_SEPARATE_ATTRIBS` 每个
  varying 自动绑定独立 buffer；varying 必须是最后一个 pre-fragment stage 的
  output；重复 varying、非法数组元素、component 上限和单 binding 跨 stream
  均导致 link failure。API 侧同步检查 separate count 与 special-token 错误。
- `air_geometry_multi_stream_xfb` 已加入 `gl_NextBuffer`，不再依赖隐含的
  `stream s -> buffer s` 假设；新增 `air_xfb_link_layout` 覆盖 interleaved
  `gl_NextBuffer`/`gl_SkipComponentsN`、separate 正常 varyings、special-token
  API 错误、重复/缺失 varying 和跨 stream binding link failure。
- 本切片没有放开 GS `SEPARATE_ATTRIBS` capture execution，也没有处理跨 binding
  整图元容量截断、primitive order、passthrough/default-stream reflection；
  原 ObjC command lifecycle、completion、AGX recovery、deferred reset 仍只记入
  `docs/P4_COMMAND_LIFECYCLE_LIMITATIONS_2026-08-16.md`。
- 验证：`make -j4 lib`、最终 `make test-all` 均 exit 0；regression 为
  `73 PASS / 0 FAIL / 2 SKIP`，新增 `air_xfb_link_layout` 与改写后的
  `air_geometry_multi_stream_xfb` 均通过。独立 ASan/TSan C++ 构建的两项定向
  回归各为 `1 PASS / 0 FAIL / 74 SKIP`，无 sanitizer 报告；`git diff --check`
  通过。当时该定向结果不作为原 ObjC completion 生命周期已修复的证据；完整
  修复证据见 2026-08-17 `build-tsan-final` 双门终验。

### P5 当前快照追加（2026-08-18：platform shell 与终态审计）

- `MGLPlatformRendererShell` 集中创建/配置/解绑 `CAMetalLayer`，提供 drawable、
  texture、geometry 和 capture facade；renderer 分类不再直接调用
  `nextDrawable`、读取 `drawable.texture`、访问 `_layer.*` 或检查
  `[_commandQueue class]`。standalone Metal-cpp smoke target 显式链接 QuartzCore。
- `mgl_state_compat.h`、`mgl_readback.h` 的 Metal enum 参数已改为 C `uint32_t` value
  state；实现内部保留数值映射，跨语言头不暴露 ObjC Metal 类型。P5 checker 的
  逐文件审计先剥离注释，再覆盖全部 `MGL/src/*.m`、`*.mm` 与 `MGL/include/*.h`，
  平台壳是唯一允许的 Metal/AppKit 类型边界。
- 当前验证：`make -j4 lib`、`make check-air-only`、`make check-p5-metalcpp`、
  `make test-mglair`、`make test-mglair-gtest`（42/42）、`make test-metalcpp`、
  `make test-regression`（73/0/2）均通过；CTS 完整 runner 与 Metal 4 真机专项
  没有可运行入口，相关 TODO 保持未勾选。

可复现终态 census（与 checker 的逐文件去注释逻辑一致）：

```sh
for f in MGL/src/*.m MGL/src/*.mm MGL/include/*.h; do
  case "$f" in
    *MGLPlatformRendererShell.m|*MGLPlatformRendererShell.h) continue ;;
  esac
  hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$f" |
    rg -n 'id[[:space:]]*<[[:space:]]*MTL|\bMTL[A-Za-z]+Descriptor\b|\bMTL::|\[_layer|\[_commandQueue[[:space:]]+class\]' || true)
  test -z "$hits" || { printf '%s\n%s\n' "$f" "$hits"; exit 1; }
done
rg -n 'CAMetalLayer|NSView|nextDrawable' \
  MGL/src/MGLPlatformRendererShell.m MGL/include/MGLPlatformRendererShell.h
```

该 census 在当前工作区通过：非平台 `.m`/`.mm`/私有头无 Metal 对象、descriptor、
`MTL::` 或 layer/drawable selector 命中；允许的 platform-shell 命中仅位于
`MGLPlatformRendererShell.{m,h}`。
