# OpenGL 4.6 / GLSL 4.60 核对：Geometry Shader 与 Layered Rendering

审计对象：`fix/gs-layered-rendering` HEAD `acac3c9`（不要对照 `main`）。  
行号相对 **acac3c9**。该分支之后已前进到 `d23a725` 一带（例如 `assembleReturn` 交叉赋值约在 `mgl_air_backend.cpp:5696-5705`，本文件仍写 acac3c9 的 `5659-5671`）。本审计不跟到 `d23a725`。  
性质：合规核对。本 PR 已落地 #1–#9 运行时修复（判定列标 **FIXED**）；笔记仍描述 acac3c9 上的原问题。  
规格来源（本次直接检索，不以仓库内旧笔记为准）：

- *The OpenGL Graphics System: A Specification (Version 4.6 (Core Profile) - May 5, 2022)*  
  https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf
- *The OpenGL Shading Language, Version 4.60.8*  
  https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html

Metal 没有 GS。实现路径是 VS capture → compute expansion kernel → 生成的 passthrough VS。下表只判断 **GL 可见行为**，不评价 Metal 映射是否好看。

判定：

- **COMPLIANT**：从源码可核对，与所引条款一致。
- **NON-COMPLIANT**：GL 可见行为与条款冲突（错误码、光栅化、查询、链接结果）。
- **FIXED**：本 PR 已改运行时代码，对应 acac3c9 上的不合规点。
- **UNCLEAR**：条款存在，但仅靠静态阅读无法判定（需 CTS / 运行）。

---

## 假设核对（用户提供的先验，全部重新验证）

| 假设 | 结论 |
| --- | --- |
| passthrough GS bypass 仍用 `strstr` 匹配 `EmitVertex` / `EndPrimitive` / 特定 `gl_Position=gl_in` 行 | **成立。** `mglProgramHasPassthroughGeometryShader`（`MGL/include/mgl_program_reflection.h`）正是这组源码子串。 |
| `gl_ViewportIndex` 在只写其中一个时与 `gl_Layer` 别名 | **部分成立，且规格不支持该别名。** VS `assembleReturn` 会交叉拷贝；GS `emitGeometryVertex` **不会**。VS 写 `gl_Layer` 本身是本树有意支持的扩展（近 `ARB_shader_viewport_layer_array`，回归测试覆盖），不是不合规点。代码注释误引 GL 4.6 §11.1.3.5 / §11.1.3.6（这两节实际是 Texture Access / Atomic Counter Access）。 |
| layeredness 在 FBO attachment 之间 sticky（commit `f55683f`） | **Metal `renderTargetArrayLength` 侧成立。** `mglRenderPassArrayLength` 跳过非 layered attachment，不把已有 array length 清零。规格 9.8 的 layered 定义更严（见下）。 |
| no-attachment FBO fallback 是 dummy 2D color，不是 layered texture | **成立。** `mglRenderPassFallbackRenderTargetForSize` 固定 `MGLTextureType2D`、`array_length = 1`，且不读 `FRAMEBUFFER_DEFAULT_LAYERS`。 |

---

## Verdict 表

非合规项在前，按规格严重度：错误光栅化 / 错误错误码 / 缺失查询（#1–#8）。  
`EndStreamPrimitive` 忽略 stream（#9）是真实 mismatch，但 **不与 #1–#3 同级**：多 stream 且非 points 应在链接失败（#20），points-only 下危害有限。

| # | 项 | 规格引用 | 代码位置 | 判定 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| 1 | Passthrough GS 源码启发式跳过整段 GS | **GLSL 4.60 §8.13**：“Emits the current values of output variables…”；**GL 4.6 §11.3**：“When the program object currently in use includes a geometry shader, its geometry shader is considered active”；**GL 4.6 §11.3.5**：“the counter is incremented every time the geometry shader is invoked” | `mglProgramHasPassthroughGeometryShader`（`MGL/include/mgl_program_reflection.h:60-73`）；跳过：`handleGeometryDrawIfNeeded`（`MGLRenderer+DrawSupport.m:1566-1568`）、`bindMTLProgram`（`MGLRenderer+RenderPass.m:1681-1689`）、`mglRenderBindAIRProgram`（`mgl_render.cpp:2791-2792`） | **FIXED** | 规格没有 “passthrough GS” 优化。启发式要求源码同时包含 `EmitVertex()`、`EndPrimitive()`、字面量 `gl_Position = gl_in[n_vertex_index].gl_Position`，且不含 `gl_Layer`/`gl_ViewportIndex`/`gl_PrimitiveID`。匹配则整段 GS 不跑（普通 VS→FS）。后果：(a) 其它输出/副作用被丢掉；(b) `GEOMETRY_SHADER_INVOCATIONS` / `GEOMETRY_SHADER_PRIMITIVES_EMITTED` 不走 `mglRecordActiveGeometryShaderQueryDraw`；(c) 注释/字符串误匹配会假阳性。XFB 程序被排除（`transform_feedback_varying_count > 0`），只避免捕获丢失，不修复光栅化。 |
| 2 | 无附件且 `FRAMEBUFFER_DEFAULT_LAYERS > 0` 的 layered FBO | **GL 4.6 §9.2.1**：“When a framebuffer has no attachments, it is considered layered (see section 9.8) if and only if the value of FRAMEBUFFER_DEFAULT_LAYERS is non-zero”；同节：“If there are no attachments, the number of layers will be taken from the framebuffer object’s default layer count” | 存储/查询：`framebuffers.c` `GL_FRAMEBUFFER_DEFAULT_LAYERS`；渲染：`mglRenderPassFallbackRenderTargetForSize`（`MGLRenderer+RenderPass.m:89-118`）、绑定 fallback（同文件 `:3344-3362`，`layered=NO`） | **FIXED** | CTS no-attachment layered 用例会写 `gl_Layer` 选层。实现补一块 2D `BGRA8Unorm`、`array_length=1` 的 dummy color。`default_layers` 只被 Get/Set，渲染路径不读。Metal `renderTargetArrayLength` 也不会来自 default layers。 |
| 3 | 只写 `gl_Layer` 或只写 `gl_ViewportIndex` 时交叉别名（VS） | **GLSL 4.60 §7.1.4**：“If a geometry shader does not assign a value to gl_ViewportIndex, viewport transform and scissor rectangle zero will be used”；**GL 4.6 §9.8**：layer 为 0 if “the current geometry shader does not statically assign a value to … gl_Layer”；**GL 4.6 §13.8.1**：“If no geometry shader is active, or if the active geometry shader does not write to gl_ViewportIndex, the viewport numbered zero is used”。未写的那一个必须是 **0**，两内建相互独立。GLSL 4.60 §7.1.1 内建列表确实没有 VS `gl_Layer`/`gl_ViewportIndex`；本树（含回归测试，如 `test_regression/main.c` 的 “plain VS writing gl_Layer=1”）有意支持 VS 写 `gl_Layer`，接近 `ARB_shader_viewport_layer_array`。**VS 写 layer 本身不是不合规**；不合规只在别名。 | `assembleReturn`（`mgl_air_backend.cpp:5659-5671`，acac3c9）：`if (hasLayer && !hasViewport) viewportIndex = layer;` 反向亦然。注释误写 “§11.1.3.5/§11.1.3.6 tie layer and viewport index to the same value”。 | **FIXED** | 只写 `gl_Layer=3` 的 VS 会把 viewport 也设成 3。规格要求未写的 viewport 为 0。GS emit 路径（#19）不别名、默认 0，更接近规格。 |
| 4 | `layout(invocations=0)` / `invocations<=0` 未报编译错误 | **GLSL 4.60 §4.4.1.2**：“If a shader specifies an invocation count greater than the implementation-dependent maximum, or less than or equal to zero, a compile-time error results” | 解析：`mgl_glsl_parser.c:1536-1537` 写入值；`:1597` 仅当 `layout_invocations >= 1` 才拷到 TU。`0` 被丢掉，后端默认 1（`mgl_air_backend.cpp:7676-7677`）。sema 无范围检查。 | **FIXED** | 非法 layout 被当成 “未声明 → 1”，程序能链上并按 1 次 invocation 跑。 |
| 5 | 未声明 `max_vertices` / 输入或输出 primitive type：应链接失败，实现可能编译期失败或静默当 0 | **GL 4.6 §11.3.1**：“A program will fail to link if the input primitive type is not specified…”；**§11.3.2**：“A program will fail to link if either the output primitive type or maximum output vertex count are not specified…”；**GLSL 4.60 §4.4.2.2**：“At least one geometry shader (compilation unit) in a program must declare a maximum output vertex count” | TU `calloc` 后 `layout_max_vertices==0`（`mgl_glsl_parser.c:2087`）。后端拒绝条件是 `< 0 \|\| > 1024`（`mgl_air_backend.cpp:6732`），**0 通过**。缺输入 topology：`layout_primitive==0` 会在 `:6709-6720` 编译失败（规格允许该编译单元成功、链接失败）。缺输出 topology：`:6722-6730` 同样编译失败。 | **FIXED** | 缺 `max_vertices` 且从不 `EmitVertex` 的 GS 可编译+链接，`GEOMETRY_VERTICES_OUT` 为 0。缺 in/out layout 则以 `COMPILE_STATUS=FALSE` 而非 `LINK_STATUS=FALSE` 失败。 |
| 6 | 片段着色器 `in int gl_Layer` / `gl_ViewportIndex` | **GLSL 4.60 §7.1.5**：FS 内建包含 `in int gl_Layer;` / `in int gl_ViewportIndex;`。“The input variable gl_Layer is filled with the value written to the gl_Layer geometry shader output, if a geometry shader is present.” | sema：`check_expr`（`mgl_glsl_sema.c:1589-1594`）把 `gl_Layer` / `gl_ViewportIndex` 当 **通用内建**（`int`），无 stage 门，FS 引用 **不会** 变成未声明标识符。codegen：`emitExpr` `MGL_EXPR_VAR_REF`（`mgl_air_backend.cpp:3145-3153`）走 **out-variable read-back**（本次 invocation 最后一次写入；未写则 `i32 0`）。没有 FS 输入 lowering，也没有上一阶段的 layer 传入。 | **FIXED** | sema 接受这两个名字；codegen 从未把它们接成 FS 输入。FS 读到的是局部 0（或本 invocation 自己写过的值），不是前一阶段写出的 layer/viewport。 |
| 7 | `LAYER_PROVOKING_VERTEX` / `VIEWPORT_INDEX_PROVOKING_VERTEX` 不驱动选顶点 | **GL 4.6 §11.3.4.6**：“The vertex conventions followed for gl_Layer and gl_ViewportIndex may be determined by calling GetIntegerv with pnames LAYER_PROVOKING_VERTEX and VIEWPORT_INDEX_PROVOKING_VERTEX” | 查询：`get.c:867-868`，初值来自宿主 GL（`glm_params.c:354-355`）。GS→PTVS 与 VS 路径 **从不读取** 这两个状态；`glProvokingVertex` 只改 `GL_PROVOKING_VERTEX`。 | **FIXED** | 查询可能返回宿主值，但多顶点 primitive 的 layer/viewport 选择未按该约定实现。规格允许 `UNDEFINED_VERTEX`，但查询值与实际选择必须一致。 |
| 8 | 已链接但不走 compute route 的 GS 在 draw 时 `INVALID_OPERATION` | **GL 4.6 §11.3.4.7** 只规定 mode 与 GS 输入类型不匹配时 `INVALID_OPERATION`。成功链接的 GS 程序必须执行，不能另造 “unsupported” 错误。 | `handleGeometryDrawIfNeeded`（`MGLRenderer+DrawSupport.m:1615-1627`）：`gs_route != COMPUTE` 或没有 metallib → `GL_INVALID_OPERATION`。`program.c:1536-1537` 在链接成功后仍可标 `MGL_GS_ROUTE_UNSUPPORTED`。 | **FIXED** | 对应用表现为 “合法程序 draw 失败”。常见触发：`invocations` 越界被默认掉之后仍可能 route 成功；真正 unsupported 时应在 **链接** 失败。 |

### 较轻的不合规（不与 #1–#3 同级）

| # | 项 | 规格引用 | 代码位置 | 判定 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| 9 | `EndStreamPrimitive(stream)` 忽略 stream 参数 | **GL 4.6 §11.3.4.3**：“EndPrimitive and EndStreamPrimitive may be used to end the primitive being assembled on a given vertex stream”；**GLSL 4.60 §8.13**：`EndStreamPrimitive(int stream)` “Completes the current output primitive on stream stream” | `mgl_air_backend.cpp:3614-3650`：校验 stream 为常量后，只把 strip 计数器清 0，不按 stream 分状态。 | **FIXED**（较轻） | mismatch 真实存在。多 stream 且非 points 应在链接失败（#20），因此实际执行路径是 points-only；此时 `EndStreamPrimitive` 忽略 stream 对 strip 状态几乎无 GL 可见效果。不要把它和 #1–#3 的错误光栅化并列。 |

### 合规项

| # | 项 | 规格引用 | 代码位置 | 判定 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| 10 | `GetProgramiv(GEOMETRY_*)` 无 GS 时 | **GL 4.6 §7.14**：“An INVALID_OPERATION error is generated if GEOMETRY_VERTICES_OUT, GEOMETRY_INPUT_TYPE, GEOMETRY_OUTPUT_TYPE, or GEOMETRY_SHADER_INVOCATIONS are queried for a program which has not been linked successfully, or which does not contain objects to form a geometry shader.” | `program.c:2311-2321`：`!link_success \|\| !shader_slots[_GEOMETRY_SHADER]` → `GL_INVALID_OPERATION` | **COMPLIANT** | 对照：同文件 `TESS_*` 在无 TES 时返回 0（`:2352-2353`），规格同样要求 `INVALID_OPERATION`——相关但不在本次 GS 主范围。 |
| 11 | Pipeline 无 active VS 时 draw | **GL 4.6 §11.3**：“An INVALID_OPERATION error is generated by any command that transfers vertices to the GL if the current program state has a geometry shader but no vertex shader.”；**§11.1** 可编程顶点处理在无 VS 时结果未定义，但 pipeline 含图形阶段无 VS 是明确错误 | `validate_program`（`draw_buffers.c:479-496`）在 bound pipeline 无 VS 且存在 TCS/TES/GS/FS 时返回 false；`mglExecuteDrawCommand` 等路径 `ERROR_RETURN(GL_INVALID_OPERATION)`（`:1606-1608`）。链接：非 separable 缺 VS（`program.c:1446-1459`）链接失败。 | **COMPLIANT** | 抽查的 draw 入口都经过 `validate_program`。 |
| 12 | GS 输入 primitive 类型与 draw mode | **GL 4.6 §11.3.1 / §11.3.4.7**：POINTS↔POINTS；LINES↔LINES/STRIP/LOOP；LINES_ADJACENCY↔LINES_ADJACENCY/LINE_STRIP_ADJACENCY；TRIANGLES↔TRIANGLES/STRIP/FAN；TRIANGLES_ADJACENCY↔TRIANGLES_ADJACENCY/TRIANGLE_STRIP_ADJACENCY | `mglGeometryInputModeAccepts`（`MGLRenderer+DrawSupport.m:278-295`）；不匹配：`:1593-1613` `GL_INVALID_OPERATION` | **COMPLIANT** | |
| 13 | Table 10.1 `gl_in` 顺序（triangle strip + adjacency） | **GL 4.6 §10.1.14 Table 10.1**（1-based）：first 三角形 core 1,3,5 adj 2,7,4；only 三角形 adj 2/3=6；odd/even middle 与 last 行。规格允许 Table 10.1 **或** 10.2。 | `mglGeometryGatherTopology`（`MGLRenderer+DrawSupport.m:190-220`）。q=0 发出 0,1,2,6,4,3（0-based）= 1,2,3,7,5,4，对应 gl_in[0..5] = core1, adj1/2, core2, adj2/3, core3, adj3/1。`last` 时用 q+5 代替 q+6，对应 only/last 的顶点 6。 | **COMPLIANT** | 静态对照 Table 10.1 一致。未跑 CTS 所以边缘奇偶 last 行仍见 “Still unknown”。 |
| 14 | 非 adjacency strip/fan 分解 | **GL 4.6 §10.4 / §11.3.1**：strip/fan/loop 先分解成 list primitive 再进 GS；奇 triangle strip 交换前两顶点 | `MGLRenderer+DrawSupport.m:221-250` | **COMPLIANT** | |
| 15 | `EmitVertex` / `EndPrimitive` 基本语义 | **GLSL 4.60 §8.13**：EmitVertex 发出当前输出；“On return from this call, the values of output variables are undefined.”；超过 `max_vertices` 的 EmitStreamVertex 结果 undefined。EndPrimitive 结束当前 primitive，不发顶点。 | `emitGeometryVertex`（`mgl_air_backend.cpp:2533-2797`）：`emitCount < max_vertices` 才写；超额静默丢弃（undefined 的合法实现）。`EndPrimitive` 清 strip 计数器（`:3614+`）。Emit 后 **不** 清空 lvalues——规格说 undefined，保留旧值合法。points / line_strip / triangle_strip 展开与 ABI `mgl_air_gs_abi.h:138-158` 一致。 | **COMPLIANT** | |
| 16 | GS invocations 与 `gl_InvocationID` | **GL 4.6 §11.3.4.2**：“each input primitive spawns N invocations, numbered 0 through N−1”；后续阶段先收 invocation 0 的 primitive。**GLSL 4.60 §4.4.1.2**：未声明则每 primitive 一次。 | 工作项 `prim*N+inv`（`mgl_air_backend.cpp:7672-7685`）；dispatch `workItemCount = drawPrimitiveCount * invocationCount`（`DrawSupport.m:1721-1727`）。按 work item 0..n-1 间接 draw，顺序为 prim0-inv0, prim0-inv1, … | **COMPLIANT** | `layout(invocations=0)` 见 #4。上限 32 与 `gl_MaxGeometryShaderInvocations` 硬编码一致；是否等于 `glGet(MAX_GEOMETRY_SHADER_INVOCATIONS)` 未在本树交叉验证。 |
| 17 | `glFramebufferTexture` 把 array/cube/3D 整层挂成 layered | **GL 4.6 §9.2**：“If texture is the name of a three-dimensional texture, cube map array texture, cube map texture, one- or two-dimensional array texture, or two-dimensional multisample array texture, … the framebuffer attachment is considered layered.” | `mglFramebufferWholeTextureAttachmentIsLayered`（`framebuffers.c:1949-1961`）；`new_layered` 仅当 `attachment_type==GL_NONE`（即 `glFramebufferTexture` / named 变体）（`:2294-2318`）。`glFramebufferTextureLayer` 走非 layered（单层）。 | **COMPLIANT** | |
| 18 | `FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS` | **GL 4.6 §9.4.2**：“If any framebuffer attachment is layered, all populated attachments must be layered. Additionally, all populated color attachments must be from textures of the same target …” | `mglFramebufferHasLayerTargetMismatch`（`framebuffers.c:1393-1439`），`#ifdef MGL_GL_CORE`（Makefile 定义了该宏） | **COMPLIANT** | 混 layered + non-layered，或 layered color 的 `textarget` 不一致 → incomplete。 |
| 19 | GS 写出的 `gl_Layer` / `gl_ViewportIndex` 经 PTVS 送到 Metal `render_target_array_index` / `viewport_array_index` | **GL 4.6 §11.3.4.6**；GLSL 未写则 viewport 0、layer 0（见 #3） | GS 分槽存储 offset 40/44（`mgl_air_backend.cpp:2561-2592`）；PTVS（`RenderPass.m:853-865`）从同一 vec4 的 `.z`/`.w` 转发。未写则为 0，**不交叉别名**。Metal 属性在 `mgl_air_backend.cpp:9204-9216`。 | **COMPLIANT**（GS 路径） | 与 VS 路径 #3 不同。PTVS 用 `strstr(gs->src, "gl_Layer")` 决定是否声明这两个输出：源码提到但未赋值时仍声明，值为记录里的 0。 |
| 20 | 多 stream XFB：points-only、stream 0 光栅化、其它 stream 丢弃 | **GL 4.6 §11.3.4.3**：“Geometry shaders that emit vertices to multiple vertex streams are currently limited to using only the points output primitive type”；“Primitives emitted to all streams but stream zero are discarded after transform feedback.” | `EmitStreamVertex` 在 stream>0 时要求 points（`mgl_air_backend.cpp:3602-3606`）。stream>0 记录降序写入，不进 raster。 | **COMPLIANT**（设计） | 端到端正确性见 unknown。 |
| 21 | GS XFB：整 primitive 截断、内建 varying、数组 varying | **GL 4.6 §13.3.2**：“If recording the vertices of a primitive … would result in either exceeding the limits of any buffer object’s size … then no vertices of that primitive are recorded in any buffer object”；数组按元素顺序写；`gl_Position`/`gl_PointSize` 可捕获 | 链接计划 `mglValidateTransformFeedbackVaryings`（`program.c:840+`），GS 为 feedback stage。两遍：pass-1 可见性 + pass-2 `mgl_gs_xfb_scatter`（`mgl_air_gs_abi.h` §5b，`DrawSupport.m` prefix-sum/scatter）。内建 offset：`DrawSupport.m:2005-2018`。CPU 路径明确排除 GS（`draw_buffers.c:956-962`）。 | **COMPLIANT**（设计意图与代码结构） | 未执行 CTS，字节布局/截断见 unknown。 |
| 22 | `glFramebufferTexture` 主要错误码 | **GL 4.6 §9.2**：缺纹理对象 → `INVALID_VALUE`；buffer texture → `INVALID_OPERATION`；`COLOR_ATTACHMENTm` 越界 → `INVALID_OPERATION`；其它非法 attachment → `INVALID_ENUM`；零绑定到 target → `INVALID_OPERATION`；非法 level → `INVALID_VALUE` | `framebuffers.c:2093-2147`（越界 color / 缺对象 / buffer tex）；`:2185-2208`（level）；`:2284-2291`（无法解析 attachment，含默认 FBO） | **COMPLIANT**（`glFramebufferTexture` / named） | `glFramebufferTexture1D/2D/3D` 在对象不存在时仍可能 `newTexture` 占位（`:2117-2125`），与 “must name an existing texture object” 不符——相关、非本次主入口。 |
| 23 | Sticky Metal `renderTargetArrayLength` | **GL 4.6 §9.8**：“A framebuffer is considered to be layered if it is complete and all of its populated attachments are layered.” **§9.2**：“If the number of layers of each attachment are not all identical, rendering will be limited to the smallest number of layers of any attachment.” | `mglRenderPassArrayLength`（`mgl_render.cpp:13817-13843`）：只累积 `attachment.layered`，取 min；无 layered 则 1 或 0。`f55683f` 防止后挂的非 layered depth 把 length 写成 0。 | **COMPLIANT**（在 FBO 已 complete 且全部 layered 时） | commit 说明写 “once any attachment is layered the framebuffer is layered”——这不是 9.8 原文。混挂应走 #18 incomplete，draw 应 `INVALID_FRAMEBUFFER_OPERATION`。sticky 是 Metal 映射细节，对 complete layered FBO 的 min-layer 行为符合 §9.2。 |
| 24 | 无附件 FBO 完整性（宽高） | **GL 4.6 §9.4.2**：“There is at least one image attached … or the value of … DEFAULT_WIDTH and … DEFAULT_HEIGHT … are both non-zero.” | `framebuffers.c:1488-1492` | **COMPLIANT** | layered 层数见 #2，不在这条完整性规则里。 |

---

## 分项说明（优先级 1–5）

### 1. Geometry shaders

输入五类、输出三类、adjacency 窗口、strip 奇三角形交换、`EmitVertex` 上限、`GEOMETRY_*` 查询在有 GS 时取值：源码与 §11.3 / Table 10.1 对齐。

主要缺口是 **跳过 GS**（#1）、**layout 合法性**（#4、#5）、以及链接成功后 draw 再 `INVALID_OPERATION`（#8）。

`gl_in[]` 越界常量下标在 codegen 报错（`mgl_air_backend.cpp:1874+`），对应 §11.3.1 “program … will not link” 更接近编译失败，属于 #5 同类（失败阶段不对）。

### 2. Layered rendering

`glFramebufferTexture` 对 array/cube/3D/cube-array/2DMS-array 置 `layered`（#17）。混挂在 `MGL_GL_CORE` 下 incomplete（#18）。GS 写出的 layer/viewport 分槽进 Metal（#19）。

真正坏光栅化的是：无附件 layered FBO 的 2D dummy（#2），以及 VS **别名** layer↔viewport（#3；VS 写 `gl_Layer` 本身是有意扩展）。FS 读 `gl_Layer` 能通过 sema，但 codegen 是 out 回读，得到局部 0（#6）。`f55683f` 本身不是规格违规；它修的是 Metal array length 被非 layered attachment 清零。9.8 要求的是 **complete 且全部 populated attachment 都 layered** 才叫 layered framebuffer。

### 3. Passthrough GS vs 规格

规格要求：只要程序含 GS，该 GS 就是 active 的（§11.3），每次 invocation 都计入查询（§11.3.5），`EmitVertex` 发出当时的全部输出（GLSL §8.13）。

实现用 **Minecraft/CTS 源码形态** 当身份变换，而不是数据流分析。这不是规格条款，是可以改错像素和查询计数的优化。

### 4. Transform feedback + GS

设计是有序两遍多 stream（ABI 写明 GL 4.6 §13.3.2 整 primitive 截断）。CPU passthrough 捕获明确拒绝 GS。链接计划支持 `gl_NextBuffer`、`gl_SkipComponentsN`、`gl_Position`/`gl_PointSize`、数组元素名。

静态上看结构对准规格；**正确性依赖 scatter kernel 与 record 布局**，本审计未跑 CTS。

### 5. 相关错误码

- `GEOMETRY_*` 无 GS：`INVALID_OPERATION` — 合规。  
- Pipeline 无 VS：`INVALID_OPERATION` — 合规。  
- `glFramebufferTexture` 主错误码 — 大体合规。  
- 额外：`TESS_*` 无 TES 返回 0、`COMPUTE_WORK_GROUP_SIZE` 无 CS 返回 `{0,0,0}`，规格 §7.14 两者都要求 `INVALID_OPERATION`（代码注释与规格相反）。不在 GS 主范围，列为相关。

---

## Still unknown

1. Table 10.1 **last** 行（`i = n-1` 且 even/odd）与 gather 循环 `q + 5 < segmentCount` 的最后一次迭代是否对所有 `n` 都撞上 `last` 分支；规格还允许 Table 10.2，实现声称只实现 10.1。  
2. `gl_PrimitiveIDIn` 在 **instanced** draw 下是全局 primitive 计数还是 per-instance（§11.3.4.4：“The first primitive generated by a drawing command is numbered zero”）。  
3. GS XFB scatter 的分量对齐、整 primitive 截断、多 buffer 同一 stream 是否与 §13.3.2 逐字节一致。  
4. `GEOMETRY_SHADER_INVOCATIONS` 查询在 **非** passthrough 路径上是否等于 `primitiveCount * invocations`（含实例化）。  
5. 宿主 `getMacOSDefaults` 失败时 `LAYER_PROVOKING_VERTEX` 初值（可能为 0，非合法 enum）。  
6. VS 写 `gl_Layer` 时 Metal 是否要求 `inputPrimitiveTopology`（代码注释 vs 仅 GS/POINTS 设置 topology）——影响无 GS 的 layered VS 三角形。  
7. `max_vertices=0` 且含 `EmitVertex`：codegen 报 “EmitVertex requires the output ABI”，表现为编译失败而非链接失败。  
8. 无附件 FBO 上 `gl_Layer` 写入是否被 Metal 静默丢掉（#2 已从源码判定 dummy 2D；CTS 像素级结果未跑）。

---

## 本审计未改动的运行时结论

不要在本分支修以上 mismatch。优先（若后续单独修）按光栅化影响：

1. 删除或收紧 passthrough `strstr` bypass，或至少让查询仍计入 invocation。  
2. no-attachment FBO：`default_layers > 0` 时用 layered 纹理（或等价 Metal array length），不要 2D dummy。  
3. 去掉 VS `assembleReturn` 的 layer↔viewport 交叉赋值。  
4. `invocations<=0` 与缺 `max_vertices`/in/out layout 按 GLSL/GL 在编译或链接失败。
