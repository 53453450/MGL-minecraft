# C 层架构审查

日期：2026-09-13（正文）；**2026-09-21 续审见 §9**
对象：`MGL/src/*.c` + `*.cpp`（正文口径 127 文件 / 161,068 行；§9 重测 164 / 204,135）与 `MGL/include/*.h` + `MGL/src/*.h`
性质：只读架构审查，不含代码改动
配套：`docs/C_LAYER_CONSOLIDATION_TODO.md`（重复与消减）、
`docs/STATE_DATAFLOW_TODO.md`（状态数据流）、`docs/STATE_MACHINE_REVIEW_2026-09-13.md`（结构）

---

## 0. 结论

**C 层的问题不是"文件少"或"没分层"，而是：一个 1.9 万行的核心单体、一个 3,796 行/722 声明的
god header、以及 22% 的函数依赖非类型化的 `void *` 边界。
文件数（127）看起来像模块化，实际代码是双峰分布——73 个小文件只占 9.7%，21 个大文件占 69.2%。**

同时**有两处必须先纠正的误判**（本文第 3、4 节给了证据）：

- `void *` 大多**不是**设计失误。只有 9.3% 是回调表所需的泛型参数，
  其余是「每个文件自己的私有 ctx 结构」这一**有意的作用域技巧**——
  它让 `.c` 之间共享实现而无需在头文件里暴露类型。
  **真正该改的是别处**（第 5.2 节的 `owner` 抽象）。
- 「无调用者的 C 文件」绝大多数是**GL 分派表入口**（`non_core_unimplemented.c` 的 370 个导出
  **全部**被取地址入表），不是死码。

---

## 1. 规模与分布

| 指标 | 数值 |
|---|---|
| `.c` + `.cpp` 文件 | **127** |
| C 层总行数 | **161,068** |
| 头文件（`MGL/include/*.h`） | **143** / 23,132 行 |
| 最大文件 | `mgl_render.cpp` **19,445** |
| 次大 | `mgl_air_backend.cpp` 14,753 |
| 函数定义总数 | **3,313** |
| 函数体中位数 | **11 行** |

### 1.1 双峰分布（关键结构事实）

```
前   1 文件占   19445 行 =  12.1%
前   3 文件占   41766 行 =  25.9%
前   5 文件占   55881 行 =  34.7%
前  10 文件占   78187 行 =  48.5%
前  20 文件占  109373 行 =  67.9%
前  30 文件占  125759 行 =  78.1%

中位数 386 行 / 均值 1,268 行 / 最大 19,445
<500 行的文件: 73 个 (仅 15,607 行 = 9.7%)
>2000 行的文件: 21 个 (111,508 行 = 69.2%)
```

**即：127 这个数字具有误导性。** 真正承载逻辑的是 21 个文件，
其余 73 个是平均 214 行的域模块。这是"沉入"的典型产物形态——
ObjC 里的一行调用被替换成一个小的 C 域文件，但逻辑主体仍留在原地。

### 1.2 按前缀聚类的模块视图

（源文件由 Makefile wildcard 收集，无显式分组，故按前缀聚类）

| 簇 | 文件 | 行数 | 占比 |
|---|---|---|---|
| GL 对象/状态实现（`state/textures/buffers/framebuffers/uniforms/program/...`） | 16 | 37,933 | **23.6%** |
| Metal 后端（`mgl_render.cpp` / `mgl_renderer_backend.cpp` / ports） | 5 | 22,286 | **13.8%** |
| 未归类（见 §1.3） | 27 | 20,389 | 12.7% |
| AIR 编译器 | 9 | 20,272 | 12.6% |
| GL 入口/上下文 | 5 | 12,608 | 7.8% |
| 基础设施（`hash_table/pixel_utils/utils/error/draw_command/non_core_*`） | 6 | 11,465 | 7.1% |
| GLSL 前端（cpp/lex/parse/sema） | 4 | 10,591 | 6.6% |
| draw 编排（tess/GS/encode/issue） | 10 | 8,595 | 5.3% |
| **batch（deferred draw 批处理）** | **14** | **5,388** | **3.3%** |
| 纹理辅助 | 6 | 3,919 | 2.4% |
| binding 策略/阶段/纹理 | 5 | 2,661 | 1.7% |
| blit / readback / buffer / 顶点 / trace / tess / program | 17 | 4,565 | 2.8% |
| focus/lifecycle 杂项 | 1 | 96 | 0.1% |

### 1.3 "未归类" 27 个文件暴露了命名缺口

```
mgl_attachment_binding.c  mgl_aux_assets.c     mgl_byte_hash.c      mgl_capability.c
mgl_compile_artifact.c    mgl_coordinate.c     mgl_fake_draw_executor.c
mgl_frame_activity.c      mgl_frontend_session.c  mgl_gl_extensions.c
mgl_gpu_recovery.c        mgl_index_buffer.cpp  mgl_ir.c            mgl_legacy_compat.c
mgl_metal_ref.c           mgl_metallib_writer.cpp  mgl_pixel_format.c
mgl_pso_format_class.c    mgl_region_value.cpp  mgl_rt_sync.c       mgl_sampler_compat.c
mgl_shader_resource.c     mgl_state_compat.c   mgl_state_log.c     mgl_sync.c
mgl_thread_affinity.c     mgl_uniform_reflection.c
```

这不是分类失败，而是**命名不自解释**：`mgl_*` 前缀被同时用于
GL API 辅助、编译器产物、平台同步、格式分类、调试日志等互不相关的域。
没有 `mgl_<域>_` 的登记表，新增文件无法判断该归哪。

---

## 2. God header：`MGL/src/mgl_render.h`

| 指标 | 数值 |
|---|---|
| 行数 | **3,796** |
| 声明函数 | **722** |
| 被 include 的 TU | **55**（= C 层 116 个有导出的文件的 **47%**） |
| 位置 | `MGL/src/`（**不在** `MGL/include/`） |

首参类型分布（前 12）：

```
45  void *binding_state          30  uint32_t pixel_format
29  void *owner                  22  void *render_encoder
21  uint32_t type                21  GLMContext glm_ctx
21  uint32_t target              16  ...
15  void
11  void *render_encoder_owner   11  void *compute_encoder
11  void **owner                 10  void *command_buffer
```

- [x] **A1** 按域拆分。**2026-09-21 完成**（A1a/A1b/A1c，§9.6 / §9.14 / §9.10）：
      实现进域 TU，声明进 `mgl_render_api_*.h`，`mgl_render.cpp` 只留残余 `core`。
      当初点名的 binding / encoder / format 族已落在对应窄头与实现文件里。
- [x] **A2** 把 `mgl_render.h` 移出 `MGL/src/`。**2026-09-21 完成**：文件在 `MGL/include/mgl_render.h`。include 仍是 `"mgl_render.h"`（`-IMGL/include` 已在搜索路径上），fan-in 没变。窄头见 §9.10。
- **为什么这是首要问题**：55 个 TU（含 AIR 编译器、GLSL 前端、纯 C 状态机）
      都能看到 722 个渲染器内部符号。任何内部重构的改动面因此自动放大到 47% 的代码。

---

## 3. `void *` 边界的真实性质（**纠正我先前的判断**）

| 指标 | 数值 |
|---|---|
| 参数含 `void *` 的函数 | **738 / 3,313 = 22.3%** |
| 以 `void *` 作首参（伪 this） | 467 |
| **其中被赋给结构体字段（= 真回调/ops 表）** | **69** |
| 其余（直接调用） | **669** |
| `void *` 在各文件内部被转成的**不同类型** | 只有 **8 个 (文件,类型) 组合** |

按文件的 `void *` 密度：

```
358/742  48.2%  mgl_render.cpp
 29/31   93.5%  mgl_batch_dyn_bind_encode.c
 28/29   96.6%  mgl_batch_issue_encode.c
 27/97   27.8%  textures.c
 25/25  100.0%  mgl_draw_encode.cpp
 18/66   27.3%  buffers.c
 11/11  100.0%  mgl_batch_flush_restore_encode.c
  9/9   100.0%  mgl_batch_rt_mark_host.c
```

### 3.1 结论：大部分是**有意的作用域技巧**，不是失误

只有 8 个 `(文件, 类型)` 组合被藏在 `void *` 后面：

```
mgl_batch_dyn_bind_encode.c   MGLDynVertexCtx, MGLDynApplyCtx
mgl_batch_issue_encode.c      MGLIssueEncCtx
mgl_batch_icb_mdi_encode.c    MGLIcbMdiCtx
mgl_batch_restore_host.c      MGLKeyRestoreCtx
mgl_batch_rt_mark_host.c      MGLBatchRtMarkHostCtx
glm_context.c                 MGLRendererBackendHandle
rendering.c                   GLfloat
```

**即每个文件定义自己的私有 ctx 结构，用 `void *` 传递，永远不暴露在头文件里。**
这带来的实际好处是真实的：`mgl_batch_*` 那 14 个文件之间可以共享实现，
而不需要把 ctx 类型放进 `mgl_render.h`（否则 722 个声明会更膨胀）。

- [ ] **A3** 因此**不要**把 669 个"伪 this"当成待清理项。
      要区分的是：`mgl_render.cpp` 的 358 个属于**跨语言边界**
      （`void *renderer` 是 ObjC 侧 renderer 句柄，C 侧不该知道 ObjC 类型）——这是正确的。
- 但要注意代价：类型检查在这些边界上消失，正是
  `docs/STATE_DATAFLOW_TODO.md` P0-1（`active_state` 代理分裂）那类**靠约定维持**的问题
      得以存在的结构性原因之一。**方向是"补断言/补不变量"，不是"改类型"**（见 §6）。

---

## 4. C 层的连通性（**纠正"无调用者=死码"的误判**）

按"导出符号被谁调用"给 116 个有导出的 C 文件分类：

| 类别 | 数量 | 说明 |
|---|---|---|
| A. 被其它 C 文件调用 | **81** | 真正的 C 层内部实现 |
| B. **只被 `.m` 调用（C 是叶子）** | **20** | 见 §4.1 |
| C. 全库无调用者 | 15 | 见 §4.2，**绝大多数是误报** |

### 4.1 「C 是叶子、ObjC 是编排者」——20 个模块

```
43 导出  mgl_binding_texture.c        39 导出  mgl_binding_stage.c
12 导出  mgl_trace_strategy.c         11 导出  mgl_batch_flush_restore_encode.c
10 导出  mgl_render_pass_plan.c       10 导出  mgl_readback.c
 8 导出  mgl_binding_state_ops.c       6 导出  mgl_draw_support.c
 5 导出  mgl_blit_plan.c               4 导出  mgl_texture_readback_clear.c
 4 导出  mgl_blit_sampled_copy.c       3 导出  mgl_blit_clip.c
 3 导出  mgl_buffer_query.c            3 导出  mgl_coordinate.c
 3 导出  mgl_thread_affinity.c         3 导出  mgl_vertex_layout.c
 2 导出  mgl_vertex_attrib_plan.c      2 导出  mgl_gpu_recovery.c
 2 导出  mgl_attachment_binding.c      2 导出  mgl_focus_program.c
```

**已逐条核实**：`mgl_blit_plan.c` 的全部 5 个导出**只被 `MGLRenderer+Blit.m` 调用**
（`:831 / :842 / :849 / :856 / :861`），没有任何 C 调用者。

> 这是本轮审查最重要的架构发现：
> **沉入产出的是一批"C 叶子"，而不是"C 子系统"。**
> 计算进了 C，但**编排仍在 ObjC**——所以 ObjC 变薄、C 变厚，
> 而**结构复杂度没有下降**（该被消除的那一层控制流还在 `.m` 里）。
> 这也解释了为什么 C 层在壮大却查不到 ObjC 重复
> （见 `docs/C_LAYER_CONSOLIDATION_TODO.md` §6）。

- [x] **A4** 对这 20 个模块逐个判定编排归属。**2026-09-21 重测：仓库内 `*.m` = 0，
      20/20 都已有 `.c`/`.cpp` 调用者**（§9.1）。「编排仍在 ObjC」这条在沉入完成后作废，
      不再要求给模块头加「ObjC 叶子」注释。

### 4.2 "无调用者"15 个：**绝大多数是误报**

**逐条核实结果**：`non_core_unimplemented.c` 的 **370 个导出全部被取地址入 GL 分派表**；
`samplers.c` 21 个里 16 个入表；`fence.c` 16 个里 10 个入表；`compute.c` 3 个里 2 个入表。

`mgl_draw_gs.cpp` 的 `mglDrawGsRunDraw` 也有真实调用点
（`MGL/src/mgl_draw_metal_port.m:1613`，我最初的 grep 因为排除了同名注释行而漏判）。

- **结论：这 15 个不是死码清单**，是一份"需要按取地址/入口语义单独复核"的清单。
      本审查的静态方法（按 `name(` 调用点判定）**无法识别经由分派表和 ops 表注册的入口**，
      这是该方法的已知边界。要得到可信的死码清单，需要用链接期符号引用
      （`-Wl,-dead_strip` 试跑，或 `nm -u` 交叉比对），而不是文本 grep。

---

## 5. 结构性失衡

### 5.1 复杂度集中在 3.3% 的代码里

| 簇 | 代码占比 | 承载的复杂度 |
|---|---|---|
| GL 对象/状态实现 | 23.6% | 状态读写 |
| Metal 后端 | 13.8% | 后端细节 |
| **batch（deferred draw）** | **3.3%** | **批合并正确性、snapshot、dirty 域、replay 编排** |

`docs/STATE_DATAFLOW_TODO.md` 列出的 **P0 级缺陷（T0–T4）全部落在这 3.3% 里**
（合计 5,388 行 / 14 文件）。这是典型的**复杂度与代码量不成比例**：
最需要隔离、最需要不变量保护的部分，恰好是文件最小、最容易被"顺手塞进去"的部分。

- [x] **A5** 给 batch 簇加结构性约束。**已落地**（§6.7）：公共面收进
      `mgl_batch_public.h`，I1–I8 由 `scripts/state_machine_invariants.py` 执行。
      正文写「被 `.m` 调」时 ObjC 还在；§9.1 重测后编排已在 C/C++。

### 5.2 类型化不足的少数派：`owner` 抽象

`mgl_render.h` 里 `void *binding_state`(45) + `void *owner`(29) + `void **owner`(11)
= **85 个声明**用 `void *` 传递一个**本该是具体 C 类型**的东西
（`MGLBindingState` 之类），而 `GLMContext` 在同一个头里是**有类型的**（21 个声明）。

- [x] **A6** 把 `binding_state` / `owner` 系列具体化。
      2026-09-21 重测曾是 54 + 82 + 28。门面和 `mgl_render_api_*.h` 里这三项现为 0（§9.12）。
      **与 A3 不冲突**：A3 是私有 ctx；这里是跨文件公共抽象。

---

## 6. 分层检查

### 6.1 通过项

| 检查 | 结果 |
|---|---|
| C/C++ 文件 include ObjC 私有头 | **仅 1 处**：`glm_context.c:51` `#include "MGLRenderer.h"` |
| C 文件 import 任何 ObjC 头 | 同上，仅 1 个文件 |

- [x] **A7** `#include "MGLRenderer.h"` 已从 `glm_context.c` 移除（§9.2 / §9.15）。
      `mtlPixelFormatForGLFormatType` 走已有的 `mgl_glfw_abi.h`；
      `CppCreateMGLRenderer*` 声明在纯 C 头 `mgl_platform_shell_result.h`。

### 6.2 唯一的头文件环 —— **已断（A8 完成）**

原环（185 个头文件里唯一一个）：

```
mgl_pso_format_class.h:33 --(1)--> glm_context.h --(2)--> mgl_renderer_backend.h
        --(3)--> mgl_render.h --(4)--> mgl_pso_format_class.h
```

**根因**：`mgl_pso_format_class.h` 只需要 `Program` 这一个**不透明指针**
（全文件仅 `mglRenderVSWritesLayer(const Program *)` 用到），却为它 include 了整个
`glm_context.h`。仓库已有 `struct Program_t;` 前置声明的惯例
（`mgl_batch_replay.h` / `mgl_buffer_slots.h`）。

**已落地**：把边 (1) 换成前置声明。

```c
/* mgl_pso_format_class.h */
struct Program_t;
typedef struct Program_t Program;
```

**验证**：185 个头文件，**环 0 个**（断环前 1 个）；`make lib` 干净（0 error）；
10 个门禁全绿。

#### 6.2.1 断环暴露的真实问题：`mgl_render.h` 自身不自洽

先前我实测"直接删边会破 16 个 TU"，所以计划是"先补窄声明再删"。实际断环后报 7 个失败，
其中 **2 个是我方法的假阳性**：`gl_core.c` / `gl_es.c` 报 `no member named 'dispatch'`，
但这两个文件只在 `-DMGL_GL_CORE` / `-DMGL_GL_ES` 下编译，我最初的语法检查没带这些宏。
**带上真实宏后 0 错。**

真正破的是 **5 个 TU**：

| TU | 怎么拿到 `mgl_render.h` |
|---|---|
| `mgl_capability.c` | 直接 include |
| `mgl_vertex_layout.c` | 直接 include |
| `mgl_pipeline_cache_path.c` | 经 `mgl_pipeline_cache_path.h` 间接 |
| `mgl_render_pass_plan.c` | 经 `mgl_render_pass_plan.h` 间接 |
| `mgl_tess_compute_ops.c` | 经 `mgl_renderer_ports.h` 间接 |

**根因不是这 5 个文件写错了，而是 `mgl_render.h` 自己不自洽**：
它用了 3 处 `GLuint`（`mglCurrentRenderProgramKey`、`mglRestoreProgramPipelinePair` 等），
却从不 include GL 注册表 —— 一直靠 `glm_context.h → mgl_renderer_backend.h` 传递供给。

**修法**：让门面自给自足，而不是给 5 个 TU 各补一行。

```c
/* mgl_render.h */
#include <GL/glcorearb.h>   /* GLuint/GLenum/GLboolean，供其自身声明使用 */
```

**验证**：140 个 `.c`（带真实宏）**0 error**。

> 这正是 §6.3 那条边的"载荷"性质：它不是类型依赖，而是**一批 TU 的隐含 ABI 供给线**。
> 断环的正确做法不是删边，而是**让每个头对自己用到的类型负责** ——
> 这次通过让 `mgl_render.h` 对 `GLuint` 负责，一行修好了 5 个 TU。

#### 6.2.2 A10 重复声明去重：**44 → 0 组（完成）**

原状：**44 组函数在 ≥2 个头文件里声明，且逐字相同**。`mgl_render.h` 却不 include
任何被它重复的头——所以去重会破那些靠它拿声明的 TU。

**实测驱动的做法**：先删**全部**重复声明，再用"带真实宏（`-DMGL_GL_CORE`/`-DMGL_GL_ES`）
的 140 个 `.c` 全量语法检查 + `make lib`"量破面，按破面决定归属。分四个批次：

| 批次 | 删除 | 破的 TU | 处理 |
|---|---|---|---|
| 第一批（vs `mgl_render.h`） | 22 条 | 20 个 | 见下 |
| 第二批（窄头 vs `mgl_render.h`） | 9 条 | 0 个 | 直接完成 |
| 第三批（纯 C 头之间） | 3 条 | — | 单一来源化 |
| 第四批（C 头 vs ObjC 私有头） | 10 条 | 1 个 | 见下 |

**第一批的 20 个失败几乎全集中在一个函数**：`mglResolveProgramForStageFromState`
——它的唯一声明在 `MGLRenderer+RenderPass_Private.h`（**C 不能 include 的 ObjC 私有头**），
所以 4 个 C 文件各自写了局部 `extern`。同类还有 `mglCurrentRenderProgramKey`、
`mglRendererSyncFramebufferBindingNames`。

**根因不是调用方写错，而是声明的归属错了**：这三个函数的**定义都在 C 里**
（`mgl_renderer_entries.c:601,672,684`、`mgl_draw_support.c:40`），
只有声明被留在了 ObjC 私有头里。

**判据（已固化）**：**定义在哪个语言里，声明就归哪一侧的头。**

第三、四批按此判据处理，其中第三批把错误 API 也单一来源化了：

```
mglDispatchError  -> glm_context.h（它的 ERROR_RETURN* 宏与 19 个调用方 TU 依赖它；
                     且 error.h 已 include glm_context.h，反向会成环）
mglGetError       -> glm_context.h（原在 error.h + mgl.h 各一份）
```

顺带修了一个名实不符：`glm_context.h` 的 `mglDispatchError` 形参名是 `type`，
而定义（`error.c:130`）与 `error.h` 都是 `error`——C 里形参名不影响签名，所以能编，
但文档上自相矛盾。已统一为 `error`。

**结果：44 → 0 组重复声明**；`make lib` 0 error；10 个门禁全绿。
（独立口径复验：`mglGetError` / `mglDispatchError` / `mglRestoreProgramPipelinePair` /
`mglWriteProgramMSLDump` / `mglNoteBufferEncoded` / `mglTraceShouldLogReplay` /
`mglRendererGetValidatedVAO` / `mglMarkTextureLevelRenderTargetWrittenImpl`
各自都只剩 1 处声明。）

#### 6.2.2.1 一个重要更正：原先判为"有意重复"的 11 组，其实也是归属错误

上一轮我写的是"11 组 C 头 ↔ ObjC 私有头是**有意的隔离手段，不该去重**"。
这个判断**不准确**。`mgl_renderer_host.h:30-33` 的注释暴露了真相：

```c
/* The three BOOL-returning symbols ... keep their ObjC-header declarations;
 * the C TUs that need them restate them locally as `signed char` (rule 26),
 * so they are not declared here to avoid clashing with
 * MGLRenderer+Draw_Private.h when the .m includes this header. */
```

即"避免冲突"是**症状**——冲突的来源就是两处都有声明。正确做法不是维持重复，
而是**选一个所有者**（按上面的判据）**让另一侧让路**。第四批 10 条全部按此处理，
构建 0 error，说明私有头并不真的需要自己那一份。

#### 6.2.3 A12 类型层平台依赖：3 处多余已删

实测各头对 mach 类型的**真实使用**：

| 头 | 用了什么 | 处理 |
|---|---|---|
| `mgl_types_buffer.h:35` | `vm_address_t buffer_data`（真用） | 保留 |
| `mgl_texture_debug.h` | `mach_timebase_info_data_t` / `mach_absolute_time`（真用，但来自 `mach/mach_time.h`） | 删 `vm_types.h`（多余） |
| `mgl_texture_transfer.h` | **无** | 删（多余） |
| `glm_context.h` | **无** | 删（多余） |

**结果**：3 处多余 `vm_types.h` 已删，140 个 `.c` 0 error，构建与门禁全绿。
类型层里现在只剩 `mgl_types_buffer.h` 的 `vm_address_t` 一处真实平台依赖。

#### A12 的副产品：`mgl_draw_mode.h` 原本不是 C 兼容的

去 `vm_types.h` 之后暴露出来的：`mgl_draw_mode.h` 声明着 `extern "C"` 接口，
却含 `#include <objc/objc.h>`（为 `BOOL`）且两处 inline 返回 `BOOL`、用 `NO` / `YES`。
它被 `mgl_draw_support.c` 引用，所以**这个头本来就无法在纯 C 下编译**
（之前靠传递包含侥幸通过）。

**已改**：`BOOL` → `bool`、`NO`/`YES` → `false`/`true`、删掉 `objc/objc.h`。
另修了它里面的 `NSUInteger indexCount` → `uint64_t`（`mglPrimitiveModeHasDrawableSegment`）。

> 这与 §6.2.1 是同一类问题：**头文件对自己使用的类型不负责**，
> 靠传递包含侥幸工作，一旦上游 include 变化就暴露。

#### 6.2.4 M2 快照去重的复核（含一个已修的泄漏）

M2（按 key 去重 snapshot）的实现已在工作区落地：`snapshot_shared` 字段、
`mglInitializeOrShareBatchStateSnapshot()`、新的 `g_mglSnapshotShareHitsSinceSwap` 计数器、
以及两阶段释放。复核结论：

**设计是对的**，几处关键点都处理到了：

| 不变量 | 实现 | 复核 |
|---|---|---|
| 只有非 stream-merged 批次能当 donor | `if (… \|\| donor->stream_merged \|\| !donor->state_snapshot) continue;` | ✅ stream-merged 的 snapshot 带特化指针（patched buffer/element），不能共享 |
| donor 必须在**更早的索引** | 扫描范围是 `0..cb->batch_count`，而调用点在 `cb->batch_count++` 之前 | ✅ 结构上保证 donor < sharer |
| 两阶段释放（引用先、内存后） | `mglResetCommandBufferForContext` 两个 for 循环 | ✅ donor 在低索引、sharer 在高索引；先全部降引用再统一释放，sharer 不会读到已释放的 donor snapshot |
| `snapshot_shared` 不会残留到复用的槽位 | 5 个 `mglReleaseBatch` 调用点**后面都紧跟** `memset`；另有 2 处 memset 覆盖新槽位 | ✅ 逐点核对 |
| sharer 自己 retain 程序/缓冲引用 | 共享分支里调 `mglRetainBatchProgramReferences` / `mglRetainBatchBufferReferences` | ✅ 引用从 live state 重新取，不依赖 donor 的引用 |

**发现并修复的一处真泄漏**：`mglReleaseBatchMemory` 的 `snapshot_shared` 分支
只把 `state_snapshot` / `vao_snapshot` 置 NULL（正确——donor 拥有），
**但把 `commands` 的释放一起吞掉了**。`commands` 不是快照，它是批次自己的
`realloc` 数组（`draw_command.c ~4351`），两种模式下都由批次自己拥有：

```c
if (batch->snapshot_shared) {
    batch->state_snapshot = NULL;
    batch->vao_snapshot = NULL;
    if (!batch->arena_managed && batch->commands) {   /* ← 原来缺失 */
        free(batch->commands);
    }
    batch->commands = NULL;
} else if (!batch->arena_managed) { … }
```

**后果**：非 arena 路径下（`MGL_ARENA_SNAPSHOT=0`）每个共享批次漏一个 commands 数组。
arena 路径无影响（arena 整体 reset）。

**同时把 I7 检查从"窗口启发式"改成"分支契约"**：原版只要 `state_snapshot` 附近
600 字符内出现 `snapshot_shared` 字样就放行——所以它**在泄漏存在时仍报 OK**。
现在 I7 正向扫描每个 `if`、取真正包含该 free 的最内层条件，
要求其条件同时排除 `snapshot_shared` 与 `arena_managed`；
另加一条结构检查：`snapshot_shared` 分支必须处理 `commands`。

**反证测试**：往共享分支注入一句未加保护的 `free(batch->state_snapshot);`
→ 审计器 `exit=1` 并精确报出行号；还原后 `exit=0`。

#### 6.2.5 M2 已用专用测试验证（含两个会让后来者踩空的坑）

新增 `test_legacy_compat/test_state_snapshot_share.c` + `make test-state-snapshot-share`
（已挂进 `test-all`），两种内存模式（arena 默认 / `MGL_ARENA_SNAPSHOT=0`）都跑。实测结果：

```
baseline (无重复 key):  share_hits=0
revisit  (A,B,A):       share_hits=1  captures=2  bytes=129504
  ok: no share hit when no key repeats
  ok: revisited key produces a share hit
  ok: the third draw reuses a snapshot (2 captures, not 3)
  ok: shared-snapshot replay still renders (centre is green)
```

批次级证据（程序身份 A/B/A，第三个批次确实挂在第一个的快照上）：

```
batch[0] prog=3 snap=0x…8018 shared=0
batch[1] prog=6 snap=0x…5b08 shared=0
batch[2] prog=3 snap=0x…8018 shared=1   ← 复用 batch[0]
```

**代价**：`captures=2` 而非 3，省下 **129,504 字节/次命中**（= `mglSnapshotHotStateBytes()`
61,240 + `sizeof(VertexArray)` 3,512 = 64,752 的申请量口径；`bytes` 计数按实际写入计算）。

**这两个坑值得写下来，因为构建测试时都踩到了：**

**坑 1：`MGL_PERF_ADD` 被 `mglPerfSummaryEnabled()` 门控。**
不开 `MGL_PERF_SUMMARY=1` 时**所有** `g_mgl*SinceSwap` 计数器恒为 0：

```c
#define MGL_PERF_ADD(var, value) \
    do { if (mglPerfSummaryEnabled()) MGL_FRAME_ADD((var), (value)); } while (0)
```

我第一版测试读到 `captures=0 / share_hits=0`，一度以为"批次建了却没走快照捕获"，
追了分配点、计数器重置点、`draw_defer_enabled` 才定位到这一条。
**任何读这些计数器的测试都必须自带 `setenv("MGL_PERF_SUMMARY", "1", 1)`。**

**坑 2：M2 明确排除 stream-merged 批次，而"朴素 A,B,A"会全部落进 stream-merge。**

`mglInitializeOrShareBatchStateSnapshot` 的 donor 扫描里写着
`donor->stream_merged … continue`，而 stream-merge 的初始化
（`mglInitializeStreamMergedBatch`）是**另一个函数**、不走共享逻辑。
所以一串小的 `glDrawArrays` 会被全部 stream-merge 掉，测试打到的是**排除路径**而不是特性：

```
DEBUG batch[0] vao=1 prog=3 stream=1 shared=0   ← stream=1，永远不共享
DEBUG batch[1] vao=2 prog=3 stream=1 shared=0
DEBUG batch[2] vao=1 prog=3 stream=1 shared=0
```

理由（代码注释里也有）：stream-merged 的快照带**批次特有的指针补丁**
（临时 stream buffer 被写进冻结的 VAO），与普通捕获不可互换。

所以测试必须 `setenv("MGL_DISABLE_STREAM_MERGE", "1", 1)`。

**附带发现：靠 VAO 制造 key 差异是无效的。** 我先用"交替绑两个 VAO"来造 A/B/A，
结果**三个 draw 全并成一个批次**（`batch_count=1`）——因为 VAO 差异被
`keysEqualIgnoringDynamicBindings` 那条 per-draw 动态绑定覆盖路径吸收了。
**要制造真实的 key 差异，必须换程序（`program_name` 是 key 的一部分且无法被覆盖吸收）。**
这条对以后写批处理测试都有用。

### 6.3 上游真因：`glm_context.h` 才是把后端拉进 88 个 TU 的那条边

后台审查的测量（我没重复，采信其数字并标注为二手）：

- `glm_context.h` / `mgl_renderer_backend.h` / `mgl_render.h` 三者的**传递可达集完全相同**
  （各 88 个 TU，差集为空）；
- 仅删 `glm_context.h:96` 这一条边，`mgl_render.h` 的可达集从 **88 → 48 TU**，
  `mgl_renderer_backend.h` 从 88 → 7。

即 **`mgl_render.h` 的直接 include 只是表层；
真正把它扩散到 88 个 TU（C 层的 69%）的是 `glm_context.h:96`。**

> **数字已过期，方向未变（实测刷新）**：那批 88 / 43 的测量之后，重构继续推进，
> 现在 **`mgl_render.h` 直接 include = 72 TU**（原 43）、
> `glm_context.h` 直接 include = 32（原 27）、
> `mgl_pso_format_class.h` = 7、`mgl_render.h` 现 3,803 行。
> 直接 include 的比例上升说明"靠 `glm_context.h` 传递"这条路径在变弱——
> **方向是好的**，但 72 个直接 include 意味着门面本身仍是最大的收敛目标（A1）。
> §6.2 的断环不影响这 72 个（它们本来就是直接 include）。

**但我用 6.2 的实验证明了：单独删这一条边同样会破编译**（那 16 个失败里，
多数同时依赖这两条路径）。所以 A1（拆 `mgl_render.h`）与 A8 是**同一件事**，
必须一起做，且顺序是「先补窄声明 → 再删边 → 最后才谈拆」。

### 6.4 其余分层事实（后台审查，已抽查相符）

| 项 | 数值 |
|---|---|
| 指向 BACKEND 的层次越界边 | **59** 条（51→`mgl_render.h`，8→`mgl_renderer_backend.h`） |
| 其中 GL API 面 → 后端 | 4 条：`buffers.c:46`、`textures.c:57`、`program.c:56`、`tex_param.c:22` |
| 其中公共头 → 后端 | 10 条（含 `glm_context.h:96`、`mgl_vertex_layout.h:16`、`mgl_command_state.h:25`…） |
| 反向 include 路径（公共头 → `MGL/src`） | 1 条：`MGLRenderer_Private.h:59` → `mgl_safety.h` |
| 域实现 → AIR 层 | 1 条：`mgl_vertex_format.c:17` → `mgl_air_loader.h` |
| 库内 include `external/` 头 | **0**（`external/` 只服务测试与工具） |

**重复声明：40 个函数在 ≥2 个头里声明，且逐字相同**（抽查相符，例：
`sizeForFormatType` 同时在 `glm_context.h:218` / `MGLContext.h:30` / `pixel_utils.h:46`；
`mglRenderVertexAttribComponentSize` 同时在 `mgl_vertex_format.h:64` 与 `mgl_render.h:1073`）。
其中 25 组涉及 `mgl_render.h`，而 `mgl_render.h` **不 include** 那 18 个被重复声明的头。
C 允许这样，所以今天能编；**风险是漂移**——改一份、另一份静默分叉。

- [ ] **A10** 每对重复声明**保留窄头那一份**，删 `mgl_render.h` 里的副本
      （与 A1 同向，且能顺带缩小 A1 的改动面）。

**死头文件 4 个（我独立复核，均 0 引用）**：
`mgl_spirv_compile.h`（9 行纯 license，无实现文件）、`MGLBindingSync.h`、
`MGLQueryManager.h`、`enums.h`（其 `get_enum` 只有声明，**无定义无调用**）。

- [ ] **A11** 删除这 4 个。

### 6.5 C 层对 ObjC 私有头的纪律：**干净，且是被刻意维护的**

后台审查给"0 命中"，我复核后**修正其结论**：`*.c`/`*.cpp` 里确实有 8 处提到
`*_Private.h`，但**全部是注释**，没有一处 `#include`。

更值得注意的是这些注释本身：

```
mgl_draw_support.c:33   "Objective-C MGLRenderer+Draw_Private.h, which C cannot include."
mgl_vertex_layout.c:29  "MGLRenderer+VertexLayout_Private.h, which C cannot include."
mgl_renderer_ports.c:126 "MGLRenderer+Draw_Private.h / MGLRenderer+DrawSupportUtil.h, which are..."
```

**即" C 不能 include ObjC 私有头"是一条被显式认知并靠注释维持的边界**，
绕过的办法是把需要的内联/常量**复制一份到 C**（`mgl_state_log.c:81`
"从 ObjC static inline 搬来"、`mgl_blit_sampled_copy.c:89` "Local mirror of …"）。
这是**可接受但有代价**的形态：边界干净，代价是 6.4 里的重复声明与本地镜像。

### 6.6 类型层的触达面（可重建性）

`mgl_types_*.h` + `glm_limits.h` / `glm_params.h` 触达 **88–92 / 127 TU**：
任何一处编辑要重建约 **72%** 的库。类型层本身的纯度良好
（唯一越界是 `mgl_types_buffer.h:36` 与 `mgl_types_sync.h:45` 引 `glm_params.h`，
而后者自身是干净的数据层；另有 `mgl_types_buffer.h:35` 引入 `<mach/vm_types.h>`）。

- [ ] **A12** `mgl_types_*.h` 的平台依赖（`<mach/vm_types.h>`）应收敛到一处，
      否则类型层无法在非 Darwin 上复用。

---

### 6.7 batch 簇的入口面与不变量（A5 的前半，已落地）

#### 6.7.1 入口面实测：只有 3 个函数被外部调用

状态机簇 **19 文件 / 10,669 行**（`STATE_DATAFLOW_TODO.md` 拆分口径），
头文件里共声明 **193 个函数**。但按实际调用方统计：

| 函数 | 状态机簇外的调用点 | 性质 |
|---|---|---|
| `mglFlushCommandBuffer` | **65** | 公共入口 |
| `mglFlushPendingDraws` | **47** | 公共入口 |
| `mglRecordDrawCommand` | **4** | 公共入口 |
| `mglResetCommandBufferForContext` | 1 | 生命周期 |
| `mglAppendDrawCommand` / `mglComputeStateKey` / `mglStateKeysEqual` / `mglBatchRestoreStateForBatch` | **0** | 内部 |

另 157 个函数声明在 `mgl_batch_*.h`（`mgl_batch_issue.h` 43、`mgl_batch_replay.h` 40、
`mgl_batch_rt_mark.h` 28、`mgl_batch_mtl_encode.h` 23、`mgl_batch_restore.h` 15 …）。

> **结论修正（两处）**：
>
> ① 状态机的问题不是"入口面太宽"，而是**缺一道墙** —— 157 个内部机械的声明对整层可见，
>    任何 TU 都能直接插手批次内部状态。这与 §5.1 的"3.3% 代码承载最多复杂度"是同一件事的两面。
>
> ② 上表那 3 个是**直接调用**的入口，但真正的跨边界面更大：按"簇外 TU 实际引用了哪些
>    batch 符号"普查，是 **9 个**（见 §6.7.3）—— 另外 6 个没有直接调用点计数是因为
>    它们经 `mgl_batch_*.h` 声明被调用（如 `mglBatchBindActiveTexturesToMTL`、
>    `mgl_batch_icb_support_indirect_command_buffers`）。
>    **教训：按"调用点计数"估公共面会低估，必须按"符号引用"普查。**

#### 6.7.2 不变量已固化为可执行检查

新增 `scripts/state_machine_invariants.py`（`make test-state-invariants`，已挂进 `test-all`）。
7 项检查，当前全绿：

| 检查 | 内容 | 当前结果 |
|---|---|---|
| I1 | `MGLStateKey` 每个字段都被 `mglComputeStateKey` 写入 | ✅ 16/16 |
| I2 | key 内部无隐式 padding 落在 `memcmp`（M0 已修） | ✅ 96 字节无洞 |
| I3 | `mglCopyHotStateFields` 的三个 region `_Static_assert` 在位 | ✅ |
| I4 | `DIRTY_*` 单一来源，无手抄 `= 1u << N` 枚举（M0 已修） | ✅ |
| I5 | `cb->batch_count` 只由 recorder 改动 | ✅ 无外部改动 |
| I6 | `active_state` 只经双代理 helper **重绑** | ✅ 无手工重绑 |
| I7 | 共享 snapshot 不被借用方 free（M2 相关） | ✅ |

**I6 的语义区分值得记录**（我第一版检查在这里报了假阳性）：
同一个语法有两种含义——
- **重绑**：`...active_state = &ctx->state` / `= NULL` → 这是 helper 的职责，
  手工做会 `MGL_STATE()` 与渲染器缓存指针失配（P0-1 的机制）。**判失败。**
- **重同步**：`core->activeState = ctx->current_active_state` → 上下文已在 replay workspace 上，
  这是**修复渲染器缓存副本**。注意 `mglCoreRestoreLiveActiveState` 会**同时重绑到 live 并把缓存置 NULL**，
  所以这里**不能**调它。**判记录、不判失败。**

当前有 2 处属后者（`mgl_batch_flush_restore_encode.c:64`、`:295`），已写入 baseline。

**反证测试**：向 `draw_command.c` 注入一句 `ctx->active_state = &ctx->state;`
→ 审计器 `exit=1` 并精确报出行号；还原后 `exit=0`。即门禁真的会拦。

#### 6.7.3 A5 的墙：已完成（`mgl_batch_public.h` + I8 检查）

新增 `MGL/include/mgl_batch_public.h`，把"簇外允许调用什么"写成显式契约。
公共面是**普查得出**的，不是凭意图划的 —— 用脚本扫出簇外 TU 实际引用的 batch 符号：

```
batch 头共声明 157 个符号；被簇外 TU 引用的只有 9 个：
  mglBatchFlushBegin                               mgl_platform_shell.cpp
  mglBatchFlushRunBatches                          mgl_platform_shell.cpp
  mglBatchTeardownReplay                           mgl_platform_shell.cpp
  mglBatchRecordArrayDrawSubmitted                 mgl_draw_metal_port.c
  mglBatchRecordElementDrawSubmitted               mgl_draw_metal_port.c
  mglBatchBindActiveTexturesToMTL                  mgl_binding_state_ops.c
                                                   mgl_render_pass_manager_ops.c
  mgl_batch_icb_support_indirect_command_buffers   mgl_air_loader.cpp
                                                   mgl_blit_pipelines.c
  mgl_batch_mtl_create_icb                         mgl_platform_shell.cpp
  mgl_batch_replay_fill_sampler_params             mgl_renderer_ports.c
```

**做法**：把这 9 个的声明集中到公共头（含 `MGLBatchFlushPass` 类型 —— 调用方
`mgl_platform_shell.cpp:1661` **栈上分配**它，所以它是接口类型不是簇内部类型），
从 6 个内部头里删掉，内部头改为 include 公共头；7 个外部 TU 改用公共头。

**判据**（写进公共头注释）：不改变现有调用点（9 个都是真实调用），
目的是让**下一个**跨边界调用成为一次有意识的决定而不是意外。

**验证**：`make lib` 0 error；140 个 `.c` 语法 0 错；11 个门禁全绿。

#### 6.7.4 顺带修掉一个真实的类型错误

`mgl_batch_issue.h` 曾把 `mglBatchBindActiveTexturesToMTL` 声明为返回 **`bool`**，
而定义（`mgl_batch_replay.cpp:1194`）返回 **`int`**：

```c
/* mgl_batch_issue.h（旧） */  bool mglBatchBindActiveTexturesToMTL(void *renderer, GLMContext glm_ctx);
/* mgl_batch_replay.cpp:1194 */ extern "C" int mglBatchBindActiveTexturesToMTL(void *renderer, GLMContext glm_ctx)
```

**没有任何编译器报错**，因为所有调用点都把它写在布尔上下文里（`if (...)`、
`RETURN_FALSE_ON_FAILURE(...)`）。搬到公共头时才暴露。新头按定义写 `int`。

> 这与 §6.2.2 是同一类问题：**同一符号的调用方与定义方各拿一份声明，漂移无人发现**。
> 区别是这次是返回类型而不仅仅是重复。

#### 6.7.5 I8：让墙能真正拦人（而且我第一版是假通过的）

新增 I8 检查：簇外 TU 引用了公共头未声明的 batch 符号即失败。

**我的第一版 I8 是坏的**：`main()` 里的 `cluster` 变量**只含簇内文件**，
而我把这个列表传给了 I8，于是它只扫了簇内文件、然后**全部跳过**，
却仍然打印 `OK`。一个"永远通过"的检查比没有检查更糟。

反证测试暴露了它（注入跨边界调用 → 仍报 OK）；修法是让 I8 **自己枚举全树**
而不是信任调用方传进来的列表。修好后同一注入被精确抓到：

```
FAIL  MGL/src/mgl_blit_pipelines.c calls mglBatchRestoreStateForBatch,
      declared in mgl_batch_restore.h but not in mgl_batch_public.h
```

**这是我在这轮里第二次修掉"检查本身是坏的"**（第一次是 I7 的窗口启发式，
见 §6.2.4）。两次都说明同一件事：**审计器必须用反证测试验证，否则它会给你虚假信心。**

#### 6.7.6 A1（拆 `mgl_render.h`）：实测后判定**不能按原计划做**，附证据

原计划（§7 第 2 项）是"把 `mgl_render.h` 按域拆分，让 TU 只 include 需要的窄头"。
**实测后这个计划不成立**，原因如下（全部为脚本普查结果）：

**① 命名不含域信息。** 694 个声明里 **689 个都是 `mglRender*`** 前缀。
按"`mglRender` 之后的词干"聚类，结果只得到一个桶：

```
Render 689 / Resolve 1 / Current 1 / Renderer 1 / Restore 1 / Xfb 1
```

即**无法从名字推断域**，拆分必须逐个声明判定用途 —— 那是 694 次人工判断，不是重构。

**② 97.7% 的声明由同一个文件支撑。** 按"定义所在的实现文件"普查：

```
672  mgl_render.cpp
  4  mgl_renderer_entries.c
  2  mgl_render_pass_plan.c
 16  未找到（cull-distance 族，在 mgl_air_*/mgl_draw_* 里）
```

`mgl_render.h` 不是"多域声明的集合"，而是**单一实现文件的镜像**。而那个文件
`mgl_render.cpp` 有 19,445 行、**零分区横幅**（`grep -c "^/\* =\{5,\}"` = 0）。

> **结论：头文件的形状是单体实现的投影。**
> 只拆头不拆实现，会得到一个**在单体之上的多域 facade**——
> facade 再包一层 facade，72 个 TU 的 fan-in 一点不减，只是徒增间接层。
> **正确的顺序是"先拆实现，再让头跟着落"**，而不是反过来。

**③ 但拆是有意义的，而且优先级更清楚了。** 按每个 TU 实际用到多少个声明统计：

| 用量 | TU 数 | 说明 |
|---|---|---|
| 0 个 | 3 | 完全不需要（已在本轮移除） |
| 1 个 | 9 | |
| 2 个 | 3 | |
| 中位 | **8** | 72 个 TU 的中位用量 |
| 54–70 个 | 4 | blit / backend / texture upload / pass manager |
| **688 个** | **1** | `mgl_render.cpp` —— 只有它真的需要全部 |

**中位 8 个声明却 include 了 694 个。** 24 个 TU 用 ≤2 个 —— 这是真实的耦合浪费，
而且它们正是"只需一个窄头"就能解决的那批。

**已落地（本轮）**：移除 3 个零用量 TU 的 include
（`mgl_draw_entry.c` / `mgl_shader_resource.c` / `mgl_texture_entries.c`，实测移除后编译干净）。
**剩余零用量 TU 归零。**

**未做及原因**：那 24 个低用量 TU 各需 1–2 个符号，而这些符号**全部定义在 `mgl_render.cpp` 里、
没有现成窄头可归**（`mglRenderSamplerUnitExplicit`、`mglRenderQueryCapability`、
`mglRenderBytesPerPixelForInternalFormat`、`mglRenderTargetIsRenderbuffer`、
`mglRenderDecodeVertexAttribComponent`、`mglRenderInvalidateProgramPipelines` …）。
给它们现造一个"杂项窄头"会把同一个单体的声明面切成两块，**不会减少耦合**。

- [x] **A1 修正后的做法**（顺序不可颠倒）。**2026-09-21 已按此顺序做完**：
      1. `mgl_render.cpp` 按域抽到实现文件（§9.7–§9.9、§9.13–§9.14）；
      2. 声明进同域窄头（§9.10）；
      3. `mgl_render.h` 收成 facade（含类型 + include 窄头）。
      `make core` 链接通过；抽查符号闭包见 §9.14。
- [x] **A1 的前置条件**：域清单已落地为 `scripts/mgl_render_domain_map.py`（§9.6 / A1a）。

#### 6.7.7 A1 前置：`mgl_render.cpp` 域清单（已完成普查，含三项否证）

为了给 19,494 行的 `mgl_render.cpp` 切分，试了四种口径。**三种被否证**，第四种给出了可用的域信号。

**① 按名字词干 —— 否证。** 694 个声明里 689 个是 `mglRender*`；按 `mglRender` 之后的词干聚类
只得到一个桶（`Render` 689，其余 5 个都是单例）。名字不含域。

**② 按体内类型的主导声明头 —— 否证。** 用"函数体里出现的大写类型名 → 哪个头声明了它"
反推域归属：**672 个里只有 1 个能归域**。原因是单体里的类型基本都是**文件内自定义**的
（不在任何头里），所以这条通路整体是空的。

**③ 按行带找连续同名区域 —— 否证。** 定义分布是散开的：`mglRenderAcquireSampleQuerySlot`
@12013、`mglRenderBeginSampleQuery` @12016 相邻，但 `mglRenderCreateCullDistanceArrayPlan`
在 @18265。**域存在，但实现顺序不是按域排的** —— 所以"按行区间切"不成立。

**④ 按函数体内的被调函数词干统计域信号 —— 可用。** 每 2000 行一带，累计该带所有函数
体内调用的 `mgl*` 词干（去掉 `mglRender`/`mglBatch`/`mglDraw`… 前缀后取首个词）。
结果**每带主题相当清晰**：

| 行带 | 函数 | 行数 | 主导被调主题 |
|---|---:|---:|---|
| 1..2000 | 18 | 702 | Snapshot, Note, Create, Init, Release, Bind |
| 2001..4000 | 49 | 1876 | Create, Get, Bind, End, Texture, Wait |
| 4001..6000 | 20 | 1996 | **Float×42, Snorm×15**, Copy, Texture |
| 6001..8000 | 82 | 1770 | Texture, Pixel, Copy, Expected, Prefer |
| 8001..10000 | 140 | 1506 | **Readback×20, Clear×15, Buffer×13** |
| 10001..12000 | 47 | 1949 | **Resolve×66**, Texture, Read, Stored |
| 12001..14000 | 70 | 1876 | Create, Get, Binding, Describe |
| 14001..16000 | 79 | 1998 | **Binding×35, Command×13, Encode×8, Classify** |
| 16001..18000 | 106 | 2015 | **Create, Encode, Command**, Allocate, MDI |
| 18001..20000 | 61 | 1375 | **Set, Active, Binding, Encode, Fill** |

**附带测出的规模事实**：672 个已声明函数的定义跨度合计 **17,063 行 = 文件的 88%**；
跨度中位 **12 行**，最大 394 行（`mglRenderConvertVertexBuffer`）。
即这是一个**长尾极长、单体极宽**的分布：大量小函数 + 少量几百行的大函数。

#### 结论：域清单可以给到"带级"，但切分必须**按函数抽取，不能按行区间**

① 的否证与 ④ 的结果合起来说明：**域是真实的（④ 的带级主题一致），但函数在文件里按域交错
存放（③）**。所以切分方式是：

- 给每个函数标注域（④ 的带级主题可作为初判，仍需人工确认边界处的函数）；
- 按域**抽取函数体**到新 TU，**不是**按行区间剪切；
- 每个新 TU 只需 include 它自己用到的头 —— 这正是要收获的东西。

- [x] **A1a** 已落地：`python3 scripts/mgl_render_domain_map.py` →
      `scripts/mgl_render_domain_map.json`（748 个定义，不是按行切）。
      详见 §9.6。`core` 残余在抽走其它域之前还要再判。
- [x] **A1b** 按该映射抽取函数到域 TU。非 `core` 定义已进对应域文件；`mgl_render.cpp` 只留规则残余 `core`（§9.14）。`make core` 链接通过；抽查符号只在一个 `.o` 里有定义。
- [x] **A1c** 已把非 `core` 的声明移到 `MGL/include/mgl_render_api_*.h`。`mgl_render.h` 在类型定义之后 include 这些窄头，所以原来 include `mgl_render.h` 的 TU 符号可见集不变。详见 §9.10。

> **本节的价值主要是三条否证**：它们把"拆 `mgl_render.h`"从一个看起来机械的任务，
> 变成"需要先做一次 672 项函数级域标注、并按函数抽取"的任务。
> 不做这三条否证就动手，会在第一刀就撞上"函数是散开的"。

## 7. 建议顺序

| 序 | 动作 | 理由 | 风险 |
|---|---|---|---|
| — | ~~**A11 删死头文件**~~ | ✅ **已完成**：删 `mgl_spirv_compile.h` / `MGLBindingSync.h` / `MGLQueryManager.h` / `enums.h`（前三个是纯 license 存根、零代码；`enums.h` 的 `get_enum` 无定义无调用）。构建 0 error | — |
| — | ~~**A8 断环**~~ | ✅ **已完成**：`mgl_pso_format_class.h` 改用 `struct Program_t;` 前置声明 + 让 `mgl_render.h` include `<GL/glcorearb.h>` 自给自足。**环 1 → 0**，140 个 `.c` 0 error，10 个门禁全绿。详见 §6.2 | — |
| — | ~~**A5 给 batch 簇定入口与不变量**~~ | ✅ **已完成**（§6.7 / §6.7.3–§6.7.5）：`mgl_batch_public.h` + I1–I8。§5.1 的勾选与本节曾不同步，§9 收口 | — |
| 2 | **A1 拆 `mgl_render.h` —— 修正后不可直接做** | 实测（§6.7.6）：694 声明中 689 个是 `mglRender*`（**名字不含域**），且 672/678 的定义都在 `mgl_render.cpp` 单体里（19,445 行、零分区）。**头是单体实现的投影，只拆头不拆实现等于 facade 包 facade。** 正确做法是先拆实现 | 高（需先做实现切分） |
| — | ~~**A1 前置：`mgl_render.cpp` 域清单普查**~~ | ✅ **已完成**（§6.7.7）：试了 4 种口径，**3 种否证**（按名字词干 / 按体内类型头 / 按行带连续区域），第 4 种（按带内被调函数词干）给出**带级域信号**（10 带，主题清晰）。同时测出：672 个声明覆盖文件 **88%**、跨度中位 **12 行** | — |
| — | ~~**A1a 函数级域标注**~~ | ✅ **已完成**（2026-09-21）：`scripts/mgl_render_domain_map.py` 对 `mgl_render.cpp` 里 **748** 个 `mgl*` 定义给出 `symbol → 域`。`core` 142 个是规则没吃到的残余，不是一个可抽取的域。见 §9.6 | — |
| — | ~~**A1b 按域抽取函数**~~ | ✅ **已完成**（§9.14）：非 `core` 定义进域 TU；`mgl_render.cpp` 只留 `core` 残余。`make core` 链接通过 | — |
| 2c | ~~**A1 已落地部分**~~ | ✅ 移除 3 个**零用量** TU 的 include（`mgl_draw_entry.c` / `mgl_shader_resource.c` / `mgl_texture_entries.c`），实测编译干净，剩余零用量 TU **归零**。另测得 24 个 TU 用 ≤2 个声明、72 个 TU 中位仅 8 个 | — |
| — | ~~**A10 重复声明去重**~~ | ✅ **已完成**：**44 → 0 组**。判据固化为"定义在哪个语言里，声明就归哪一侧的头"；顺带单一来源化了错误 API（§6.2.2） | — |
| — | ~~**A4 判定 20 个 C 叶子的编排归属**~~ | ✅ **已由沉入完成**（§9.1）：`*.m` 为 0，20 个文件全部有 C/C++ 调用者。历史结论「编排仍在 `.m`」不再成立 | — |
| — | ~~**A6 `owner`/`binding_state` 具体化**~~ | ✅ **已完成**（§9.12）：门面与 `mgl_render_api_*.h` 的 `void *binding_state` / `void *owner` / `void **owner` 为 0。`make core` 链接通过 | — |
| — | ~~**A9 GL API 面改 include 同域窄头**~~ | ✅ **已完成**（§9.16）：四 TU 改窄头；窄头 standalone 走 `mgl_render_fwd.h`，不拉 facade。`make core` 链接通过 | — |
| — | ~~**A12 收敛类型层平台依赖**~~ | ✅ **已完成**：3 处多余 `vm_types.h` 已删；只剩 `mgl_types_buffer.h` 的 `vm_address_t` 一处真实依赖。副产品：修好了 `mgl_draw_mode.h` 的 ObjC 依赖（§6.2.3） | — |
| — | ~~**A7 解 `glm_context.c` 的 `MGLRenderer.h`**~~ | ✅ **已完成**（§9.15）：`CppCreateMGLRenderer*` 进 `mgl_platform_shell_result.h`；`glm_context.c` 不再 include `MGLRenderer.h`。`make core` 链接通过 | — |
| 8 | **A3 用链接期符号引用重做死码清单** | 静态方法无法识别分派表入口（已证实误报） | 低 |

**A8 的结果修正了原计划**：原计划"先补 16 个 TU 的窄声明再删边"是多余的——
真正需要补的不是调用方，而是 **`mgl_render.h` 自己对 `GLuint` 负责**（一行）。
断环后实测只破 5 个 TU，且都不是它们写错，而是门面不自洽。见 §6.2.1。

---

## 8. 证据边界

- 所有文件数 / 行数 / 分布 / 函数计数均为脚本实测（函数定义体按花括号配对提取，
  共识别 3,313 个 `mgl*` 函数定义）。
- **同类名提取的已知偏差**：函数定义识别依赖 `^[类型] mgl*Name(` 且紧随 `{` 的模式，
  宏生成的函数、多行返回类型、以及 `.def`/X-macro 展开的符号**会漏计**。
  故 3,313 是**下界**。
- "只被 `.m` 调用"的判定基于调用点文本匹配；经**函数指针表**调用的符号会被误判为
  "无调用者"（§4.2 已证实该偏差真实存在）。
- §6 的 include 图数字（fan-in/fan-out、59 条越界边、40 组重复声明、
  三头可达集相同、删边后 88→48）来自**后台审查的测量**，我抽查了其中可验证的部分
  （重复声明确认逐字相同；4 个死头文件独立复核为 0 引用；
  `.c/.cpp` 不 include `*_Private.h` 复核为真但需修正为"有注释、无 include"）。
  **未逐条重算全图**，故这些标为二手。
- §6.2 的破编译结果是我**实测**的：临时删掉 `glm_context.h:96` 后对 106 个 `.c`
  逐个 `gcc -fsyntax-only`，16 个报错；报错清单见该节。**实验后已立即还原
  `glm_context.h`，确认 `git diff` 为空、`buffers.c` 恢复正常编译。**
- 未做"补窄声明后删边"的完整验证（那需要真正新建头文件 = 改代码），
  故 88→48 的收益是**后台审查的图论测量**，不是编译实测。
- 未测编译时间与链接时间；A1/A8 的收益应以实际增量编译时长验证，而非行数。
- §3 的 `void *` 统计基于"函数定义体按花括号配对 + 参数文本匹配"，宏生成的函数会漏计；
  3,313 是下界。
- 本审查不涉及逻辑正确性（那在 `docs/STATE_DATAFLOW_TODO.md`）。

---

## 9. 2026-09-21 续审（收口未决项）

正文日期是 2026-09-13。下面只补**当时没写完、或已被后续沉入推翻**的项。
不改代码。§1 的 127 / 161,068 保留为当时快照，不以本节数字回写。

### 9.0 重测规模（不要和 §1 混读）

| 指标 | 2026-09-13 | 2026-09-21 |
|---|---:|---:|
| `MGL/src` 下 `.c`+`.cpp` | 127 | **164** |
| 这些文件的行数 | 161,068 | **204,135** |
| `mgl_render.cpp` | 19,445 | **19,494** |
| `mgl_render.h` 行数 / `mgl*` 声明出现次数 | 3,796 / 722 | **3,759 / 696** |
| 直接 `#include "mgl_render.h"` 的 TU | 文中先后写过 55、72 | **88** |
| `*.m` | 有（§4 的调用方） | **0** |

文件数上升、`mgl_render.h` 行数略降，说明沉入在加 C 文件，**没有拆掉单体**。
A1 的「先拆实现再拆头」仍然成立（§6.7.6–§6.7.7）。A1a（672 项 `symbol → 域`）**本轮未做**，仍是 A1 的工作量本身。

### 9.1 A4：20 个「只被 `.m` 调用」的叶子 —— 作废

仓库里已经没有 `.m`。对 §4.1 那 20 个文件重扫「导出 `mgl*` 是否被其它 `.c`/`.cpp` 调用」：

**20/20 都有 C/C++ 调用者。** 调用者最少的是单点编排，不是死叶子：

| 文件 | C/C++ 调用方（抽样） |
|---|---|
| `mgl_batch_flush_restore_encode.c` | `mgl_platform_shell.cpp` |
| `mgl_blit_plan.c` | `mgl_blit_color_paths.c` |
| `mgl_blit_clip.c` | `mgl_blit_drivers.c` |
| `mgl_vertex_attrib_plan.c` | `mgl_buffer_map.c` |
| `mgl_focus_program.c` | `mgl_sampled_sampler.c` |
| 其余 15 个 | 2–9 个 C/C++ 调用方 |

「计算在 C、编排在 ObjC，所以结构复杂度没下降」描述的是 2026-09-13 的中间态。
编排已经跟着进了 C/C++。**A4 关闭。** 不要再给这些头加「ObjC 叶子」注释。

`mgl_batch_flush_restore_encode.c` 仍只有 `mgl_platform_shell.cpp` 一个调用方，
但公共面已经由 `mgl_batch_public.h` + I8 卡住（§6.7），不是漏边界。

### 9.2 A7：`glm_context.c` 的 `MGLRenderer.h`

`#include "MGLRenderer.h"` 仍在 `glm_context.c:51`。头文件结构是：

- `__OBJC__` 才 `#import` AppKit / 声明 `@interface MGLRenderer`
- `#else` 才给 C 看 `mtlPixelFormatForGLFormatType`
- `CppCreateMGLRendererHeadless` 等在 `#endif` 之后，C/C++ 都可见

`glm_context.c` 是 `.c`，**不会进入 `__OBJC__` 分支**。§6.1「否则无法作为纯 C TU 编译」
在当前头结构下**不成立**（它已经在 `make` 里当 C 编译）。

仍依赖这次 include 的真实符号：

| 符号 | 用法 |
|---|---|
| `mtlPixelFormatForGLFormatType` | `glm_context.c:177`、`:303`、`:310` |
| `CppCreateMGLRendererHeadless` | `:107` 调用；`:88` 又写了一遍本地 `extern`（与头重复） |

`platform_renderer_shell` 的释放走 `CFRelease`（`:1014`），不需要这个头。

**收口（2026-09-21，§9.15）**：`CppCreateMGLRenderer*` 已进 `mgl_platform_shell_result.h`；
`glm_context.c` 改 include 该头 + 已有的 `mgl_glfw_abi.h`（提供 `mtlPixelFormatForGLFormatType`），
并删掉本地 `extern`。C/C++ TU 不再 include `MGLRenderer.h`。

### 9.3 A9：GL API 面 → `mgl_render.h`（已收口，见 §9.16）

原先直接 `#include "mgl_render.h"` 的 GL 对象入口是这 4 个：`buffers.c` / `textures.c` /
`program.c` / `tex_param.c`。现已改 include 同域窄头（§9.16）。

### 9.4 原开放项（现已按现状勾选）

| 项 | 状态 |
|---|---|
| **A1a** | ✅ 见 §9.6 |
| **A1b / A1c** | 均 ✅。A1c 见 §9.10；A1b 见 §9.14（`mgl_render.cpp` 只留 `core`） |
| **A2** | ✅ `MGL/include/mgl_render.h` |
| **A6** `binding_state` / `owner` 具体化 | ✅ 见 §9.12。门面与窄头里这三项为 0 |
| **A7** `glm_context.c` ↔ `MGLRenderer.h` | ✅ 见 §9.15 |
| **A9** GL 面 include 窄头 | ✅ 见 §9.16 |
| **§7 第 8 项** 用链接期引用重做死码清单 | ✅ 见 §9.11。`nm -u` 确认 §4.2 的分派表入口有跨 TU 引用 |

### 9.5 续审证据边界

- 行数、include 次数、`void *` 子串计数、20 个叶子的调用方，均为 2026-09-21 对工作区的脚本扫描。
- 叶子扫描用「行首 `mgl*` 定义 + 其它 TU 里 `name(`」；函数指针取地址仍会漏，和 §4.2 同一边界。本节能说的是「**有**直接调用者」，不能说「导出都有调用者」。
- 未改任何 `.c` / `.h`，未跑 `make`。

### 9.6 A1a：`symbol → 域`（2026-09-21）

生成：

```sh
python3 scripts/mgl_render_domain_map.py
```

产物 `scripts/mgl_render_domain_map.json`。口径是**定义**（列首的 `mgl*(`，花括号在分号之前），含 `static`，所以是 748 而不是 §6.7.7 的 672 个已声明函数。

规则：先把 `Swizzle` / `Snorm` / `Unorm` 归到 `pixel_convert`；其余跳过动词（`Set`/`Get`/`Create`/…）后，用名字里的域词匹配。先匹配者胜。

| 域 | 定义数 |
|---|---:|
| core（残余，未归域） | 142 |
| command | 124 |
| binding | 114 |
| texture | 98 |
| pixel_convert | 70 |
| buffer | 56 |
| draw | 49 |
| readback | 43 |
| query | 30 |
| lifecycle | 22 |

每个域的定义都从文件前部散布到尾部（相邻定义行距 >80 的缺口都在两位数）。这和 §6.7.7 ③ 一致：**不能按行段剪切**。

跨域 `static` 调用（调用方域 ← static 所在域，次数）：

| 次数 | 调用方 | static 所在 |
|---:|---|---|
| 74 | lifecycle | pixel_convert |
| 35 | texture | pixel_convert |
| 13 | binding | command |
| 9 | core | command |

所以 A1b 的第一刀不是「把 pixel_convert 的 70 个函数剪走」——`mglHalfToFloat` 这类 static 还被别的域用。要先把这 46 个 static 的可见性定下来，再按域搬非 static 定义。上表是抽离前的快照；抽离后的数字见 §9.7。

### 9.7 A1b 第一刀：像素 helper（2026-09-21）

从 `mgl_render.cpp` 抽出 21 个彼此闭包、不调用本文件其它 `mgl*` 的 `static`。其中 7 个（`mglHalfToFloat`、`mglFloatToHalf`、`mglFloatToFloat10`、`mglFloatToFloat11`、`mglPackRGBToSharedExp`、`mglPackUnsignedFloatFromUNorm8`、`mglUnpackUnsignedFloatComponent`）与 `pixel_utils.c` 同名且函数体一致（差在注释），继续用 `pixel_utils.h`，没有第二份定义。其余 14 个定义在 `MGL/src/mgl_render_pixel.cpp`，声明在 `MGL/src/mgl_render_pixel.h`，`mgl_render.cpp` 只保留调用。

`make core` 链接通过。`nm -u` 显示这 14 个符号都由 `mgl_render.o` 引用，定义在 `mgl_render_pixel.o`。

抽离后重跑 `scripts/mgl_render_domain_map.py`：`mgl_render.cpp` 定义 **727**（原先 748），`pixel_convert` **49**（原先 70），跨域 `static` **15**（原先 37）。`mgl_render.cpp` 当时 19065 行。这是第一刀的快照；当前分布见 §9.8。

### 9.8 A1b 第二刀：不依赖 `mgl::` 的定义（2026-09-21）

能搬走的条件：已在 `mgl_render.h` 声明、函数体不写 `mgl::`、不调用本文件 `static`、也不调用只在 `mgl_render.cpp` 里可见的 helper（`rendererOwner`、`BackendLeaseScope`、`tessDomainInput`、`MGLRenderReadIndexBytes` 等）。按这个条件从 `mgl_render.cpp` 抽出的定义：

| TU | 定义数 |
|---|---:|
| `mgl_render_texture.cpp` | 67 |
| `mgl_render_pixel.cpp` | 50 |
| `mgl_render_command.cpp` | 42 |
| `mgl_render_buffer.cpp` | 39 |
| `mgl_render_draw.cpp` | 30 |
| `mgl_render_binding.cpp` | 29 |
| `mgl_render_readback.cpp` | 27 |
| `mgl_render_query.cpp` | 10 |
| `mgl_render_lifecycle.cpp` | 5 |
| 仍在 `mgl_render.cpp` | 442 |

合计 741（`python3 scripts/mgl_render_domain_map.py`）。`make core` 链接通过，没有重复的 `mgl*` 定义。`core` 没有单独 TU：它是分类残余，不是域。

§9.8 当时没搬走的是还在匿名命名空间里的类型。那一步已经做完，见 §9.9。

### 9.9 A1b 第三刀：内部类型出匿名命名空间（2026-09-21）

`MGL/src/mgl_render_internal.h` 收进原先匿名命名空间里的 owner / `BindingState` / `Renderer`，以及 `BackendLeaseScope`、`rendererOwner`。自由函数标成 `inline`，两份 `std::atomic` 计数也是 `inline`，所以多个 TU 包含这份头不会重复定义。`make core` 链接通过。

这之后又搬走 188 个已经在 `mgl_render.h` 声明、且不调用文件私有 helper 的定义。当前分布（`python3 scripts/mgl_render_domain_map.py`）：

| TU | 定义数 |
|---|---:|
| `mgl_render_command.cpp` | 109 |
| `mgl_render_binding.cpp` | 86 |
| `mgl_render_texture.cpp` | 80 |
| `mgl_render_buffer.cpp` | 54 |
| `mgl_render_pixel.cpp` | 50 |
| `mgl_render_readback.cpp` | 36 |
| `mgl_render_draw.cpp` | 32 |
| `mgl_render_query.cpp` | 24 |
| `mgl_render_lifecycle.cpp` | 16 |
| 仍在 `mgl_render.cpp` | 254 |

留下的包括：名字分类的残余 `core`（不单开 TU），以及还在调用 `loadAuxLibraryLocked`、`recordBufferSlot`、命令恢复那组文件私有函数的定义。A1c 见 §9.10。

### 9.10 A1c / A2：声明进窄头，门面移出 `MGL/src`（2026-09-21）

按域映射把 `mgl_render.h` 里的函数声明拆出去（`core` 和映射里没有的 21 个声明留在门面里）：

| 窄头 | 声明数 |
|---|---:|
| `MGL/include/mgl_render_api_command.h` | 120 |
| `MGL/include/mgl_render_api_binding.h` | 109 |
| `MGL/include/mgl_render_api_texture.h` | 93 |
| `MGL/include/mgl_render_api_buffer.h` | 56 |
| `MGL/include/mgl_render_api_pixel.h` | 44 |
| `MGL/include/mgl_render_api_readback.h` | 38 |
| `MGL/include/mgl_render_api_draw.h` | 37 |
| `MGL/include/mgl_render_api_query.h` | 29 |
| `MGL/include/mgl_render_api_lifecycle.h` | 20 |

类型定义仍在门面里，因为它们和声明交错，而且被多个域共用。`mgl_render.h` 在类型之后 `#include` 这些窄头。窄头若被单独 include，走 `mgl_render_fwd.h`（不完整类型），不再拉整份 facade（§9.16）。`make core` 链接通过。

门面文件从 `MGL/src/mgl_render.h` 挪到 `MGL/include/mgl_render.h`。源文件仍写 `#include "mgl_render.h"`。`MGL/src/mgl_render_pixel.h` 是另一份 C++ helper 头，没有被这个名字盖住（`-IMGL/include` 在 `-IMGL/src` 前面，所以公共窄头用了 `mgl_render_api_*` 这个名字）。

### 9.11 链接期引用重做 §4.2（2026-09-21）

对 `build/core/MGL/src/*.o` 做 `nm -g -U`（定义）对 `nm -u`（其它 TU 的未定义引用）。这能看见跨 TU 调用和取地址，看不见同一 .o 内部的调用。

| 目标 | 导出 | 有其它 TU 引用 | 无跨 TU 引用 |
|---|---:|---:|---:|
| `non_core_unimplemented.o` | 370 | 370 | 0 |
| `compute.o` | 2 | 2 | 0 |
| `fence.o` | 11 | 9 | 2（`isSync`、`newSync`，同文件使用） |
| `samplers.o` | 20 | 16 | 4（`findSampler`、`getSampler`、`isSampler`、`newSampler`，同文件使用） |
| `mgl_draw_gs.o` | 40 | 30 | 10（同文件的 gather/topology helper） |

§4.2 的结论成立：分派表入口不是死码。无跨 TU 引用的 `_mgl*` 还有 459 个，多数是本文件调用的外部链接 helper，不能据此删。`-dead_strip` 对 dylib 导出会把它们当成活符号，所以没有拿它当删除清单。

### 9.12 A6：`binding_state` / `owner` 具体化（2026-09-21）

门面在 `extern "C"` 之前给出不透明 typedef（`MGLBindingState`、`MGLCommandBufferOwner`、`MGLCommandBufferRecoveryOwner`、`MGLCommandQueueOwner`、`MGLCullDistanceIndexPlan`、`MGLMDIScratchOwner`、`MGLPendingEventOwner`、`MGLPipelineCacheOwner`、`MGLQueryStateOwner`、`MGLRenderEncoderOwner`、`MGLRenderPassIdentityOwner`、`MGLRenderPassStateOwner`、`MGLTextureStagingOwner`）。布局仍是 `mgl_render_internal.h` 里的 `mgl::` 类型。两边不是同一 C++ 类型，所以跨边界用 `reinterpret_cast`，不用 `static_cast`。

重测 `MGL/include/mgl_render.h` 与 `mgl_render_api_*.h`：

| 参数形态 | §9.4 | 现在 |
|---|---:|---:|
| `void *binding_state` | 54 | 0 |
| `void *owner` | 82 | 0 |
| `void **owner` | 28 | 0 |

`MGLCommandState` 的 owner 槽和 `MGLRendererBackendHandle` 里的 owner 字段改成了同样的不透明指针。`make core` 链接通过。

没动的是端口袋，不是这组声明：`mgl_renderer_ports.h` 仍有 `void **binding_state_owner`、`void **gpu_recovery_command_owner`、`void *query_state_owner`（槽地址，由 shell 填）。`mgl_batch_replay.h`、`mgl_batch_mtl_encode.h`、`mgl_draw_gs.h` 的端口参数也还是 `void *`。

### 9.13 A1b 第四刀：不碰文件私有 helper 的定义（2026-09-21）

从 `mgl_render.cpp` 再搬走 33 个定义。判据：不是 `static`，域不是 `core`，函数体不调用本文件的 `static` 或那 16 个文件私有 helper。其中 5 个（两个 texture descriptor 创建函数、三个 attachment getter）调用了 `static` 写在上一行的 helper，编译失败后移回。`mglRenderIsVirtualizedGPU` 的定义补回 `extern "C"`，和 `MGLRenderer.h` 的声明一致。

`make core` 链接通过。抽查符号只在一个 `.o` 里有定义：`mglRenderInvalidateRenderPass`（command）、`mglRenderGLIndexElementSize`（draw）、`mglRenderBeginSampleQueryCallback`（query）、`mglRenderTexturePrepareLevelUpload`（texture）、`mglRenderPipelineDescriptorSignature`（binding）、`mglRenderDoubleVertexAttribFloatFormat`（pixel）。

当前分布（`python3 scripts/mgl_render_domain_map.py`）：

| TU | 定义数 |
|---|---:|
| `mgl_render_command.cpp` | 110 |
| `mgl_render_binding.cpp` | 94 |
| `mgl_render_texture.cpp` | 87 |
| `mgl_render_buffer.cpp` | 54 |
| `mgl_render_pixel.cpp` | 52 |
| `mgl_render_draw.cpp` | 42 |
| `mgl_render_readback.cpp` | 36 |
| `mgl_render_query.cpp` | 29 |
| `mgl_render_lifecycle.cpp` | 16 |
| 仍在 `mgl_render.cpp` | 223 |

`core` 142 个仍不单开 TU。剩下的非 `core` 定义还绑在文件私有 helper 上。

### 9.14 A1b 收口：文件私有 helper 出域 + 只留 `core`（2026-09-21）

把仍挡在 `mgl_render.cpp` 里的文件私有 helper 和依赖它们的非 `core` 定义一起搬走：

- 命令恢复匿名命名空间、`mglRenderActiveRenderEncoder`、encoder reset / compute dispatch → `mgl_render_command.cpp`
- aux library / `recordBufferSlot` → `mgl_render_binding.cpp`
- 像素 readback 格式 helper → `mgl_render_pixel.cpp`（声明进 `mgl_render_pixel.h`）
- 索引展开 / cull / tess domain → `mgl_render_draw.cpp`
- `wrapDevice` / depth-stencil 构造 → `mgl_render_lifecycle.cpp`
- `clearBufferSlot` → `mgl_render_readback.cpp`

跨 TU 需要的 helper 声明放在 `mgl_render_internal.h`（或 pixel 头）。最后 5 个 attachment / texture-descriptor 函数连同其 `static` helper 也搬走。

`make core` 链接通过。`python3 scripts/mgl_render_domain_map.py` 当前分布：

| TU | 定义数 |
|---|---:|
| `mgl_render_command.cpp` | 125 |
| `mgl_render_binding.cpp` | 113 |
| `mgl_render_texture.cpp` | 99 |
| `mgl_render_pixel.cpp` | 69 |
| `mgl_render_buffer.cpp` | 56 |
| `mgl_render_draw.cpp` | 55 |
| `mgl_render_readback.cpp` | 38 |
| `mgl_render_query.cpp` | 30 |
| `mgl_render_lifecycle.cpp` | 23 |
| 仍在 `mgl_render.cpp` | 135（全部为残余 `core`，不单开 TU） |

抽查：`mglRenderInit`（lifecycle）、`mglRenderCommitCommandBufferTransaction`（command）、`mglRenderBindingUpdateVertexBuffer`（binding）、`mglRenderCreateTextureFromDescriptor`（texture）、`mglRenderExpandLineLoopIndices`（draw）各只在一个 `.o` 里有定义。

### 9.15 A7：`glm_context.c` 不再 include `MGLRenderer.h`（2026-09-21）

| 符号 | 归属 |
|---|---|
| `mtlPixelFormatForGLFormatType` | 已有 `mgl_glfw_abi.h`（`glm_context.c` 本来就 include） |
| `CppCreateMGLRenderer*` | 新声明进 `mgl_platform_shell_result.h`（shell 的纯 C 面） |

`glm_context.c` 删掉 `#include "MGLRenderer.h"` 和本地 `extern CppCreateMGLRendererHeadless`。
`MGLRenderer.h` 在 `__OBJC__` 之后改 include `mgl_platform_shell_result.h`，避免 ObjC 调用方丢声明。
`make core` 链接通过。C/C++ TU 对 `MGLRenderer.h` 的 include 归零（只剩 ObjC 私有头 `#import`）。

### 9.16 A9：四 TU 改同域窄头，窄头不再拉 facade（2026-09-21）

| TU | 原 include | 现 include |
|---|---|---|
| `buffers.c` | `mgl_render.h` | `mgl_render_api_buffer.h` |
| `textures.c` | `mgl_render.h` | `mgl_render_api_texture.h` + `mgl_render_api_lifecycle.h` |
| `program.c` | `mgl_render.h` | `mgl_render_api_binding.h`（含迁入的 `mglRenderInvalidateProgramPipelines`） |
| `tex_param.c` | `mgl_render.h` | `mgl_render_api_pixel.h` |

`mgl_render_api_*.h` 在未被 facade 打开时改为 `#include "mgl_render_fwd.h"`（不完整类型），
不再 `#include "mgl_render.h"`。standalone 编译这几个窄头时 include 图里 **没有** `mgl_render.h`。
`make core` 链接通过。


