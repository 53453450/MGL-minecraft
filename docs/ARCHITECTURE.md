# MGL 当前架构契约

本文只记录代码当前需要维护的边界和验证入口。历史审查报告是背景资料，不定义另一套目标架构。

## 分层职责

| 层 | 目录 | 可以拥有 | 不应承担 |
|---|---|---|---|
| L5 API | `MGL/src/L5_api` | GL 入口、dispatch、参数入口 | Metal 对象或 renderer 内部操作 |
| L4 Context | `MGL/src/L4_context` | GL 对象、状态、错误、program/link 编排 | Metal 生命周期和 encoder 细节 |
| L3 Validate / Plan | `MGL/src/L3_validate` | 校验、binding resolve、纯数据 plan | backend 对象分配、释放或 Metal 状态写入 |
| L2 Encode | `MGL/src/L2_encode` | batch、snapshot/replay、hazard、pass 驱动 | GLSL parser/AST、平台窗口细节 |
| L1 Backend | `MGL/src/L1_backend` | Metal 对象、PSO、cache、staging、资源生命周期 | 完整 GL context 记录或 attach 锁操作 |
| L0 Platform | `MGL/src/L0_platform` | AppKit/CAMetalLayer、device、drawable、present、异常边界 | batch/pass/AIR/compute/texture 内部实现 |
| Compiler | `MGL/src/compiler` | GLSL frontend、AIR、reflection、compile artifact | 平台或 replay 状态 |

跨层调用优先复用现有窄接口：`mgl_batch_public.h`、`mgl_render_pass_public.h`、`mgl_render_api_*.h`、`mgl_render_values.h`、`mgl_backend_handles.h`。不要为形式上的可替换后端新增生产 vtable。

## 核心所有权

- **Active GL state**：`ctx->active_state` 是唯一指针；只有 `mglContextSetActiveState` 写入。Replay 使用调用方 workspace，teardown 必须恢复 `ctx->state`。
- **Context/backend attachment**：L4 `mgl_context_renderer_backend.c` 拥有 context 锁、lease begin 和 detach；L1 只接收 opaque `GLMContext`，不读取其字段。销毁时先 detach，再由 backend 等待 lease 排空并通知 platform shell。
- **Sampler backend**：L1 `mgl_sampler_backend.c` 拥有 Metal state 的创建、替换和释放。L3 只解析 GL binding 与生成 `MGLSamplerPlan`。
- **Texture sampled facts**：L1 `mgl_texture_backend.c` 读取 Metal handle 并生成值快照；L3 不直接访问 backend。
- **CompilePlan**：`MGLCompilePlan` 是单阶段编译的纯数据输入。`MGLCompileArtifact` 拥有 metallib、reflection 和 TU，只有完整成功才发布到 Program。
- **GS XFB ABI**：C 与 Metal 共用 `mgl_gs_xfb_layout.h` 的容量和常量；两侧 struct 用编译期断言检查 size/offset。超限配置由 link/validation 显式拒绝。
- **Texture staging**：pool 记录 device 身份，最终 `mglRenderShutdown` 调用 `mglRenderTextureStagingPoolShutdown`。旧 device 的迟到 owner 不得进入新 device 的池。
- **Deferred snapshot**：hot/cold buffer 域由 `MGL_SNAPSHOT_*_BUFFER_BASE_TYPES` 单源展开；copy、retain 和 release 共用该分类。布局静态断言不得删除。

## 禁止的捷径

- 不在 `MGL/src` 根目录新增实现 TU；实现文件必须进入 L0–L5 或 compiler，根目录只保留跨层内部头。
- 不为修分层而新增无条件 flush、`glFinish` 或整 command-buffer wait。
- 不让扩展直接读取 `MGLRendererStateAreas` 或 renderer Private 数据。
- 不用静默截断代替超限错误。
- 不在测试文件里复制生产 planner 来代替链接生产实现。
- 不因目录移动或门禁修复增加违规 baseline；减少后及时更新数字。

## 验证入口

- `make test-architecture-gates`：门禁自身的临时树测试。
- `make test-arch-layer-gates`：目录级依赖与所有权规则。
- `make test-state-invariants`：状态机、batch 边界和 replay 冷域检查。
- `make test-state-dataflow`：state key、hash 和字段覆盖。
- `make test-active-state-restore`：active-state 激活与恢复。
- `make test-sampler-backend`：sampler 替换、释放和分配失败。
- `make test-compile-plan`：编译变体的纯数据测试。
- `make test-air`、`make test-metalcpp`：AIR、PSO、backend 和平台 smoke。
- `make test-all`：完整非交互门禁；真实 Apple GPU 仍是图形回归的必要环境。
