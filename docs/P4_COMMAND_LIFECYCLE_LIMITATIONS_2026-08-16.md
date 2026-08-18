# P4 command lifecycle limitations (2026-08-16)

范围：记录 P4 command lifecycle 从 guard 迁移到 owner transaction、completion
recovery 和完整 TSan 复核的边界。历史切片保留；2026-08-17 状态见文末更新。

## 2026-08-17 P4 最终状态

P4 command lifecycle 已完成；本文后续标为 2026-08-16 的段落是迁移中的历史
边界，不再代表当前阻断项。当前 gate-on 状态如下：

- `CommandBufferOwner` transaction 管理 detach/submission ownership、commit guard、
  commit、可选 wait、completion 注册、状态快照、last-submitted、next-current 与
  reset-request latch；`CommandBufferRecoveryOwner` 管理 error/success/recovery mode。
- transaction result 同时返回 recovery snapshot、单次错误记录标记、driver rejection
  与 reset 决策。skipped-error、同步 transaction failure 和异步 completion 通过同一
  recovery completion context 串行化，避免同一失败被 GL 线程与 completion worker
  重复计数。
- `.m` 中 `mglRenderCommandBufferOwnerGetCurrent` 为 0；`MGLRenderPassManager.m`
  只保留 owner C ABI adapter，不再实现 gate-on command-buffer lifecycle 策略。
- ObjC 保留平台日志、GL problematic-state 清理、最终 reset hook 和
  `MGL_USE_METALCPP=0` A/B adapter。这些是 P5 前的壳，不是 gate-on lifecycle owner。
  `commitCommandBufferWithAGXRecovery:` 不再执行 transaction 前分类或调用
  `recordGPUError`；`@catch` 只把不能跨 C++ ABI 的平台异常交回 recovery owner 并
  发布其 value-state reset 决策。
- command queue 初始化和 AGX reset 使用显式 gate 分支：effective gate-on 只调用
  C++ queue owner create/reset，失败即保持失败，不回落 ObjC `newCommandQueue`；
  gate-off 才执行 ObjC queue 创建。fallback render-encoder getter 在 gate-on 也会
  自行返回 null，防止新增调用点意外借出 encoder。
- GL 调用仍要求原有 GL 线程与外层 `METAL_LOCK` 串行化；owner mutex 不扩大为
  任意跨线程 GL API 承诺。

## 原 ObjC 行为基线（2026-08-16 历史语义）

- 原 guard 是普通 `BOOL`，只用于同一提交调用链的嵌套重入保护，不是线程间
  同步原语。C++ owner 使用普通 `bool` 保持该语义；本轮不额外引入 mutex 或
  atomic，也不宣称支持多个线程并发提交同一 owner。
- `commitCommandBufferWithAGXRecovery:` 仍负责提交前 status 校验、异常捕获、
  GPU error 记账和 deferred device reset。这些是 renderer 的 ObjC 高层恢复
  策略，不能仅靠移动一个 guard 就机械删除。
- 当 detached submission 不匹配时，旧路径仍可能回退到 Metal-cpp raw commit
  或 ObjC `[commandBuffer commit]`。本轮保留该 fallback 顺序，不把异常恢复
  差异混入 owner 状态迁移。
- 原提交校验先判断 `status >= MTLCommandBufferStatusCommitted`，再判断
  `status == MTLCommandBufferStatusError`。由于 Error 的枚举值位于 Committed
  之后，后一个专用错误分支实际上不可达。本轮的 C++ value-state 分类保持
  这个顺序，不修正既有行为。
- `recordGPUSuccess` 注释和计数要求 4 次连续成功且距最后错误超过 0.25 秒才
  重置错误计数；但 completion 随后会另加一次锁，在首个成功时直接清除
  `gpuErrorRecoveryMode`。两者并不等价。本轮 C++ owner 用独立的
  `RecordSuccess` / `ClearMode` 调用保持原顺序和可见行为，不合并或修正策略。

## 后续迁移边界（历史，已于 2026-08-17 关闭）

completion/error-recovery 的 value-state、实际提交/等待、completion 注册、
recovery owner 更新和 next-current 编排现已迁入 C++ transaction；`.m` 中 raw
current-command-buffer getter 已归零。ObjC AGX recovery 方法保留的内容已缩为
平台日志、GL 状态清理、最终 reset hook 与 gate-off adapter，随 P5 删除迁移壳。

## completion/error 分类切片

- `mglRenderClassifyCommandBufferCommit` 把提交前状态归类为 proceed 或
  already-committed skip；Error 延续原枚举顺序落入后一类。ObjC 保留日志、
  异常捕获和实际 commit fallback。
- `mglRenderClassifyCommandBufferCompletion` 统一识别普通成功、一般错误及
  `MTLCommandBufferErrorDomain/code 4` driver rejection。后续编排切片已把错误
  计数迁入 C++ owner；ObjC 仍拥有 2 秒节流与 deferred device reset 策略。
- `CommandBufferRecoveryOwner` 接管原 `_gpuRecovery` 中的 error/success 计数、
  last-error timestamp、recovery-mode 和跨 completion/GL 线程同步；8-error、
  3 秒 timeout、4-success/0.25 秒 reset 规则均逐字保留。ObjC 只传入时间、
  输出日志并执行 `clearProblematicGPUState`。

## completion/error-recovery 编排切片（2026-08-16）

`mglRenderProcessCommandBufferCompletion` 现在把一次 completion 的分类和
recovery owner 状态更新封装为单一 value-state 结果。它保留原顺序：错误完成只
记录 error；成功完成先执行 `RecordSuccess`，再单独执行 `ClearMode`。因此首个
成功 completion 清 mode 与 sustained 4-success reset 仍是两个可观察结果，未被
合并成新的策略。

ObjC completion block 仍负责日志、`MTLCommandBufferErrorDomain/code 4` 的 2 秒
driver-rejection 节流、`deviceResetRequested` 发布、异常捕获和实际 commit。
这些保留项是原 ObjC 的高层行为/限制，不在此迁移批次修复；当前 facade 也不
宣称替代多线程提交 guard 或 AGX reset 编排。

## swap-present owner-aware 切片（2026-08-16）

`mglRenderGetCommandBufferOwnerState` 与
`mglRenderPresentDrawableForCommandBufferOwner` 让 gate-on 不再为 swap 的
status 检查和 present 借出 raw current command buffer。gate-off adapter 仍执行
原 ObjC `status` / `presentDrawable:`，包括原有异常传播。

原流程的「读取 current → 检查 status → 必要时 rotate → present」并不是跨线程
原子事务；它依赖调用点位于 GL 线程且 `mtlSwapBuffers:` 外层持有 `METAL_LOCK`。
C++ owner-aware facade 保留这个前提，不新增 owner mutex，也不宣称允许另一线程
同时 reset/detach 同一 owner。这个限制属于原 ObjC 生命周期模型，本轮只记录，
不修正。

## owner-aware encoder facade（2026-08-16）

scissored clear 与 stage copy-back 现在可通过 C++ owner facade 创建 render/blit
encoder；gate-on 成功路径不再把 `CommandBufferOwner.current` 借给
`MGLRenderer.m`。`MGLRenderPassManager` 的默认 render-pass 创建和 MDI current
存在性检查也改为 owner/state snapshot；这只收口 owner 访问边界，不改变提交、
detach、等待或 AGX 异常恢复策略。

在该历史切片中，ObjC adapter 仍为 gate-off 和 C++ 创建失败保留 borrowed current fallback，且
继续依赖 GL 线程与外层 `METAL_LOCK` 的串行调用前提。owner encoder 创建与
status 检查不是跨线程原子事务；本轮不引入 mutex/atomic，也不修复原 ObjC 的
高层生命周期限制。

当前状态已收紧：borrowed getter 只服务 gate-off 且在 gate-on 自行返回 null；
command queue owner create/reset 失败也不会回落 ObjC queue。该 fail-closed 边界由
`check-p4-metalcpp` 静态门固定。

`MGLRenderer+Texture.m` 的 `mtlGenerateMipmaps` 也已使用同一 owner-first blit
adapter；专用 upload/readback command buffer 仍保留 raw helper，因其生命周期不
属于 `MGLRenderPassManager` owner。这个边界仅是适配列，未改变提交、等待或异常
恢复策略。

## 纯 CPU 分类切片边界（2026-08-16）

后续 texture format classification 切片只把 depth/stencil、packed
depth-stencil、GL internal-format 和 sampler data-kind compatibility 表迁入
C++ facade。该切片没有接触 `CommandBufferOwner`、提交/完成回调或 AGX recovery；
因此本文列出的 GL 线程、外层 `METAL_LOCK`、ObjC 异常恢复和 deferred reset 限制
仍全部有效，不能因分类 facade 通过 A/B regression 就宣称 command lifecycle
已经完成迁移。

## GPU timestamp callback 边界（2026-08-16）

`GLMMetalFuncs.mtlGetGPUTimestamp` 的 gate-on 入口已直连
`mglRenderGetGPUTimestamp`，GL timestamp ordering 由 C++
`mglRenderFlush(ctx, true)` 通过 command owner 建立提交/等待边界，再在 C++
采样 timestamp；smoke 明确断言不会调用 legacy `mtlFlush`。

render encoder 结束和 pending draw replay 仍要求上层 GL 语义在进入 callback 前
完成；callback 内的 commit/wait 已走 C++ owner。该边界不改变本文前述 GL 线程
与外层 `METAL_LOCK` 前提。

## Timer-query callback 边界（2026-08-16）

`GLMMetalFuncs.mtlBeginTimerQuery` 和 `mtlEndTimerQuery` 的 gate-on 入口已直连
C++ callback。C++ 侧用 `Renderer.queryStateOwners` 以 `GLMContext` 为 key
保存非拥有的 `QueryStateOwner` 指针，并用独立 mutex 保护 registry；renderer
lifecycle 在 owner 创建后注册、销毁前注销。该 mutex 只保护查找/注册表，不把
query owner 的内部操作或整个 GL 调用变成跨线程安全事务。

两个 callback 使用 C++ command owner flush/wait 保留 timer query 的 GL ordering；
smoke 断言不会回调 legacy `mtlFlush`。它们仍依赖 callback 位于 GL 线程、
renderer 外层 `METAL_LOCK` 已串行化，以及 `QueryStateOwner` 在 callback 期间未
被 dealloc；本轮不扩大跨线程 API 承诺。

## Completion wrapper 的 TSan 边界（2026-08-16）

`mglRenderAddCommandBufferCompletion` 直接注册
`MTL::CommandBufferHandler` block。2026-08-17 修订后，block 只捕获 raw heap
context；注册路径和 completion worker 各持一份原子引用。callback/context/destroy
字段通过同一 mutex 的 `configure`/`complete` 发布和读取，避免 block copy helper
与 worker 竞争，也避免初始化字段在 worker 侧未同步可见。最后一个引用负责销毁
wrapper，注册失败保留原 ABI 的 caller-context ownership。

## Owner transaction 增量边界（2026-08-16）

`mglRenderCommitCommandBufferTransaction` 现已作为 gate-on 的统一提交入口：
它验证 detached submission 与 command buffer 的对应关系，持有 C++ commit guard，
注册 recovery completion，并按请求执行 commit/wait，返回提交前、提交后和完成后的
value-state。compute stage copy-back 已通过该入口完成跨 command buffer 的 detach、
commit 和等待；gate-off 仍走原 ObjC 路径。

transaction 现在也负责在 C++ queue owner 上创建 next current command buffer，
并通过 recovery completion latch deferred-reset request；fence 与
last-submitted wait 共用 `mglRenderWaitCommandBufferState`。它仍不改变 GL
线程和外层 `METAL_LOCK` 前提；ObjC 保留平台日志、最终
`clearProblematicGPUState`/reset hook、gate-off adapter 和少量 borrowed getter
清理工作。

## 本轮明确不修的原 ObjC 限制（2026-08-16）

本轮工作转向 program/resource slot ownership；以下限制只记录，不借助新的
link-time 校验或 sanitizer 结果宣称已经修复：

- gate-on command buffer 的 commit/wait、completion 注册、recovery 计数和
  reset-request latch 已由 C++ owner transaction 负责；ObjC 仍拥有 pending draw
  replay、平台日志、最终 reset hook、problematic GL state 清理和 gate-off adapter。
- swap/present、timer query、GPU timestamp 和 completion callback 仍要求调用在
  GL 线程，并依赖 renderer 外层 `METAL_LOCK` 串行化；owner registry 的 mutex
  只保护查找，不把整个 GL 调用变成跨线程事务。
- detached command-buffer 与 recovery owner 通过显式 retain/release 和 completion
  context 原子引用维持生命周期；这不自动保活 ObjC renderer，也不新增任意线程
  可并发销毁 renderer 的承诺。
- ObjC fallback 与少量 raw render-encoder getter只存在于显式 gate-off adapter；
  gate-on raw current-command-buffer 借用已归零。fallback 随 P5 删除。

这些边界与本轮 `air_geometry_buffer_slot_conflict` 的 link failure 互不等价：
后者只证明用户 buffer 不会撞内部 Metal slot，不证明 command lifecycle 已经
完成 C++ 化。

## 完整 TSan 复核更新（2026-08-17）

旧 `build-tsan-slot` 在 `MGLRenderer+GPURecovery.m:215/220` 的报告已由上述
completion context ownership/publish 修复关闭。2026-08-17 使用全新目录
`build-tsan-final` 重新构建，串行执行：

```sh
DYLD_LIBRARY_PATH=build-tsan-final MGL_USE_METALCPP=0 \
TSAN_OPTIONS=halt_on_error=1 build-tsan-final/test_regression --golden-dir MGL_Golden_Images
DYLD_LIBRARY_PATH=build-tsan-final MGL_USE_METALCPP=1 \
TSAN_OPTIONS=halt_on_error=1 build-tsan-final/test_regression --golden-dir MGL_Golden_Images
```

两门均为 `73 PASS / 0 FAIL / 2 SKIP`，无 ThreadSanitizer 报告。最终 lifecycle
所有权复核加入 transaction recovery 单次应用与 ObjC exception-boundary smoke 后，
再次在同一独立目录重建并复跑，结果不变。该证据覆盖当前
完整 regression 套件的 gate-off/gate-on command completion 路径。2026-08-17
最终审计同时确认 callback census 为 `19 strict / 34 pure adapter / 0 legacy`、
9 个 runtime operation 只落到审计过的 34 个高层 GL 语义 selector 且不直接调用
encoder/command-buffer Metal selector、legacy invoke 只存在于 gate-off bridge；
compute execution plan 与 binding/Draw owner facade 已收口，严格 `id<MTL` 只剩两个
白名单外壳。queue init/reset fail-closed 与 invalid-owner reset smoke 随后也在同一
`build-tsan-final` 上重建并复跑双门，结果不变。因此完整 TSan 不再只是定向 wrapper
证据，而是 P4 终验的一部分。
