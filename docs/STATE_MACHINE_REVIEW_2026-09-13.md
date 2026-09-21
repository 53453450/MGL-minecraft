# C 状态机审查：canonical GL state 与 draw state snapshot

日期：2026-09-13
审查对象：`MGL/` 下 GL 状态子系统与 deferred draw 状态快照路径（工作区 HEAD `a521302`）
性质：只读审查 + 结构度量 + 缺陷证据 + 终态设计与迁移顺序。本文件不改变实现。

---

## 0. 结论摘要

**当前状态机的终态没有被写下来，所以它同时是"权威状态"、"快照"和"键"的叠加体。**
三份表示（live `GLMState` / per-batch `state_snapshot` / 96 字节 `MGLStateKey`）承载同一份信息，
而"可比较、可缓存"这件事实际上已经由 `MGLStateKey` 完成——snapshot 是它的**冗余缓存**。

按最小 CPU 成本的终态应当是：

> **一份 canonical `GLMState`（live）+ 每 draw 一份定长、无 padding、`memcmp` 安全的 `MGLStateKey`。
> snapshot 不是"状态的副本"，而是"把 live canonical state 推到该 key 所需的最小 delta"。**

现在恰好相反：先按 61 KB / batch 把整个状态搬走，再用 key 把 dirty bits 收窄回来。
copy 是宽的那一头，compare 才是应该宽的那一头。

四条主要结论：

| # | 结论 | 证据 |
|---|---|---|
| 1 | snapshot payload 与 key 冗余。全库只有 3 个读者：capture、restore，以及 trace（只读 5 个字段，且都在 key 里） | `draw_command.c:458`、`mgl_batch_flush_restore_encode.c:285`、`mgl_batch_replay_trace.c:102-119` |
| 2 | `MGLStateKey` 有 4 字节 padding 落在 `memcmp` 里，而结构体在栈上未清零——**同一逻辑状态可能偶尔不相等** | 实测 offset 58/60/64，`draw_command.c:2357` |
| 3 | 同一 dirty-bit 语义被手抄两份，其中一份无任何编译期绑定 | `mgl_types_state.h:48-83` vs `mgl_batch_restore.c:143-156` |
| 4 | 96 字节比较 vs 92 KB（实测 `sizeof(GLMState)`）重载 = **1 : 958**；而设计注释仍写 82 KB / 51 KB | 见 §2 度量 |

---

## 1. 终态应该是什么（重述判据）

用户给出的判据，拆成四条可验证的契约：

1. **正确性**：任何时刻 `active_state` 都是 GL 规范意义上的 canonical state；replay 结束后的 live state 与 replay 前逐字节/语义一致。
2. **最低 CPU 成本**：稳态下（状态不变、key 相同）每 draw 的边际成本是 **O(1) 次定长比较**，不是 O(state size) 次拷贝。
3. **稳定**：同一逻辑状态恒产生同一 key 字节序列；不受 padding、未初始化内存、指针地址、调用历史影响。
4. **可比较 / 可缓存**：key 是唯一比较与缓存身份；snapshot 只是把 live state 推到 key 的**派生物**，不参与身份判定。

现在逐条对照。

---

## 2. 度量（实测，非估算）

用 `gcc -I MGL/include -I MGL/include/GL -I MGL/src` 编译探针取得（本机 clang/arm64）：

```
sizeof(GLMState)              = 91952      # 设计注释仍写 82 KB（mgl_types_state.h:289）
hot bytes (copy regions)      = 61240      # 设计注释仍写 ~51 KB
offsetof(GLMState, sync_table)= 31648      # region 1 覆盖 0..31647
skipped HashTable block       =   968      # 11 * sizeof(HashTable)
skipped cold buffer_base      = 29744      # 11 * sizeof(BufferBase)
sizeof(MGLStateKey)           =    96
sizeof(MGLDrawBatch)          =   312
sizeof(MGLCommandBuffer)      =193088      # batches 数组本身只有 39936
sizeof(GLMContextRec)         =394440      # state 8 / draw_command_buffer 92032 / replay_state 302488
sizeof(VertexArray)           =  3512
```

按 batch 数的状态拷贝成本（per frame）：

| batches | snapshot 申请 | 实际 hot-copy |
|---:|---:|---:|
| 8 | 0.76 MB | 0.49 MB |
| 16 | 1.53 MB | 0.98 MB |
| 32 | 3.05 MB | 1.96 MB |
| 64 | 6.11 MB | 3.92 MB |
| 128 | 12.22 MB | 7.84 MB |

> 32 batch / 60 fps ≈ **118 MB/s 的 capture memcpy**，flush 时对称再付一次 restore。
> 这还不含 `mglBatchTeardownReplay` 里 `memcpy(active_state, &pass->saved, sizeof(GLMState))`
> 的 92 KB 整结构回写（`mgl_batch_flush_restore_encode.c:341`）。

**成本比例**：`92 KB reload : 96 B compare = 958 : 1`。

---

## 3. 膨胀的具体形态

### 3.1 snapshot 是 key 的冗余副本，且只进不出

全库调用点：

```text
MGL/src/draw_command.c:458                       mglCopyHotStateFields(batch->state_snapshot, ctx->active_state)     # capture
MGL/src/mgl_batch_flush_restore_encode.c:285     mglCopyHotStateFields(glm_ctx->active_state, batch->state_snapshot)  # restore
MGL/src/mgl_batch_replay_trace.c:102-119         snapshot->program_name / var.program_pipeline_binding /
                                                 vao / framebuffer->name                                # trace 诊断
MGL/src/draw_command.c:1326, 2579-2582, 3755-3773 读 snapshot->vao / 与 vao_snapshot 比较（hazard / dyn-VB 判定）
```

三点结论：

1. **trace 读者只取 5 个字段**，且 `program_name / program_pipeline_name / fbo_name / vao_name`
   已经在 `batch->key` 里——`mgl_batch_replay_trace.c:114-119` 同一函数就同时用了 `batch->key.*`。
   即 trace 不构成保留 61 KB payload 的理由。
2. **hazard 判定读的是 `snapshot->vao` 指针**（`draw_command.c:1326`），而该指针在 capture 时被改写成
   `batch->vao_snapshot`（`draw_command.c:463`）——它要的是 VAO 内容，见 §5.1。
3. 其余字段**没有任何读者**：既不影响合批（由 `batch->key` 决定），也不参与恢复以外的判定。

`batch->key`（96 B）在 `mglAppendDrawCommand` 里已经承担了全部批次合并判定；
`mglBatchRestoreStateFromKey`（`mgl_batch_restore_host.c:71`）已经能从 key 恢复
program / pipeline / VAO / FBO / viewport / scissor——**即 key 单独就足以驱动恢复路径**。

所以 `state_snapshot` 目前的价值只剩"把 key 没有覆盖的那些字段也搬过去"。
问题不在于它搬得不对，而在于**它先无条件全宽搬运，再靠 dirty-bit 收窄**——顺序反了。

### 3.2 一份状态，五种表示

| 表示 | 位置 | 大小 | 用途 | 与 key 的关系 |
|---|---|---|---|---|
| live `GLMState` | `ctx->state` | 92 KB | 权威 | key 由它派生 |
| replay workspace | `ctx->replay_state` | 92 KB | flush pass 隔离 | 全量 memcpy 进来 |
| per-batch snapshot | `batch->state_snapshot` | 92 KB 申请 / 61 KB 写入 | 恢复 | **冗余** |
| per-batch VAO snapshot | `batch->vao_snapshot` | 3.5 KB | VAO 内容冻结 | key 只有 `vao_name`（见 §5.3） |
| `MGLStateKey` | `batch->key` | 96 B | 身份/比较/缓存 | 规范形式 |
| `MGLDrawState` | `batch->draw_state` | 48 B | R3 不可变输入 | key 的子集（`mglDrawStateFromKey`） |

`MGLDrawState` 是 `MGLStateKey` 的严格子集，由 `mglDrawStateFromKey()` 从 key 填充
（`draw_command.h:261`）——即第 6 份表示又是前一份的投影。

### 3.3 同一语义的多套失效机制

- `GLMState.dirty_bits`：15 个域位 + `DIRTY_ALL`（`mgl_types_state.h:48-83`），renderer 消费。
- 独立对象级 dirty：`Buffer.data.dirty_bits`、`Framebuffer.dirty_bits`、`Texture.dirty_bits`、
  `Sampler`、`VertexArray.dirty_bits`——各自独立的位命名空间，且 `DIRTY_VAO_BUFFER_BASE = 0x1`
  与上下文 `DIRTY_VAO = 0x1` 数值撞车。
- 4 个 hash 缓存域 `texture/vertex_layout/render_state/uniform_buffer_dirty` + 4 个 cached hash
  + `active_sampled_texture_unit_mask[4]` 懒构建缓存（`mgl_types_state.h:209-232`）。
- 3 个相等函数：`mglStateKeysEqual`（memcmp）、`...IgnoringUniformRanges`、`...IgnoringDynamicBindings`
  （`draw_command.c:2354 / 2360 / 2675`）。
- dirty-key delta 收窄：`mgl_batch_compute_key_delta_dirty_bits` + `fold_fbo_dirty` + `finish_dirty`
  + `absolute_contract_dirty`（`mgl_batch_restore.c`）。

`mglInvalidateStateHashCachesForDirtyBits` 的注释自己承认了这层耦合的性质：
"Keep invalidation and recomputation on the same masks so renderer-side dirty-bit consumption
cannot silently stale a cache"（`mgl_types_state.h:85-86`）。
也就是**用约定而不是类型**去维持两条独立生命周期的同步。

### 3.4 结构性证据：每加一个字段要改 5 处

| 位置 | 文件 |
|---|---|
| 字段本身 | `mgl_types_state.h` `GLMState` |
| hash 是否包含 | `mglCompute*Hash` |
| key 是否携带 | `draw_command.h` `MGLStateKey` |
| hot/cold 归类 | `mgl_types_buffer.h:64-91` |
| dirty delta 域窄化 | `mgl_batch_restore.c` |

现有 `_Static_assert`（`mgl_types_state.h:314-331`）只保证 region 边界和 hot+cold 覆盖完整，
**它保证的是"拷贝不漏"，而不是"拷贝不多"**：region 是按 `[pack, end)` 这类连续区间整段加的，
所以新字段默认进快照，且没有任何机制能把它移出快照。

---

## 4. 确定的缺陷

### 4.1 `MGLStateKey` 的 padding 被 `memcmp` 读取（稳定性/可比较性契约被破坏）

实测布局：

```
caps_flags      off=58 size=2 end=60
texture_hash    off=64
PADDING gap     = 4 bytes  (offsets 60..63)
sizeof          = 96,  tail padding = 0
```

`mglStateKeysEqual()` 是 `memcmp(a, b, sizeof(MGLStateKey))`（`draw_command.c:2357`）。
而 `mglComputeStateKey` 是**栈上未初始化**的 `MGLStateKey key`（`draw_command.c:4011`），
函数只清零了 `vertex_program_name / fragment_program_name / scissor[4]` 这几个"条件写入"字段
（`draw_command.c:2236-2240`），**offset 60..63 从未被写**。

该函数的注释恰恰声明了相反的设计意图：

> "any cached/padding fields there would be uninitialized garbage and break batch-merge
> comparisons"（`draw_command.c:2282-2284`）

代码里的显式清零注释也写着 "keep memcmp-based equality correct without a full-struct memset"。
结论：**意图正确，实现漏了 padding**。

后果不是崩溃，而是**概率性不相等**：同一逻辑状态，两次栈分配残留不同时 key 不等 →
批次不合并 → 多开 batch → 多付 61 KB snapshot + 多一次 flush。
与"稳定、可比较、可缓存"直接冲突。也解释了 `mglStateKeysEqualIgnoringUniformRanges()` 为什么
改成逐字段比较（`draw_command.c:2373-2392`）——两种相等语义在同一结构体上并存，
只修了其中一个。

**修法（零成本，可立即做）**：把 `mglStateKeysEqual` 改成逐字段比较（或给结构体显式加
`uint8_t _pad[4]` 并纳入 `mglComputeStateKey` 的固定清零清单）。前者顺带获得确定性字节序列。

### 4.2 手抄的 dirty-bit 枚举与权威定义无编译期绑定

`mgl_batch_restore.c:143-156`：

```c
/* Keep in sync with mgl_types_state.h dirty* enum / DIRTY_* masks. */
enum {
    MGL_BATCH_DIRTY_VAO = 1u << 0,
    ...
};
```

实测两份数值当前一致（`DIRTY_VAO=1 DIRTY_BUFFER=4 DIRTY_TEX=8 ... DIRTY_BUFFER_BASE=16384`），
但没有任何 `_Static_assert` 绑定。旁边 `mgl_batch_restore_full_dirty_bits()`（:158）又是第三份
手写清单（未含 `DIRTY_FBO / DIRTY_DRAWABLE / DIRTY_SHADER`，FBO 靠 `fold_fbo_dirty` 单独折叠）。
这是 §3.4 五处同步点的具体实例：**权威定义改一位，这里静默漂移**。

`mgl_batch_restore.c` 是纯 C 且只 `#include "mgl_batch_restore.h"`（为了能单独链进
`test_batch_restore`），这正是它抄一份而不敢 include 权威头的原因。修法：把掩码定义搬到一个
无依赖的小头（例如 `mgl_batch_dirty_bits.h`），权威头与 batch 头都 include 它，加
`_Static_assert(MGL_BATCH_DIRTY_VAO == DIRTY_VAO, ...)`。

### 4.3 注释里的成本模型已经失真（12% / 22% 漂移）

`mgl_types_state.h:289-300` 写 "GLMState is 82KB … only ~51KB … saves ~37.5% (31.6KB)"。
实测 91952 / 61240 / 省 30712（33.4%）。`draw_command.c:455` 的
"~51KB vs 82KB full"、`draw_command.c:443` 的块注释同理。
更重要的是 `MGL_PERF_ADD(g_mglSnapshotBytesAllocatedSinceSwap, sizeof(GLMState) + sizeof(VertexArray))`
（`draw_command.c:471`）上报的是**申请量**（95.5 KB），而实际写入是 61 KB——
现有帧计数器给出的"snapshot 成本"本身偏高约 56%。

### 4.4 replay workspace 吞掉 GL 错误队列（latent，改变行为）

`GLMState` 的 region 1 覆盖 `[0, sync_table)`，其中 offset 32 起是
`error_queue[16] / error_head / error_count`——**GL 错误队列在快照 payload 里**。

当前 `ctx->active_state` 在 flush pass 期间指向 `replay_state`
（`mglCoreActivateReplayState`，`mgl_batch_flush_restore_encode.c:193`），
而 `STATE(_VAR_)` 展开为 `ctx->active_state->_VAR_`（`glm_context.h:71`），
错误队列读写全部走 `STATE(...)`（`error.c:49-66, 181-193`）。

因此 replay 期间产生的错误写进 `replay_state`，而 `mglCoreActivateReplayState` 下一次
pass 开头 `memcpy(&ctx->replay_state, &ctx->state, sizeof(...))`（`mgl_renderer_core_state.c:30`）
会把它们整体覆盖；`mglCoreRestoreLiveActiveState` 只切指针，不回拷。
→ **replay 期间产生的 GL 错误对客户端不可见，且被静默丢弃。**

这是"把 GL 规范状态和渲染器工作区塞进同一个 struct"的直接代价：
任何在 replay 期间写状态的路径，只要不在 snapshot 恢复路径上，就会进黑洞。
`error_queue` 在快照里也毫无用处（capture 时搬走、restore 到 workspace），属于 §3.1 的冗余 payload。

### 4.5 `replay_state` 的迁移状态（92 KB × 1 的固定开销）

`glm_context.h:121-124` 的注释写着"batch replay may later redirect to replay_state once
remaining ctx->state readers migrate"，说明 R3 只做了一半：
`active_state` 索引已经引入，但 92 KB 的 `replay_state` 副本**仍然是 pass 级全量 memcpy**，
没有变成增量/按域工作区。要么补完（按域工作区），要么退回去。

### 4.6 `MGLCommandBuffer` 193 KB 内联在 context 里，其中 95% 是防御性定容数组

实测逐字段（`MGL/include/draw_command.h:284-321`）：

```
sizeof(MGLCommandBuffer) = 193088
  batches[128]                        39936   # 真正的载荷
  buffer_read_ranges[4096]            98304   # 51%
  buffer_read_range_next[4096]        16384
  buffer_read_range_bucket[1024]       4096
  sampler_snapshot_sets[256]          12800
  sampler_snapshot_keys[128]           7680
  sampler_snapshot_set_index[512]      1024
  sampler_snapshot_key_index[256]       512
  texture_read_index[1024]             4096
  texture_read_objects[512]            4096
  texture_write_objects[256]           2048
  texture_write_index[512]             2048
  --------------------------------------------
  小计                               188424   # 逐字段累加，余数 ~4.7 KB 为零散标量/对齐
```

真正的批次载荷（`batches[]`）只占 **20.7%**（39936 / 193088）；其余 153 KB 是
pending buffer range / texture hazard 追踪表和 sampler 侧表的**定容内联数组**。
`MGLCommandBuffer` 作为 by-value 成员内联在 `GLMContextRec`（`glm_context.h:136`），
所以每个 GL context 无条件付这 193 KB（占 `GLMContextRec` 394 KB 的 **49%**）。

这与"最低 CPU 成本"无关，但与"膨胀"直接相关，属可回收的静态内存债：
hazard 表与 sampler 侧表都可以按需增长（容量上限不变），
或至少把低频的 `buffer_read_ranges` 容量与真实 pending draw 数挂钩。
注意 `mglFlushPendingDraws*` 系列的正确性依赖这些表，**收缩容量前必须先量出真实峰值占用**，
不能凭"看起来很大"就改。

---

## 5. 需要区别对待的：不是所有"多出来的表示"都该删

审查中发现的**合理**设计，迁移时不要误伤：

### 5.1 VAO snapshot 目前有真实理由

`batch->vao_snapshot`（3.5 KB）复制的是 VAO **内容**，因为 `MGLStateKey` 只带 `vao_name`
（整型名，`draw_command.h:197`）。VAO 在记录之后被客户端改写时，仅靠名字无法重建当时的
attribute 布局。`source_vao` 指针用于 hazard 判定（`draw_command.c:1326`）。
要消掉这份快照，前提是引入 **VAO 版本号/世代**（key 带 `vao_name + generation`）或把布局
在记录时固化成不可变 plan。这是个有界的设计题，不是顺手能删的东西。

### 5.2 sampler snapshot 与 key 是不同维度

`sampler_snapshot_id`（`MGLSamplerSnapshotKey/Set`）解决的是采样器参数矩阵的组合爆炸，
不是状态身份问题。它的分页/索引结构可以搬到堆上并按需增长（§4.6），但机制本身该留。

### 5.3 三层 key 相等不是纯粹的历史包袱

`IgnoringUniformRanges` 表达"只有 per-draw UBO offset 不同，可以合批"，
`IgnoringDynamicBindings` 表达"只有 VAO/UBO/纹理绑定不同，可走 per-draw override"。
两者都对应真实的 MC 负载形态。问题不在有三个，而在**没有一个把"哪些字段属于身份、哪些属于
per-draw delta"写成数据**——现在是三份手写字段列表。终态里应该是同一份字段表驱动这三个投影。

---

## 6. 终态设计

### 6.1 契约

```
canonical state  : 唯一权威 = live GLMState。任何时刻可被 glGet* 完整回答。
draw identity    : MGLStateKey —— 定长、无 padding、逐字段相等、确定性字节序列。
draw delta       : 记录期捕获的 per-draw 增量（uniform range / dynamic binding / stream 数据）。
```

关键反转：**key 是快照，snapshot 不再存在**。
批次恢复 = "把 live canonical state 推到 key"，而不是"把快照灌回 state"。

### 6.2 三条不变量

1. **比较必须在 key 上，且 key 必须 `memcmp` 安全**——显式 padding 字段 + 固定清零，
   或逐字段比较。加 `_Static_assert` 锁定布局（现有 offsetof 断言的正确用法可以照搬）。
2. **身份与增量分离**——key 只放身份；per-draw 差异一律走 delta 通道。
   这样"同 key 合批"永远安全，不需要靠 958:1 的宽拷贝去兜底。
3. **失效只有一个来源**——`dirty_bits` 由权威头定义，掩码跨 TU 加 `_Static_assert` 绑定，
   禁止任何第二份手抄清单（§4.2）。

### 6.3 恢复路径

```
ReplayBatch(b):
    if (b->key == current_key)                     -> 什么都不做（已有 skip 路径）
    else if (delta_from_key(current_state, b->key) 可表达) -> 只应用 delta
    else                                            -> 按域从 key 重建
```

`mglBatchRestoreStateFromKey` 已经是"按域从 key 重建"的雏形（program/pipeline/VAO/FBO/viewport/scissor），
终态是把它补全成唯一的恢复入口，删掉 snapshot 分支。

### 6.4 成本模型（终态 vs 现状）

| 操作 | 现状 | 终态 |
|---|---|---|
| 新 batch capture | 95 KB 申请 + 61 KB memcpy | 96 B key 写入 |
| 同 key 合并判定 | 96 B memcmp（含 padding 抖动） | 96 B 定长比较 |
| per-batch restore | 61 KB memcpy + clean + 收窄 | delta 应用（稳态 0） |
| pass teardown | 92 KB `sizeof(GLMState)` memcpy | 0（无 workspace 副本）或按域回拷 |
| 新增 state 字段 | 改 5 处 | 改 2 处（字段 + 是否入 key） |

---

## 7. 迁移顺序（每步独立可验收）

每一步都要求"旧的宽路径保留为 oracle"，用 §8 的计数器做 A/B，**不做无 oracle 的删除**——
这与 `docs/OBJC_CATEGORY_DISMANTLE_TODO.md` §O7.5 的既有验收口径一致。

**M0 — 修正 key 的确定性与布局绑定（无行为风险，先做）**
1. `mglStateKeysEqual` 改逐字段比较；或显式 padding 字段并纳入固定清零。
2. 加 `_Static_assert(offsetof/texture_hash 无 gap)` 锁定布局，防止回退。
3. 手抄 dirty 枚举 → 共享小头 + `_Static_assert` 绑定（§4.2）。
4. 修 §4.3 的注释与 `g_mglSnapshotBytesAllocatedSinceSwap` 上报口径（改报实际写入字节）。
   验收：`make test-batch-restore test-batch-issue test-dirty-hash`，
   并确认 `g_mglMergeRejectStateDiffersSinceSwap` 在固定 trace 下**只降不升**。

**M1 — 让 snapshot 可被逐域裁剪**
把 region-based 拷贝换成字段表驱动（同一份表同时产出 key、hash 域、hot 集）。
此步不改行为，只把"拷贝不多"变成可表达、可测量的性质。

**M2 — capture 侧先按 key 判重，再决定要不要快照**
同 key 复用 batch 时不再新建 snapshot（现路径已部分做到），
并让 `mglInitializeBatchStateSnapshot` 只复制"key 未覆盖域"。
验收：`g_mglSnapshotBytesAllocatedSinceSwap` 与 capture memcpy 字节数按比例下降。

**M3 — restore 侧走 key 优先**
`mglBatchRestoreStateForBatch` 的 snapshot 分支改为 fallback；A/B 用
`g_mglSameKeyOracleWouldSkipSinceSwap` 与 replay trace 文本逐条 diff。
验收：`mgl_batch_flush_restore_encode.c` 的 trace 输出与旧实现逐条一致。

**M4 — 删除 snapshot payload，并处理 workspace**
1. 删 `state_snapshot`（保留 `vao_snapshot` 直到 §5.1 的世代方案落地）。
2. `replay_state` 二选一：按域工作区，或退回单 workspace（省 92 KB 与一次全量 memcpy）。
3. 顺带修 §4.4：`error_queue` 不进任何快照/工作区，错误始终写 live。

**M5 — command buffer 的冷侧表下沉堆**
sampler snapshot 结构改为堆上按需增长，回收约 145 KB/context 静态占用。

---

## 8. 验收与可观测性（无需新建基础设施）

现有帧计数器已经覆盖了本次审查需要的全部口径（`mgl_frame_activity.h:168-196`）：

| 计数器 | 用途 |
|---|---|
| `g_mglSnapshotAllocationCountSinceSwap` / `g_mglSnapshotBytesAllocatedSinceSwap` | capture 侧成本；M0 先修口径 |
| `g_mglReplayMemcpyCountSinceSwap` | restore 侧 memcpy 次数 |
| `g_mglSameKeyRestoreSkipsSinceSwap` / `g_mglSameKeyOracleWouldSkipSinceSwap` | M3 的 A/B 判据 |
| `g_mglMergeRejectStateDiffersSinceSwap` | M0 的直接受益指标（padding 抖动应体现在这里） |
| dirty-key delta 收窄计数（`mgl_frame_activity.c:250-262` 已 dump） | M2/M3 的窄化率 |

回归面：`test-batch-restore` / `test-batch-issue` / `test-dirty-hash` / `test-batch-path` /
`test-batch-hazard` / `test-batch-icb` / `test-arch-correctness`，
外加 hotspot（非通过集合逐条 diff 为空）与 MC 实机 FPS。

---

## 9. 证据边界

- 本文数字均由本机探针实测（结构体大小/offset、dirty 位数值、拷贝字节推导），
  探针不落在仓库内。
- §3.1 "snapshot 无其它读者"基于全库符号 grep（`state_snapshot` / `mglCopyHotStateFields`）；
  若有通过 `void *` 泛型存储后再 cast 读取的路径，本次审查未穷尽——M1 落地前应再确认一次。
- §4.4 是**静态推演**（active_state 指向 + `STATE` 宏展开 + error.c 写入点），
  未构造运行时反例。建议用一个"replay 期间触发状态修复警告 + 随后 glGetError"的探针坐实。
- 未测量 GPU 侧影响；本文只讨论 CPU 成本与状态正确性。
- 未运行完整回归与 CTS。
