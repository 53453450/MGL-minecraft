# C 状态机 / deferred draw：优化与数据流加固 TODO

日期：2026-09-13
基线：`MGL-minecraft` 工作区（`draw_command.c` 等状态子系统文件当时未被该轮 ObjC 重构改动）
性质：**只读审查产出的待办清单**。本文件不含任何代码改动，所有源码修改均已撤销。
前置审查：结构与缺陷详见 `docs/STATE_MACHINE_REVIEW_2026-09-13.md`。

## 0. 两条硬约束

1. **GL 2.x / legacy 兼容面是需求，不是技术债。**
   `mgl_legacy_compat.c`（GLSL 源码级重写：pre-3.30 语法 → core）、固定管线 `GLMCaps`、
   `glGet*` / `glIsEnabled` 的可查询性都必须保留。
   清理任何"看起来没人读"的状态前，先过判据：**`glGet`/`glIsEnabled` 的 switch 是否覆盖
   + `mgl_legacy_compat.c` 是否引用 + 是否属于 GL 规范要求的兼容行为**。
2. **优化必须带 oracle。** 每条优化先保留旧路径，用 §5 的既有计数器做 A/B。
   不做"无法证明没改行为"的性能改动——OpenGL 隐式状态机最容易在这里出事。

## 1. 症状到机制的映射现状

| # | 症状 | 状态 |
|---|---|---|
| 1 | Sodium 状态错乱 | 有候选（T1/T2 同源：对象身份丢失导致合批越界） |
| 2 | FBO 切换异常 | 有协议缺口（T3），**影响面未坐实** |
| 3 | texture binding 泄漏 | **已定位到机制**（T1 已确认；T2 同源第二入口） |
| 4 | blend/depth 状态污染 | **未找到静态机制**——`state.c` 约 30 个 setter 全部标记了覆盖 render_state 哈希的位；需要运行时证据（§4.2） |
| 5 | deferred draw 合并错误 | 有候选（T1、T4，以及 §3 的 key 侧问题） |
| 6 | encoder 状态错误 | **已定位**（§2.1 `active_state` 代理分裂，错误队列实例已坐实） |

---

## 2. 已确认的缺陷（按优先级）

### P0-1 `active_state` 代理分裂：replay 期间的写入永久丢失

`STATE(x)` 展开为 `ctx->active_state->x`（`glm_context.h:71`），而 `active_state`
在 flush pass 期间指向 `replay_state`（`mgl_batch_flush_restore_encode.c:193` →
`mgl_renderer_core_state.c:30-32`）。后果：**同一符号在不同时刻指向两个对象，调用点看不出区别**；
replay 期间的写入落在 workspace，pass 结束只切指针不回拷，下次 pass 开头被整块覆盖。

**已坐实的实例（GL 错误队列）**：

```
replay 编码报错: mgl_batch_issue_encode.c:136,172 -> mgl_draw_encode.cpp:88 mglDispatchError
              -> error_func -> 写 STATE(error_queue)（error.c:181-196）
写入落在 replay_state
pass 结束: mglCoreRestoreLiveActiveState 只切指针（mgl_renderer_core_state.c:35-44）
下次 pass: memcpy(&replay_state, &state, sizeof)（:30）整块覆盖
=> 该错误对客户端永久不可见
```

- [x] **T0-1** 让 replay 显式接收 `GLMState *` 参数，取消靠全局 `active_state` 隐式切换；
      在 pass 边界用已有的 `mglCoreAssertDualProxy` 断言。
      **2026-09-21**：
      - restore / from-key / `FCtx.replay=&pass->saved` 显式 `GLMState *`；
      - `MGLEncodeContext.state` 由 flush `fEnc` 写入；四个 issue 入口
        `mglEncodeContextRequireReplayState`（须 `state == active_state`）；
      - `processGLStateLocked` 入口 dual-proxy 断言；读路径经 `mglPdState`
        （优先 `core->activeState`）。
      - `STATE()` 仍是 **record** 宏；flush 簇本身不经 `STATE()`。
        activate 仍双写 proxy，供尚未显式传参的 bind 路径对齐 workspace。
- [x] **T0-2** `error_queue` 不进任何快照/工作区，错误始终写 live。
      **2026-09-21**：`error.c` 经 `LIVE_STATE` 读写 queue；`glGetError` 只看 live；
      单槽 `error` 镜像到 `active_state` 以免 replay 内 `STATE(error)` 探针失明。
- [x] **T0-3** 运行期哨兵：统计 replay 期间有多少写入落在 workspace 上（量化同类问题的总量）。
      **2026-09-21**：`g_mglReplayErrorRedirectsSinceSwap` 在 `error_func` 于 replay
      workspace 激活时递增（其它 writer 可随后挂同一计数器 / G7）。
- **未坐实**：未构造运行时反例。建议探针＝"replay 期间触发 `firstVertex < 0` 的
  line-loop 退化路径 + 随后 `glGetError`"。

### P0-2 纹理对象身份丢失（= texture binding 泄漏的直接机制）

`mglComputeTextureHash` 按**裸指针**哈希纹理（`draw_command.c:1877` 原
`tex_ptr`；typed 槽同处 `:1907-1908`），而缓冲区/VAO 混入了名字
（`mglHashBufferPointerName`，`:2047-2052`，用 `ptr ^ (name<<1)`）。
纹理是 `malloc` / `free`（`textures.c:222` / `:1197`），所以：

```
删除纹理 -> free -> 新建纹理拿到同一地址
=> texture_hash 完全相同
=> mglStateKeysEqual 是整结构 memcmp（draw_command.c:2357）判等
=> 新纹理的 draw 合并进持有旧纹理的 batch
=> replay 绑到陈旧纹理  == texture binding 泄漏
```

`draw_command.c:4050-4061` 那段"哈希碰撞概率可忽略"的注释**假设了指针身份等于对象身份**，
正是这个假设失效。

- [x] **T1-1（最小改动）** 纹理移到与缓冲区/VAO 相同的约定：哈希混入 name。
      两条路径都要改：`active_textures[unit]`（`:1877`）与 `texture_units[unit].textures[t]`（`:1907`）。
      **2026-09-21**：`mglComputeTextureHash` 经 `mglTexturePointerNameBits` 混入 `ptr ^ (name<<1)`。
- [x] **T1-2（彻底）** 引入 per-object generation（`HashTable.deletion_generation` 已为此存在，
      `hash_table.h:41-43`）。**T1-1 不能覆盖"地址+名字同时复用"的残留情形**——
      名字会被 `getNewName` 回收（`hash_table.c:661-676`），此时地址与名字都相同，
      必须靠 generation 区分。
      **2026-09-21**：`Texture` / `Sampler` 增加 `identity_generation`（创建时取自对应
      table 的 `deletion_generation`）；哈希混入 `<< 17`。
- [x] **T1-3** 同类审查 `Sampler`（`mglComputeTextureHash` 也用 `sampler_ptr`，`:1912-1913`）；
      **2026-09-21**：经 `mglSamplerPointerNameBits` 混入 name + generation；image unit 路径与 helper 对齐。

### P0-3 动态纹理绑定门只有指针比较（P0-2 的第二入口）

`mglCaptureDynamicTextureBindings`（`draw_command.c:2737-2758`）比较
`snapshot->texture_units[u].textures[t]` 与当前指针，**没有 name / generation 校验**。

- [x] **T2-1** 加 name（或 generation）校验，使同地址重绑不再被判为"未变化"。
      注意纹理是唯一缺少 `Program` / `Framebuffer` 那种名字校验模式的主要对象类型
      （对照 `MGLRenderer.m:469-486` 的 `mglResolveProgramFromState`）。
      **2026-09-21**：capture 已把 live `texture_name` 写入 `MGLDynamicTextureBinding`；
      快照指针可能悬空，不能对 snapshot 对象解引用读 name。合批侧靠 T1-1/T1-2 的
      哈希（name+generation）拒绝同址复用合并。门上仍保留指针比较并加了注释。

### P1-1 `glBindFramebuffer` 的 read-target 分支不落任何 dirty 标记（协议缺口）

- `STATE(readbuffer)` 写在 `framebuffers.c:970`；`STATE(read_buffer)` 写在 `:707-716`；
- 而标记在 `:1090-1098`，只看 `drawTargetChanged`。

- [x] **T3-1** 补标记，或**显式注释说明为何不需要**（二选一，不要留沉默缺口）。
      **2026-09-21**：在 `mglBindFramebuffer` 的 draw-dirty 块后注明 read-only 目标
      故意不置 dirty（deferred encode 只消费 draw FBO）。
- **影响面已复核为"当前无"**：deferred encoder 不读 `read_buffer`（只有 trace-log 在
  `RenderPass` 里用），且 `rendering.c:1315-1320` 的注释表明这是有意设计。
  所以这是**一致性问题，不是已坐实缺陷**——不要写成 bug。

### P1-2 双代理回拷丢弃哈希缓存标志

`mglBatchTeardownReplay`（`mgl_batch_flush_restore_encode.c:336`）只从 workspace 同步
11 张哈希表（`mgl_batch_replay.cpp:554-568`），**不同步 replay 期间被改的 4 个 `*_dirty`
标志**。`mgl_batch_replay.cpp:1187` 就是一个具体写入点（replay 失效了纹理哈希缓存）。

- [x] **T4-1** 回拷时一并同步哈希缓存标志；或论证 replay 期改它们不会影响 live。
      **2026-09-21**：`mgl_batch_replay_sync_hash_tables_from_replay` 对 4 个
      `*_dirty` 做 OR（不可覆盖，否则 flush 会抹掉 live 侧失效；见
      `test_dirty_hash`）；workspace 若清掉 sampled-mask 有效位则同步失效 live。
      不回拷 cached digest。
- **未坐实运行时后果**：需要构造"replay 期改哈希缓存"的场景。

---

## 3. key / 快照侧的正确性问题

### P1-3 `MGLStateKey` 的对齐洞落在 `memcmp` 里（潜在地雷）

实测布局：字段 0–59 致密，**offset 60..63 是 4 字节对齐洞**（`uint64_t texture_hash` 的
对齐要求），`sizeof` 96，尾部无 padding。`mglStateKeysEqual` 是
`memcmp(a, b, sizeof(MGLStateKey))`（`draw_command.c:2357`）；`mglComputeStateKey` 收的是
**栈上未初始化**的 `MGLStateKey key`（`:4011`），只清零了条件写入字段（`:2233-2238`）。

**严重性必须说准：当前是良性的。** 唯一进入存储的路径先在 `:4134` 做了
`memset(batch, 0, sizeof(*batch))`，再 `batch->key = key`（`:4136`），
所以 `memcmp` 两侧的 padding 恰好都是 0。另一条新建 batch 路径（`:4153-4155`）同样先 memset。

- [x] **T5-1** 给 offset 60 加显式 `uint32_t _reserved;`（迫使命中写入清单），
      或 `mglComputeStateKey` 开头 `memset(out, 0, sizeof(*out))`；
      再加 `_Static_assert` 锁死"无隐式洞"。
      **这一步不修 bug，是拆地雷**——不要写成"修了一个合批错误"。
      **2026-09-21 核对已落地**：`MGLStateKey._padding` @60 + `mglComputeStateKey`
      显式清零 + `_Static_assert`（`draw_command.h`）。
- 为什么仍要做：任何新路径把栈上 key 存进未清零的存储（第二个 command buffer、
  调试/测试 harness），就会比较 4 个不确定字节。

### P1-4 `GLMCaps` 的 18 个未初始化字段进入 `render_state_hash`

context 是 `malloc` 分配且 `state` 无 `memset`（`glm_context.c:231`，全文件无 `calloc`）；
`createGLMContext` 只写了 29 处 caps 字段。**实测恰好 18 个字段全树 0 引用**：

```
alpha_test      auto_normal     color_array     color_material
edge_flag_array fog             index_array     lighting
line_stipple    normal_array    normalize       point_smooth
polygon_stipple texture_coord_array
texture_gen_s   texture_gen_t   texture_gen_r   texture_gen_q
```

而 `mglComputeRenderStateHash` 把**整个 86 字节 caps 按字节哈希**
（`draw_command.c:2119-2121`）→ 18 个来自堆的不确定字节参与 `cached_render_state_hash`。

**严重性也说准**：哈希**不是**相等判据（合批决策是 96 字节 `memcmp`），所以
**不会**错误合批。真实影响是**跨运行不可复现**——进程内稳定，跨进程每次都不同。
这让"固定 trace 下计数器对比"这类跨运行验收失去可信度。

- [x] **T6-1** 在 `createGLMContext` 里按 GL 规范默认值显式初始化 `GLMCaps`
      （18 个固定管线启用位默认 `GL_FALSE`），并对 `state` / `replay_state` 补 `memset` 兜底；
      context 可顺手改 `calloc`。
      **2026-09-21**：18 个 legacy caps 显式赋 `false`；整 context `calloc`。
- **约束**：这 18 个字段是 **GL 2.x 兼容面**（`GL_LIGHTING`/`GL_FOG`/`GL_NORMALIZE`/
      `GL_ALPHA_TEST`/…）——**只初始化、不删除、不从哈希里摘**。
      当前它们还不可达（`glEnable`/`glIsEnabled` 的 switch 不含，
      `MGL/src/state.c:134/199/710`；`glGet` 只见 `get.c:515-516` 两项），属**休眠兼容面**，
      正因如此修法要选"初始化"而非"删除"，以免堵死将来接通固定管线语义的路。
- [x] **T6-2** 第二道防线：context 改 `calloc`，防的是**将来**新增字段忘记初始化。
      **2026-09-21**：`createGLMContext` 用 `calloc(1, sizeof(GLMContextRec))`。

### P2-1 手抄的 dirty-bit 枚举无编译期绑定

`mgl_batch_restore.c:143-156` 手抄了同一组 11 个位，注释写着
"Keep in sync with mgl_types_state.h dirty* enum / DIRTY_* masks."，但**无任何 `_Static_assert`**。
`mgl_batch_restore_full_dirty_bits()`（`:158`）又是第三份手写清单。

- [x] **T7-1** 把掩码定义搬到无依赖小头（如 `mgl_batch_dirty_bits.h`），
      权威头与 batch 头都 include，加 `_Static_assert(MGL_BATCH_DIRTY_VAO == DIRTY_VAO, ...)`。
      动机：`mgl_batch_restore.c` 为了能单独链进 `test_batch_restore` 才抄一份而不敢 include 权威头。
      **2026-09-21 核对已落地**：`mgl_dirty_bits.h` 为单一来源，含 `_Static_assert`。

### P2-2 `MGLStateKey` 的 padding → 见 P1-3（同处）

### P2-3 死位

`DIRTY_SHADER` / `DIRTY_DRAWABLE` 无写者、无读者（仅 `mgl_state_log.c` formatter 引用）。
已确认**不属于**兼容面（不在 caps 里、不被 legacy 路径使用）。

- [x] **T8-1** 删除或补上消费者。
      **2026-09-21**：不删。`DIRTY_DRAWABLE` 已有写者（sRGB drawable）；
      `DIRTY_SHADER` 与 `Shader_t.dirty_bits` 共用宏，batch restore 不折叠二者。
      在 `mgl_dirty_bits.h` 注明保留原因。

---

## 4. 快照 / workspace 的冗余（结构膨胀）

### P2-4 快照 payload 约 50% 是确定性死区

`mglCopyHotStateFields` 实测写 61,240 B（`sizeof(GLMState)` = 91,952）。确定无消费者：

| 字段 | 字节 | 证据 |
|---|---|---|
| 11 个 cold `buffer_base` 类型 | 29,744 | `MGL_SNAPSHOT_COLD_BUFFER_BASE_TYPES` 列出的 11 个类型**全部 0 次** `buffer_base[T]` 访问 |
| 11 张 HashTable 本体 | 968 | `mgl_batch_replay_copy_object_hash_tables`（`mgl_batch_replay.cpp:536-552`）在每个 batch 恢复后从 `savedState` 重拷 10 张表，覆盖快照值 |
| hash-cache 块 + `error_queue` | ~156 | 只在 record 期被读；`error_queue` 见 P0-1 |

- [x] **T9-1** 把 region-based 拷贝换成字段表驱动（同一份表同时产出 key、hash 域、hot 集）。
      **前提**：region 是按连续区间加的，所以"跳过一段连续内存"与"精确跳过若干字段"
      在实现上不是一回事，必须先有字段表。
      **2026-09-21**：`kHotRegions[]` + `mgl_state_identity_table.h`（`MGL_STATE_IDENTITY_ROWS`
      标注 key / hash 域 / hot）。热拷贝仍 region memcpy（逐字段小拷不做）。
      **收尾**：G4 CHECK 1b 强制 identity 表 key 列 ↔ `MGLStateKey` 双射，且每个
      key 列均被 `mglComputeStateKey` 写入。表是 oracle/门禁，不 codegen 运行时 writer
      （条件字段与 dirty-hash 缓存仍手写）。`make test-state-dataflow` PASS。
- [ ] **T9-2** arena 申请量对齐实际写入量：现在 `arenaAlloc(arena, sizeof(GLMState))`
      申请 full sizeof 却只写 hot bytes，**每 batch 白占 ~33% arena**。
      **阻塞**：pending 期 `state_snapshot` 被以 `GLMState *` 直读；打包需改全部读点。
      计数器已按 hot bytes 上报。软关闭撤销，**保持开放**（打包布局另开）。
- 注意 §3 的阻塞清单：replay 要读 128 项 unit 数组、buffer map list、
  6 张 HashTable 的查找能力，**key 不能取代"数据在场"**——终态是把快照缩到
  "key 未覆盖域"，不是删掉它。详见前置审查 §6.3。

### P2-5 `MGLDrawBatch.draw_state` 被写 3 处、读 0 处

`draw_command.c:4138 / 4158 / 4210` 三处调 `mglDrawStateFromKey` 填充，
全库无任何读取点（其余 `draw_state` 命中是 `mgl_fake_draw_executor.c` 形参、
`reset_tess_draw_state` 回调、`mgl_backend_handles.h` 无关字段）。

- [x] **T10-1** 删除，或接通其声称的 R3 用途。
      **2026-09-21**：删除 `MGLDrawBatch.draw_state` 与 `mglDrawStateFromKey`
      （写 3 / 读 0；身份由 `batch->key` 承担）。
- **价值**：这是仓库已有的"key 接管身份后，快照变死重量"的**先例**，
      是 T9 的论证依据。

### P2-6 每 flush pass 三次 92 KB 整结构拷贝

```
pass 开始: memcpy(&pass->saved, active_state, sizeof(pass->saved))       flush_restore_encode.c
batch 恢复: mglCoreActivateReplayState(workspace=&pass->saved)           无二次整拷（T11-1）
pass 结束: sync hash tables + dirty OR；切回 live（不整拷回 live）
```

`pass->saved` 本身是完整 `GLMState`（`mgl_batch_public.h`），
与 live 构成**两份**固定副本；`ctx->replay_state` 仅作无 pass 时的回退槽。

- [x] **T11-1** `replay_state` 二选一：按域工作区，或退回单 workspace。
      **无论选哪条，必须保证 6 张对象 HashTable 在 replay 期间仍可查**——
      真正在起作用的是 `mgl_batch_replay_copy_object_hash_tables` 那次按 batch 的 10 表重同步
      （见 P2-4），整块 memcpy 只是顺带提供初值。破坏这点会以极隐蔽方式
      损坏 program / VAO / FBO / texture / sampler 解析。
      **2026-09-21**：选单 workspace——`mglBatchFlushBegin` 只拷一次到 `pass->saved`，
      `mglCoreActivateReplayState(core, ctx, &pass->saved)` 不再二次拷进
      `ctx->replay_state`。`mglCtxActiveIsReplayWorkspace` 改为
      `active_state != &ctx->state`。teardown 仍从当前 workspace sync 表 + OR dirty。
      `test-dirty-hash` / `test-batch-restore` / `test-batch-issue` PASS。
      `ctx->replay_state` 保留为 NULL-workspace 回退，未删字段（避 ABI 抖动）。
- [x] **T11-2** 把 `g_mglReplayMemcpyCountSinceSwap` 从"次数"改成"字节数"。
      **2026-09-21**：restore 路径 `MGL_PERF_ADD(..., mglSnapshotHotStateBytes())`。

### P2-7 `GLMContextRec` / `MGLCommandBuffer` 静态占用

`MGLCommandBuffer` = 193,088 B 内联在 `GLMContextRec`（`glm_context.h:136`），
占后者 ~49%。其中**批次载荷 `batches[]` 只占 20.7%**，其余 153 KB 是定容数组：

```
buffer_read_ranges[4096]   98,304   buffer_read_range_next[4096]  16,384
sampler_snapshot_sets[256] 12,800   sampler_snapshot_keys[128]     7,680
...（其余见注释）
```

- [x] **T12-1** hazard 表与 sampler 侧表下沉到堆、按需增长（容量上限不变），
      回收约 140 KB/context。**改前先量真实峰值占用**。
      **2026-09-21**：水位 API；`buffer_read_ranges`/`next` 堆化（≈−114 KiB）。
      **续**：`sampler_snapshot_keys`/`sets`、`texture_write/read_objects` 亦堆上按需增长；
      定容 hash index 仍内联。静态 payload 合计约 −134 KiB+；index 留内联。
      reset `free` 全部堆指针后 `memset`。`test-dirty-hash` / `test-batch-*` PASS。
- [x] **T12-2** 容量配比可疑：`sampler_snapshot_sets[256]` vs keys[128]。
      **2026-09-21**：语料峰值 0/0；有意保留 256，等生产非零水位再评。

**注意（反例，避免误改）**：`mglResetCommandBufferForContext` 的
整结构 `memset`（每 flush 一次）仍负责清 `buffer_read_range_bucket` /
`texture_write_index` / `texture_read_index` / sampler 侧表（"0 = 空槽"）。
T12-1 后须**先 free 堆上 `buffer_read_ranges`/`next`** 再 memset。
省掉 memset 需要给每张表单独维护 `count` 并在查询时用 count 界定，属结构性改动；
收益（≈数微秒/帧）远小于风险。**不做。**

---

## 5. 昂贵 CPU 路径（排序，含复核状态）

| # | 路径 | file:line | 每调用成本 | 频率 | 复核 |
|---|---|---|---|---|---|
| 1 | 索引范围**三次** O(count) 遍历 | `draw_command.c:3664`、`:3040`、`:3875` | `n = cmd->count` | **每个 indexed draw** | ✅ 已确认三次遍历同一件事 |
| 2 | 每 flush pass 整结构拷贝 ×3 | `flush_restore_encode.c:191`、`:342`、`core_state.c:30` | 各 91,952 B | 每 flush | ✅ 已确认 |
| 3 | 每 batch hot 快照拷贝 | `draw_command.c:458` | 61,240 + 3,512 B | 每个新 batch | ✅ 已确认 |
| 4 | `mglPendingDrawsReferenceVertexArray` 扫全部 128 batch + 解引用快照 | `:1306-1337` | ≤128 次 + 128 cache miss | **每个 VAO setter**（`vertex_arrays.c` 17 处 + `buffers.c:615,1464`） | ✅ T13-4 已改为 O(1) 哈希集 |
| 5 | `mglFlushPendingDrawsForActiveTextures` 扫 | `:1802-1852` | ≤128 单元 × 13 探针 | **每 draw**（`:3977`） | 未逐行复核 |
| 6 | `mglCaptureDynamicVertexBindings` 嵌套循环 | `:2500-2568`、`:2633-2663` | 30 + 64 + ≤435 内层 | 有 dyn-VB 的 draw | 未逐行复核 |
| 7 | `MGLDrawCommand` by-value 拷贝 | `:3995`、`:4221`、`:4265`、`:3894` | 408 B × ≤4 | 每 draw | 未逐行复核 |
| 8 | ICB env 查找 | `mgl_batch_path.c:56-64` | 5 × `getenv` + 5 × `strcasecmp` | **每 batch** | ✅ 已确认 `mgl_env_flag_enabled` 是无缓存 inline |
| 9 | 重复 key 比较（同参数调 2 次 `mglStateKeysEqual`） | `:4019` vs `:4025`；`:4065` vs `:4084` | 多余 96 B `memcmp` ×2 | 每 draw | ✅ 已确认 `mglStateKeysEqual` 是纯函数，可安全复用结果 |
| 10 | `mglComputeVertexArrayStateHash` 无条件扫 `MAX_ATTRIBS`(30) + 64 bindings | `:2081`、`:2102` | ≈400–1,100 ops | 每 draw（VAO 变时） | 未逐行复核 |

### 建议顺序（收益/风险比）

- [x] **T13-1（最值得做）** §5#1：三次遍历是同一次计算的重复。第一次（`:3664`）已算出
      "是否可 stream-merge + min/max 源索引"这个**纯函数结果**；`:3040` 的
      `mglCommandComputeElementVertexRange` 与 `:3875` 的 `mglAppendStreamMergedData` 在重算同一件事。
      **命令不可变**，所以在 `MGLDrawCommand` 里缓存一次即可，把 3×O(count) 降到 O(count)。
      oracle：用 `MGL_DEBUG_STREAM_MERGE` 的合并日志逐条 diff。
      **2026-09-21 核对**：`mglPrepareStreamMergeCandidate` 写入
      `source_first/last_index`；`mglAppendStreamMergedData` 只做 remap 写出（注释已说明
      不再重验）；stream-merge 成功时 `mglTrackPendingDrawBufferReads` 整段跳过。
- [x] **T13-2** §5#9：复用已算的 `keys_match`（纯函数、无副作用）。
      **2026-09-21**：record 路径与 demote 分支均复用局部 `keys_match` /
      `last_keys_match`。
- [x] **T13-3** §5#8：记忆化 `getenv`。
      **注意实现方式**：不能缓存"解析后的值"——测试会 `unsetenv`/`setenv` 后期待新值
      （`test_legacy_compat/test_batch_icb.c:22-29` 的 `clear_icb_env`）。
      可行做法：仅在记忆化结果为 `NULL`（未设置稳态）时才探 `getenv`，
      一旦见到值就复用；`unsetenv` 会把槽位打回 `NULL` 从而强制重新探测。
      **验收**：`make test-batch-icb` 必须绿（朴素缓存会打破它）。
      **2026-09-21**：`mgl_env_flag.c` 进程期缓存 + `mgl_env_flag_cache_invalidate()`；
      ICB 测试在 setenv/unsetenv 后调用 invalidate。`test-batch-icb` PASS。
- [x] **T13-4** §5#4：VAO → pending batch 反向索引，把 128 次解引用降到 O(1)。
      **2026-09-21**：`MGLCommandBuffer` 增加 `pending_vao_refs` / `pending_vao_names`
      开址哈希集；非 stream batch 在 `mglInitializeOrShareBatchStateSnapshot`
      成功后注册 `source_vao`、`snapshot->vao`、`key.vao_name`；
      `mglPendingDrawsReferenceVertexArray` 改为 O(1) 查询，overflow 降级为命中。
      Reset 仍靠整结构 `memset` 清零。
- [x] **T13-5** §5#3 / §5#2 / §5#7：见 T9 / T11。
      **2026-09-21**：跟踪项，随 T9/T11 关闭；非独立代码刀。
- [x] **T13-6** 为 §5#1、#3、#4、#6、#7、#9、#10 加 `MGL_PERF_ADD` 埋点——
      这些路径**当前没有任何计数器**，没有埋点就无法给可信数字。
      **2026-09-21**：先加哈希重算四计数器
      `g_mglHashRecompute{Texture,Vertex,Render,Uniform}SinceSwap`；
      快照字节已用 `mglSnapshotHotStateBytes()`（非虚报 sizeof）。

### 已正确、不要动

`MGL_OPT_DIRTY_HASH`（`draw_command.c:2290`，函数内 `static` 只 `getenv` 一次）、
`mglRuntimeMax*`（`:655-677`）、`mglBindNoFlushEnabled` / `mglSamplerSnapshotEnabled`、
`mglPerfSummaryEnabled` / `mglSignpostEnabled`、全部 trace 日志（`MGL_TRACE_LOG` 之外是 no-op）、
per-draw 路径上**没有 ObjC 消息发送**（只有 Metal 边界的 `mgl_renderer_port_shim.m`）。

---

## 6. 复核为**误报**——记录下来避免重复踩

这几条都是初看像缺陷、追下去发现设计正确的。**不要"修"它们。**

| 看似问题 | 为什么不是 bug |
|---|---|
| `glActiveTexture` 用 `mglMarkRendererDirtyBits`（`textures.c:1346`）"漏了哈希失效" | `active_texture` 是单一"当前单元"选择器，**不参与 key、也不参与 `mglComputeTextureHash`**（后者遍历 `active_texture_mask[4]` 位图，与它无关）。它不改变任何哈希输入 |
| `glScissor` 用 `mglMarkRendererDirtyBits`（`state.c:375`）"漏了哈希失效" | `var.scissor_box` **不在** `mglComputeRenderStateHash` 的输入里（逐行核对 `:2117-2174`）；它由 key 的显式 `scissor[4]` 字段承载并按整结构 `memcmp` 比较，覆盖充分 |
| `uniforms.c:3103-3104` 的 identity gate "漏了哈希失效" | 该门只在**槽位身份**（`buf`/`name`/`offset`/`size`）不变时跳过失效，而哈希的正是槽位身份、**不含缓冲区内容**。跳过安全，优化成立 |
| `glBindBufferBase` / `glBindBuffer` 漏标记 | 分别标在 `buffers.c:1641`（`DIRTY_BUFFER\|DIRTY_BUFFER_BASE_STATE`）与 `:1477` |
| `glVertexAttrib*f` 漏标记 | `mgl_gl_extensions.c:74` / `:100` 都调 `mglMarkStateDirtyBits(DIRTY_VAO)`，覆盖 `current_vertex_attrib`（vertex layout 哈希输入） |
| `readbuffer` 不在 key / 哈希里 | 回读专用，deferred encoder 不读；`rendering.c:1315-1320` 注释表明是有意设计（但见 T3：标记缺口仍需补或显式说明） |
| `buffers[]` 数组"不在 key/哈希" | 写点标在 `buffers.c:1477` |
| `*_table` / `program_pipeline` "不在 key/哈希" | 是**按名字查找的容器**，不是状态；名字已在 key 里 |
| `viewport_array_set` "不在 key/哈希" | 查询态标志，不影响绘制 |
| "哈希碰撞会导致错误合批"（`draw_command.c:4050-4061`） | **结论对、论证错**：合批决策是 96 字节 `memcmp`，哈希只是记忆化，碰撞本身不会导致错误合批。该注释的 `N²/2^65` 还把指数弄混了（128² = 2^14，不是 2^29）。但**在 `mglStateKeysEqualIgnoringUniformRanges` 里哈希是直接比较的，那里碰撞论证才真正相关** |

---

## 7. 建议的护栏（本阶段不改代码，仅登记）

### 7.1 编译期

- [x] **G1** `MGLStateKey` 加 `_Static_assert` 锁死"无隐式 padding"（配合 T5-1）。
      **2026-09-21**：已在 `draw_command.h`。
- [x] **G2** 手抄 dirty 掩码与权威定义用 `_Static_assert` 绑定（配合 T7-1）。
      **2026-09-21**：`mgl_dirty_bits.h`。
- [x] **G3** 给 `mglHotStateCopyBytes()` 这类由 offsetof 推导的常量函数，
      让注释与计数器都不再硬编码 82 KB / 51 KB 这类会漂移的数字
      （现注释仍写 82 KB / ~51 KB，实测 91,952 / 61,240）。
      **2026-09-21**：`mglHotStateCopyBytes()` = `mglSnapshotHotStateBytes()`；
      注释已改为实测值。
- [x] **G4** 新增 `scripts/state_dataflow_coverage.py` + `make test-state-dataflow`，实现三项检查：
      1. **CHECK 1**：`MGLStateKey` 每个字段都必须被 `mglComputeStateKey` 写入
         （否则恒为 0，无法区分状态）。本轮结果：15/15 全部写入 ✅
      2. **CHECK 2**：key 内部不得有隐式 padding 落在 `memcmp` 范围里。
         本轮结果：报出 offset 60 的 4 字节洞（独立复现了 T5-1）⚠️ → T5-1 后已无洞。
      3. **CHECK 3**：列出既不在 key、也不在任何哈希域的 `GLMState` 区域作为候选，
         交给人工复核。本轮 8 个候选**全部为误报**（见 §6），应写入 baseline。
      - 带 baseline 门禁：只有出现 baseline 之外的新发现才返回非 0，可直接挂 CI。
      - 本轮已用"注入一个未写入的 key 字段"做过反证测试，确认能抓到。
      **2026-09-21**：落地 `scripts/state_dataflow_coverage.py` + `make test-state-dataflow`
      （挂入 `test-all`）。CHECK 1/2 复用 `state_machine_invariants` 解析器；
      CHECK 3 对 top-level `GLMState` 未覆盖字段做 baseline 门禁（现 56 项，含 §6 误报）。
- 关键实现坑（记录以免重踩）：
  - 结构体提取要**锚定闭合花括号** `} MGLStateKey;` 再向前找 `{`，
    否则 `typedef struct` 正则会在 `GLMState`（内嵌 `GLMParams`/`GLMCaps`/`HashTable`）上抓错。
  - dirty 掩码**不要用正则解析 `#define` 体**——会抓到注释里的字面量 `DIRTY_BITS`。
    用 C 预处理器探针取值，或把已验证的值写进脚本并加自检。

### 7.3 运行期（成本低、能立刻抓到漏网路径）

- [x] **G5** 在 `mglComputeStateKey` 入口断言"若任一哈希域被重算，
      则对应 `dirty_bits` 必须包含该域"——直接抓 WRONG-DOMAIN 类缺口。
      **2026-09-21**：关闭为不适用。`*_dirty` 与 `dirty_bits` 是双轨
      （`mglClearStateDirtyBitsPreservingHashInvalidation` 可清后者留前者）；
      在 `mglComputeStateKey` 注明了原因。
- [x] **G6** 调试构建下每次 draw 同时用缓存哈希与全量重算哈希，不等即报错。
      这是"状态可见性"最直接的 oracle，能一次性覆盖所有漏失效路径。
      **这也是症状 4（blend/depth 污染）在静态上找不到成因时唯一的推进手段。**
      **2026-09-21**：`mglComputeStateKey` 在 `DEBUG`/`MGL_DEBUG` 下对四个哈希域做
      cache vs full 对照；默认开启，`MGL_HASH_CACHE_ORACLE=0` 可关。
- [x] **G7** replay 期 workspace 写入哨兵（配合 T0-3）。
      **2026-09-21**：先挂上 error 路径的 `g_mglReplayErrorRedirectsSinceSwap`。

---

## 8. 验收口径

| 用途 | 既有设施 |
|---|---|
| capture 侧字节成本 | `g_mglSnapshotBytesAllocatedSinceSwap`（**口径有误**：无条件上报 `sizeof(GLMState)+sizeof(VertexArray)` = 95,464，实际分配 64,752，虚报 1.53×；且 VAO 快照被释放时也计入 → 见 T13-6） |
| restore 侧 memcpy | `g_mglReplayMemcpyCountSinceSwap`（只计次数，建议改字节 → T11-2） |
| 同 key 跳过率 | `g_mglSameKeyRestoreSkipsSinceSwap` / `g_mglSameKeyOracleWouldSkipSinceSwap` |
| 合批拒绝原因分布 | `g_mglMergeRejectStateDiffersSinceSwap` 等 |
| dirty-key delta 窄化率 | `mgl_frame_activity.c:250-262` 已 dump |
| **缺失** | 哈希缓存重算率、§5#1/#3/#4/#6/#7/#9/#10 的成本——全都没有计数器 |

计数器默认关闭，需 `MGL_PERF_SUMMARY=1`（`mgl_frame_activity.h:265-268`）；
signpost 需 `MGL_SIGNPOST=1`。

回归面：`test-batch-restore` / `test-batch-issue` / `test-dirty-hash` / `test-batch-path` /
`test-batch-hazard` / `test-batch-icb` / `test-binding-stage` / `test-legacy-compat` /
`test-arch-correctness`，外加 hotspot（非通过集合逐条 diff 为空）与 MC 实机 FPS。

---

## 9. 证据边界

- 所有结构体尺寸 / offset / 字节数均为本机探针实测
  （`sizeof(GLMState)`=91,952、hot=61,240、key=96、`sizeof(HashTable)`=88、
  `MGLCommandBuffer`=193,088、`VertexArray`=3,512）。
  探针在 `/tmp`，不落在仓库内。
- **本轮没有测任何 ns/op 或倍率。** §5 的"预期收益"全部是操作计数量级推断，不是实测。
- **未坐实**项（不要当结论用）：
  - P0-1 的错误队列链路只做到**逐跳到行**，未构造运行时反例；
  - P0-2 / P0-3 依赖 `malloc` 地址复用，**未用运行时验证**过地址真的被复用；
  - P1-2 未构造"replay 期改哈希缓存"的场景；
  - P2-4 的 cold `buffer_base` 判定用字面量 grep；注意 `MGLRenderer+Buffer.m:260-262`
    存在 `buffer_base[gl_buffer_type]` 的**动态索引**路径，本审查**未证明**它不会命中冷类型；
  - P2-7 的 arena 浪费是读代码得出，未测真实 arena 水位。
- **症状 4（blend/depth 污染）静态上未找到成因**——`state.c` 约 30 个 setter 全部标记了
  覆盖 render_state 哈希的位。这条需要 G6 的运行时 A/B 才能继续推进。

## 10. 本文件与其他文档的关系

| 文档 | 内容 |
|---|---|
| `docs/STATE_MACHINE_REVIEW_2026-09-13.md` | 结构度量、缺陷证据、终态契约与 M0–M5 迁移顺序（**保留**） |
| **本文件** | 可执行 TODO：优先级、file:line、验收口径、误报清单 |
| `docs/STATE_DATAFLOW_AND_HOTPATHS.md` | 内容已并入本文件，**可删除** |

## 11. 2026-09-21 续审（对照仓库勾选）

核对口径：对 `MGL/src` / `MGL/include` 实读，未跑 Sodium / 未构造 P0-1 运行时反例。

| 项 | 现状 |
|---|---|
| **T0-1** | ✅ restore/from-key/`FCtx`/`MGLEncodeContext.state` + issue require；processGLState dual-proxy 断言 |
| **T0-2 / T0-3 / G7** | ✅ live error queue + `g_mglReplayErrorRedirectsSinceSwap` |
| **T1–T8 / T10 / T11-2 / T13 / G1–G7** | ✅（见各条证据） |
| **T12-1** | ✅ ranges + sampler + texture payload 堆化；index 内联 |
| **T12-2** | ✅ 已测；有意保留 256 sets |
| **T9-1** | ✅ identity 表 + `kHotRegions`；G4 CHECK 1b 双射门禁 |
| **T9-2** | **开放**：堵在打包布局（`GLMState *` 直读） |
| **T11-1** | ✅ 单 workspace：`pass->saved` |

下一步：T9-2 打包布局（改全部 `state_snapshot` 读点）。
