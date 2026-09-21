# MGL OpenGL 4.6 Core 全子系统 SPEC 对照审计

**日期**：2026-09-20  
**规范**：OpenGL 4.6 Core Profile（`docs/glspec46.core.pdf`）；GLSL 条款另引 GLSL 4.60  
**方法**：`scripts/spec_claim_check.py` + `scripts/spec_claims/*.json`（TypeSafe Jev）  
**模型**：jev-latest  
**原始结果**：`scratch/spec_claims/*.json`  

> **覆盖定义**：对 MGL 实现的全部主要 OpenGL 子系统各建 claim suite，用 SPEC 错误/语义条款做高信号对照。这不是 4.6 全书逐句证明，也不是 GL46CTS 替代品；目标是系统级合规地图 + 可复跑违规清单。

## 1. 子系统矩阵

| 子系统 suite | SPEC 焦点 | claims | conforms | violates | unspecified | no_evidence | needs_review |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `buffers` | §6 Buffers / Map / BindBufferRange | 5 | 5 | 0 | 0 | 0 | 0 |
| `compute` | §19 Compute | 4 | 3 | 0 | 1 | 0 | 0 |
| `draw_command` | §10.4 Drawing | 14 | 11 | 2 | 1 | 0 | 2 |
| `framebuffers` | §9 / §17.4 / §18.3 FBO+Blit | 4 | 4 | 0 | 0 | 0 | 0 |
| `glsl` | GLSL 4.60 + §7 / §11.2 | 7 | 3 | 2 | 0 | 2 | 4 |
| `pixel_ops` | §17–18 Clear/ReadBuffer | 2 | 1 | 0 | 0 | 1 | 1 |
| `queries` | §4.2 Queries | 2 | 2 | 0 | 0 | 0 | 0 |
| `samplers` | §8.2 Sampler objects | 3 | 3 | 0 | 0 | 0 | 0 |
| `shaders_programs` | §7 Shader/Program API | 4 | 4 | 0 | 0 | 0 | 1 |
| `state_raster` | §13–17 Enable/Scissor/Stencil/LogicOp | 5 | 5 | 0 | 0 | 0 | 0 |
| `sync_fence` | §4.1 Sync / MemoryBarrier | 3 | 3 | 0 | 0 | 0 | 0 |
| `texture_upload` | §8.5 / §8.19 TexImage/TexStorage | 9 | 8 | 1 | 0 | 0 | 2 |
| `transform_feedback` | §13.3 XFB / DrawTransformFeedback | 2 | 0 | 0 | 0 | 2 | 2 |
| `uniforms` | §7.6 Uniforms / §8.26 Images | 3 | 3 | 0 | 0 | 0 | 0 |
| `unimplemented_inventory` | Core stubs vs advertised 4.6 | 2 | 0 | 1 | 1 | 0 | 2 |
| `vao` | §10.3 VAO / BindVertexBuffer | 10 | 8 | 2 | 0 | 0 | 0 |
| **合计** |  | **79** | **63** | **8** | **3** | **5** | **14** |

## 2. Auto 违规（confidence ≥ 0.8，优先修）

| suite | id | conf | claim |
| --- | --- | ---: | --- |
| `draw_command` | `multidraw_indirect_drawcount_must_be_positive` | 0.96 | MultiDrawArraysIndirect with drawcount == 0 generates GL_INVALID_VALUE because drawcount must be positive. |
| `draw_command` | `no_vao_auto_bind_default` | 1.00 | If no vertex array object is currently bound when a drawing command runs, MGL auto-binds/creates a default VAO and continues, rather than generating GL_INVALID_OPERATION. |
| `texture_upload` | `teximage2d_rectangle_nonzero_level_error_code` | 0.97 | TexImage2D with target TEXTURE_RECTANGLE and level != 0 generates GL_INVALID_OPERATION in MGL. |
| `vao` | `bind_vertex_buffers_rejects_uncreated_gen_name` | 0.97 | BindVertexBuffers rejects a non-zero buffer name that is not yet an existing buffer object with GL_INVALID_OPERATION, without creating the object first. |
| `vao` | `draw_auto_binds_default_vao` | 0.99 | When drawing with no VAO bound, MGL auto-binds a default VAO instead of generating GL_INVALID_OPERATION as required for core profile draw/modify/query of vertex array state. |

### 2.1 建议修复顺序

1. **`validate_vao` 默认 VAO**（draw/vao 交叉）→ 无绑定报 `INVALID_OPERATION`  
2. **`MultiDraw*Indirect` `drawcount==0`** → `INVALID_VALUE`  
3. **`BindVertexBuffers` Gen 名 create-on-bind**（对齐单条 `BindVertexBuffer`）  
4. **`TEXTURE_RECTANGLE` level≠0** → 改报 `INVALID_VALUE`  

---

## 3. Review 违规 / 摘录不足（需人工）

| suite | id | verdict | conf |
| --- | --- | --- | ---: |
| `glsl` | `max_subroutines_limit_zero` | no_evidence | 0.76 |
| `glsl` | `omit_core_builtin_gl_MaxAtomicCounterBindings` | violates | 0.43 |
| `glsl` | `omit_core_builtin_gl_MaxCombinedTextureImageUnits` | violates | 0.43 |
| `glsl` | `subroutine_keyword_not_parsed` | no_evidence | 0.57 |
| `pixel_ops` | `clear_buffer_ops_present` | no_evidence | 0.25 |
| `transform_feedback` | `begin_transform_feedback_exists` | no_evidence | 0.57 |
| `transform_feedback` | `draw_transform_feedback_unimplemented_errors` | no_evidence | 0.70 |
| `unimplemented_inventory` | `advertises_46_with_stubs` | violates | 0.40 |

### 3.1 人工确认结论（2026-09-20）

对照代码 + GL 4.6 / GLSL 4.60 原文后，对 §3 及同批低 confidence 项裁定如下。

| id | TypeSafe | 人工裁定 | 证据摘要 |
| --- | --- | --- | --- |
| `glsl/omit_core_builtin_gl_MaxCombinedTextureImageUnits` | violates (0.43) | **确认违规** | `mgl_glsl_parser.c` `builtins[]` **无此名**；仅 `mgl_legacy_compat.c` 有且值为 **8**（§7.3 最小值 **96**），且只服务 GLSL ≤150。4.x 着色器引用会编不过。 |
| `glsl/omit_core_builtin_gl_MaxAtomicCounterBindings` | violates (0.43) | **确认违规** | parser / legacy 表均无 `gl_MaxAtomicCounter*`；GLSL §7.3 要求 all shaders 可见。 |
| `glsl/subroutine_keyword_not_parsed` | no_evidence (0.57) | **确认违规** | `mgl_glsl_*` 中 **零处** `subroutine`；`mglGetSubroutine*` / `mglUniformSubroutinesuiv` 走 `mgl_unimplemented`。 |
| `glsl/max_subroutines_limit_zero` | no_evidence (0.76) | **确认违规** | `glm_params.c`：`max_subroutines = 0`、`max_subroutine_uniform_locations = 0`；同时 `get.c` 广告 `GL_VERSION`=`4.6.0`。 |
| `pixel_ops/clear_buffer_ops_present` | no_evidence (0.25) | **确认合规** | `ClearBufferiv/uiv/fv/fi` 已挂 dispatch；`mgl_clear_buffer_ops.c` 存在。TypeSafe 缺证据是摘录过弱，不是缺实现。 |
| `transform_feedback/begin_transform_feedback_exists` | no_evidence (0.57) | **确认合规** | `mglBeginTransformFeedback` 有真实状态机（active/paused、重复 Begin → `INVALID_OPERATION`），非 stub。 |
| `transform_feedback/draw_transform_feedback_unimplemented_errors` | no_evidence (0.70) | **确认违规（能力缺口）** | 四个 `DrawTransformFeedback*` **无条件** `ERROR_RETURN(GL_INVALID_OPERATION)`（注释写明 Metal 未接线）。§13.3.3 定义成功语义≈`DrawArrays*`；fail-closed 优于静默成功，但相对 4.6 广告仍属缺口。 |
| `unimplemented_inventory/advertises_46_with_stubs` | violates (0.40) | **确认违规** | `GL_VERSION`=`4.6.0` + `mgl_gl_extensions.c` 内约 **24** 处 `mgl_unimplemented` + XFB draw / subroutine stub。 |
| `unimplemented_inventory/unimplemented_returns_invalid_operation` | unspecified (0.26) | **确认可接受** | stub **确实**报 `INVALID_OPERATION`（不静默成功）。问题在版本广告（上条），不在 stub 错误策略本身。 |

#### 同批低 confidence「conforms」复核

| id | TypeSafe | 人工裁定 | 说明 |
| --- | --- | --- | --- |
| `shaders_programs/use_program_unlinked_invalid_operation` | conforms (0.75) | **确认合规** | §7.3：未成功 link 的 program → `INVALID_OPERATION` 且不改当前状态；`mglUseProgram` 默认如此（`MGL_COMPAT_PROGRAM_ERRORS` 才跳过报错）。 |
| `draw_command/count_zero_is_noop` | conforms (0.55) | **确认合规** | 负 count 才 `INVALID_VALUE`；count==0 不画不报错（与 MultiDrawArrays 伪码 `count[i] > 0` 一致）。 |
| `draw_command/instancecount_negative_invalid_value` | conforms (0.70) | **确认合规** | SPEC 明文：`instancecount` 为负 → `INVALID_VALUE`；`mglDrawDispatch` 对 instanced 类型如此检查。 |
| `texture_upload/teximage_on_immutable_storage` | conforms (0.25) | **确认合规** | `mglTexImage2D`：`ERROR_CHECK_RETURN(!tex->immutable_storage, GL_INVALID_OPERATION)`；注释指向不可变存储语义。 |
| `texture_upload/unsupported_internalformat_for_metal` | conforms (0.28) | **有条件接受** | 无 Metal 映射时拒绝上传合理；是否对每个 format 使用 SPEC 规定的 `INVALID_VALUE` vs `INVALID_OPERATION` 需按表逐项核对，**本条不升格为硬违规**。 |

#### 本轮人工结论汇总

- **确认违规（须跟踪）**：GLSL 内建常量缺口 ×2、`subroutine` 整链、`MAX_SUBROUTINES=0`、广告 4.6 与 stub、`DrawTransformFeedback*` 未实现。  
- **确认合规 / 可接受**：ClearBuffer 存在、BeginTransformFeedback 存在、UseProgram 未 link 报错、count==0 / 负 instancecount、immutable 上 TexImage 拒绝、stub 报 `INVALID_OPERATION`。  
- **仍保持 auto 违规（§2，无需再辩）**：默认 VAO、Indirect `drawcount==0`、`BindVertexBuffers` Gen 名、RECTANGLE 错误码。

---

## 4. Unspecified（SPEC 留空 / UB）

- `compute/dispatch_indirect_over_max_reports_error` — When an indirect dispatch reads group counts exceeding MAX_COMPUTE_WORK_GROUP_COUNT, MGL generates GL_INVALID_VALUE.（Indirect 路径 SPEC 写 undefined；报错更严，可接受）
- `draw_command/missing_ebo_undefined_skip` — 无 EBO 时静默跳过（§10.3.10 undefined）
- `unimplemented_inventory/unimplemented_returns_invalid_operation` — stub 报 `INVALID_OPERATION`（相对静默成功更安全，但与「版本广告」冲突见 §3.1）

---

## 5. 按子系统的合规印象（本轮 claim 级）

| 区域 | 印象 |
| --- | --- |
| buffers / uniforms / framebuffers blit / samplers / state_raster / sync_fence / compute(direct) | claim 级大体对齐 |
| draw + VAO | 有明确硬违规（默认 VAO、Indirect drawcount、BindVertexBuffers） |
| texture upload | 大体对齐；RECTANGLE 错误码错 |
| GLSL / XFB / stub 清单 | 能力缺口大；TypeSafe 摘录偏窄，需人工与 CTS 交叉 |

---

## 6. 复跑

```bash
export TYPESAFE_API_KEY='…'
.venv-typesafe/bin/python scripts/run_all_spec_claims.py
```

单套：

```bash
.venv-typesafe/bin/python scripts/spec_claim_check.py scripts/spec_claims/<suite>.json \
  --json-out scratch/spec_claims/<suite>.json
```

---

## 7. 局限

- Claim 抽检覆盖各子系统的高信号错误/语义条款，**不是** 4.6 全书，**不是** GL46CTS 全量。
- TypeSafe 只读提供的 SPEC 摘录；低 confidence / no_evidence 必须人工对照全文与代码。
- `conforms` 只表示该 claim 不被摘录否定；不表示该子系统 CTS 全绿。
- 与 `docs/SILENT_GAP_AUDIT_2026-09-17.md`、`docs/SPEC_CLAIM_AUDIT_2026-09-20.md` 互补。
