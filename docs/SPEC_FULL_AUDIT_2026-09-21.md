# MGL OpenGL 4.6 Core 全子系统 SPEC 对照审计

**日期**：2026-09-21  
**规范**：规范权威源：`external/OpenGL-Registry/specs/gl/`（`glspec46.core.pdf` + `GLSLangSpec.4.60.html`）；证据语料由 `scripts/spec_registry_evidence.py` 从 Registry 抽取。  
**方法**：`scripts/spec_claim_check.py` + `scripts/spec_claims/*.json`（TypeSafe Jev）  
**模型**：jev-latest  
**原始结果**：`scratch/spec_claims/*.json`  

> **覆盖定义**：对 MGL 实现的全部主要 OpenGL 子系统各建 claim suite，用 SPEC 错误/语义条款做高信号对照。这不是 4.6 全书逐句证明，也不是 GL46CTS 替代品；目标是系统级合规地图 + 可复跑违规清单。
>
> 本轮相对 2026-09-20：证据源改为 Registry；新增 `copy_image`（7）；合计 86 claims。上轮已人工确认的 GLSL/stub 项若本轮落在 §3 低 confidence，仍以上轮人工裁定为准。

## 1. 子系统矩阵

| 子系统 suite | SPEC 焦点 | claims | conforms | violates | unspecified | no_evidence | needs_review |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `buffers` | §6 Buffers / Map / BindBufferRange | 5 | 4 | 0 | 0 | 1 | 1 |
| `compute` | §19 Compute | 4 | 1 | 0 | 1 | 2 | 1 |
| `copy_image` | §18.3.2 CopyImageSubData | 7 | 6 | 0 | 0 | 1 | 1 |
| `draw_command` | §10.4 Drawing | 14 | 11 | 2 | 1 | 0 | 2 |
| `framebuffers` | §9 / §17.4 / §18.3 FBO+Blit | 4 | 4 | 0 | 0 | 0 | 0 |
| `glsl` | GLSL 4.60 + §7 / §11.2 | 7 | 3 | 2 | 0 | 2 | 5 |
| `pixel_ops` | §17–18 Clear/ReadBuffer | 2 | 0 | 0 | 0 | 2 | 2 |
| `queries` | §4.2 Queries | 2 | 2 | 0 | 0 | 0 | 1 |
| `samplers` | §8.2 Sampler objects | 3 | 2 | 0 | 0 | 1 | 1 |
| `shaders_programs` | §7 Shader/Program API | 4 | 4 | 0 | 0 | 0 | 0 |
| `state_raster` | §13–17 Enable/Scissor/Stencil/LogicOp | 5 | 2 | 0 | 0 | 3 | 1 |
| `sync_fence` | §4.1 Sync / MemoryBarrier | 3 | 3 | 0 | 0 | 0 | 0 |
| `texture_upload` | §8.5 / §8.19 TexImage/TexStorage | 9 | 5 | 1 | 1 | 2 | 3 |
| `transform_feedback` | §13.3 XFB / DrawTransformFeedback | 2 | 0 | 0 | 0 | 2 | 2 |
| `uniforms` | §7.6 Uniforms / §8.26 Images | 3 | 2 | 1 | 0 | 0 | 2 |
| `unimplemented_inventory` | Core stubs vs advertised 4.6 | 2 | 0 | 0 | 0 | 2 | 1 |
| `vao` | §10.3 VAO / BindVertexBuffer | 10 | 8 | 2 | 0 | 0 | 1 |
| **合计** |  | **86** | **57** | **8** | **3** | **18** | **24** |

## 2. Auto 违规（confidence ≥ 0.8，优先修）

| suite | id | conf | claim | 2026-09-21 修复 |
| --- | --- | ---: | --- | --- |
| `draw_command` | `multidraw_indirect_drawcount_must_be_positive` | 0.81 | MultiDraw*Indirect drawcount==0 → INVALID_VALUE | ✅ `draw_buffers.c` |
| `draw_command` | `no_vao_auto_bind_default` | 0.98 | 无 VAO 绑定时 draw 不自动建 VAO 0 | ✅ `validate_vao` |
| `texture_upload` | `teximage2d_rectangle_nonzero_level_error_code` | 0.99 | RECTANGLE level≠0 → INVALID_VALUE | ✅ `textures.c` |
| `vao` | `bind_vertex_buffers_rejects_uncreated_gen_name` | 0.98 | BindVertexBuffers 对 gen 名应创建对象 | ✅ 交 `bindVertexBuffer` |
| `vao` | `draw_auto_binds_default_vao` | 0.99 | 同 no_vao_auto_bind | ✅ 同上 |

> 修复后未重跑 Jev（需 `TYPESAFE_API_KEY`）；以代码对照 SPEC 摘录为准。

## 3. Review 违规 / 摘录不足（需人工）

| suite | id | verdict | conf |
| --- | --- | --- | ---: |
| `compute` | `dispatch_indirect_no_buffer` | no_evidence | 0.51 |
| `copy_image` | `copy_image_zero_name_invalid_value` | no_evidence | 0.70 |
| `glsl` | `max_subroutines_limit_zero` | no_evidence | 0.22 |
| `glsl` | `omit_core_builtin_gl_MaxAtomicCounterBindings` | violates | 0.26 |
| `glsl` | `omit_core_builtin_gl_MaxCombinedTextureImageUnits` | violates | 0.29 |
| `glsl` | `subroutine_keyword_not_parsed` | no_evidence | 0.67 |
| `pixel_ops` | `clear_buffer_ops_present` | no_evidence | 0.46 |
| `pixel_ops` | `read_buffer_back_on_fbo` | no_evidence | 0.51 |
| `state_raster` | `stencil_op_invalid_enum` | no_evidence | 0.70 |
| `texture_upload` | `texstorage_ms_samples_lt_one` | no_evidence | 0.59 |
| `transform_feedback` | `begin_transform_feedback_exists` | no_evidence | 0.52 |
| `transform_feedback` | `draw_transform_feedback_unimplemented_errors` | no_evidence | 0.45 |
| `uniforms` | `bind_image_unit_out_of_range` | violates | 0.20 |
| `unimplemented_inventory` | `unimplemented_returns_invalid_operation` | no_evidence | 0.79 |

## 4. Unspecified（SPEC 留空 / UB）

- `compute/dispatch_indirect_over_max_reports_error` — When an indirect dispatch reads group counts exceeding MAX_COMPUTE_WORK_GROUP_COUNT, MGL generates GL_INVALID_VALUE.
- `draw_command/missing_ebo_undefined_skip` — When an indexed draw is issued while the VAO has no ELEMENT_ARRAY_BUFFER, MGL silently skips the draw without generating a GL error.
- `texture_upload/unsupported_internalformat_for_metal` — When an internalformat has no Metal equivalent, MGL rejects the texture image/storage call with GL_INVALID_OPERATION via checkInternalFormatForMetal.

## 5. 复跑

```bash
export TYPESAFE_API_KEY='…'
.venv-typesafe/bin/python scripts/run_all_spec_claims.py
```

## 6. 局限

- Claim 抽检覆盖各子系统的高信号错误/语义条款，不是全书。
- TypeSafe 只读提供的 SPEC 摘录；低 confidence / no_evidence 必须人工对照全文与代码。
- `conforms` 只表示该 claim 不被摘录否定；不表示该子系统 CTS 全绿。
- 与 `docs/SILENT_GAP_AUDIT_2026-09-17.md`、`docs/SPEC_CLAIM_AUDIT_2026-09-20.md` 互补。
