# MGL OpenGL 4.6 Core SPEC 对照审查（TypeSafe claim check）

**日期**：2026-09-20  
**范围**：`draw_command` / `VAO` / `texture upload` / `GLSL`  
**规范**：OpenGL 4.6 Core Profile（`docs/glspec46.core.pdf`）；GLSL 语言规则另引 GLSL 4.60（`external/OpenGL-Registry/specs/gl/GLSLangSpec.4.60.pdf`）  
**方法**：`scripts/spec_claim_check.py` + `scripts/spec_claims/*.json`  
- 代码负责抽出「实现行为 claim」与 SPEC 摘录  
- TypeSafe Jev 判断摘录相对 claim：`supports` / `contradicts` / `unspecified` / `says_nothing`  
- `confidence ≥ 0.8` → auto；否则需人工复核  
**模型**：`jev-latest`（跑批当日）  
**原始结果**：`scratch/spec_claims/{draw_command,vao,texture_upload,glsl}.json`（本地 scratch，默认不入库）

> 这不是全量 CTS，也不是证明「无违规」。它是可复跑的、按 claim 切片的 SPEC 对照。  
> 与 `docs/SILENT_GAP_AUDIT_2026-09-17.md` 互补：后者偏 CTS 现象；本文偏 API/语义条款。

---

## 1. 总览

| 子系统 | claims | conforms | violates | unspecified | no_evidence | needs_review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| draw_command | 14 | 11 | 2 | 1 | 0 | 2 |
| vao | 10 | 8 | 2 | 0 | 0 | 0 |
| texture_upload | 9 | 7 | 1 | 1 | 0 | 2 |
| glsl | 7 | 3 | 2 | 0 | 2 | 4 |
| **合计** | **40** | **29** | **7** | **2** | **2** | **8** |

**Auto 接受的违规（优先修）**：5 条（见 §2）。  
**人工确认仍视为违规（证据足、TypeSafe conf 偏低或摘录过窄）**：GLSL 内建常量缺失、subroutine 缺口（见 §3，并与 SILENT_GAP F11/F15 一致）。

---

## 2. Auto 违规（confidence ≥ 0.8）

### 2.1 无 VAO 绑定时自动绑定默认 VAO（draw + VAO 交叉）

| 字段 | 内容 |
| --- | --- |
| ids | `draw_command/no_vao_auto_bind_default`，`vao/draw_auto_binds_default_vao` |
| 代码 | `MGL/src/draw_buffers.c` → `validate_vao`：`mglGetOrCreateDefaultVAO` 后继续画 |
| SPEC | §10.3.1 / §10.4：无 VAO 绑定时，修改/绘制/查询顶点数组状态须 `INVALID_OPERATION` |
| 裁决 | **violates**（conf ≈ 1.0） |

Core profile 不允许「默认 VAO 代画」。应报错，而不是隐式创建/绑定。

### 2.2 `MultiDraw*Indirect` 的 `drawcount == 0`

| 字段 | 内容 |
| --- | --- |
| id | `draw_command/multidraw_indirect_drawcount_must_be_positive` |
| 代码 | `mglMultiDrawArraysIndirect`：`drawcount == 0` 时直接 `return`（无错误） |
| SPEC | §10.4 MultiDrawArraysIndirect Errors：`drawcount` **must be positive** → 否则 `INVALID_VALUE` |
| 裁决 | **violates**（conf 0.96） |

注意：非 Indirect 的 `MultiDrawArrays` 允许 `drawcount == 0`（空循环）；Indirect 族更严。

### 2.3 `BindVertexBuffers` 不接受「已 Gen 未创建」的 buffer 名

| 字段 | 内容 |
| --- | --- |
| id | `vao/bind_vertex_buffers_rejects_uncreated_gen_name` |
| 代码 | `mglBindVertexBuffers`：`!findBuffer` → `INVALID_OPERATION`（无 create-on-bind） |
| 对照 | 单条 `BindVertexBuffer` / `bindVertexBuffer` **已**按 §10.3.2 对 Gen 名做 create-on-bind |
| SPEC | `BindVertexBuffers` 等价于逐条 `BindVertexBuffer`；应对 Gen 名先创建再绑定 |
| 裁决 | **violates**（conf 0.98） |

### 2.4 `TexImage2D(TEXTURE_RECTANGLE, level≠0)` 错误码

| 字段 | 内容 |
| --- | --- |
| id | `texture_upload/teximage2d_rectangle_nonzero_level_error_code` |
| 代码 | `ERROR_CHECK_RETURN(level==0, GL_INVALID_OPERATION)` |
| SPEC | §8.5：须生成 **`INVALID_VALUE`**（不是 `INVALID_OPERATION`） |
| 裁决 | **violates**（conf 0.98） |

---

## 3. 人工确认的 GLSL 缺口（TypeSafe 低 conf / 摘录过窄）

下列条目在代码与 SILENT_GAP 审计中已有硬证据；本轮 TypeSafe 因「摘录未直接写实现义务」给出 `review` / `no_evidence`，**维护上仍按违规跟踪**。

| id | 现象 | SPEC | 说明 |
| --- | --- | --- | --- |
| `omit_core_builtin_gl_MaxCombinedTextureImageUnits` | core 常量表缺名；legacy 表有但值=8 且仅 ≤150 | GLSL §7.3：`= 96`，**all shaders** | 与 SILENT_GAP **F11** 同族 |
| `omit_core_builtin_gl_MaxAtomicCounterBindings` | parser builtins 无 `gl_MaxAtomicCounter*` | GLSL §7.3 最小值 1 | 同上 |
| `subroutine_keyword_not_parsed` | lexer/parser 无 `subroutine` | GLSL §6.1.2 | 与 **F15** 一致；广告 `4.6.0` 却整条链缺失 |
| `max_subroutines_limit_zero` | `max_subroutines = 0` | Core 4.6 含 subroutine | 与 API stub / 前端缺失一致 |

---

## 4. Unspecified / 需注意但非硬违规

| id | 行为 | SPEC | 结论 |
| --- | --- | --- | --- |
| `missing_ebo_undefined_skip` | 索引绘制无 EBO 时静默跳过 | §10.3.10：EBO=0 时结果 **undefined** | **unspecified**（允许，但 CTS/调试上建议至少诊断） |
| `unsupported_internalformat_for_metal` | 无 Metal 映射的 internalformat 拒绝上传 | §8.5 合法 format 集合 | 多为实现限制；错误码是否总与表一致需逐格式核对（review） |

---

## 5. 各子系统已对齐的要点（conforms，摘录）

**draw_command**：负 `count` → `INVALID_VALUE`；非法 `mode`/`type` → `INVALID_ENUM`；`DrawRangeElements` 的 `end < start`；primitive restart 比较在 `basevertex` 之前；`PRIMITIVE_RESTART_FIXED_INDEX` 优先；Indirect `stride` 须为 4 的倍数。

**VAO**：`n < 0`；未知 VAO 名绑定报错；删除当前 VAO 回落到 0；无 VAO 时 `VertexAttribBinding` / `BindVertexBuffer` 报 `INVALID_OPERATION`；单条 `BindVertexBuffer` 对 Gen 名 create-on-bind；负 `offset`/`stride`；`VertexAttribPointer` 负 stride。

**texture upload**：`border != 0` → `INVALID_VALUE`；负尺寸；immutable 上 `TexImage*` → `INVALID_OPERATION`；MS storage `samples < 1`；stencil format 配对；PBO 越界 → `INVALID_OPERATION`（claim 级）。

**GLSL / 着色器 API**：`CompileShader` 对 program 名 → `INVALID_OPERATION`；info log 被覆盖；TCS 无 TES 时 `GL_PATCHES` 绘制拒绝。

---

## 6. 建议修复顺序

1. **`validate_vao` 默认 VAO**：core 路径改为无绑定即 `INVALID_OPERATION`（与 §10.3.1 一致）；若保留兼容，须显式 gated（非默认）。  
2. **`MultiDraw*Indirect` `drawcount == 0`** → `GL_INVALID_VALUE`。  
3. **`BindVertexBuffers` / DSA 批量路径**：复用 `bindVertexBuffer` 的 Gen 名 create-on-bind。  
4. **`TEXTURE_RECTANGLE` 非 0 level** → 改报 `GL_INVALID_VALUE`。  
5. **GLSL F11/F15**：补齐 §7.3 core 常量表；实现 `subroutine` 语法 + 非零 limits + API（工作量大，但 CTS 杠杆最高）。

---

## 7. 复跑方式

```bash
# Python ≥ 3.10；本地 venv（已 gitignore）
# /opt/homebrew/bin/python3.14 -m venv .venv-typesafe
# .venv-typesafe/bin/pip install 'typesafe-sdk>=0.5.7'

export TYPESAFE_API_KEY='…'   # 勿写入仓库
for suite in draw_command vao texture_upload glsl; do
  .venv-typesafe/bin/python scripts/spec_claim_check.py \
    "scripts/spec_claims/${suite}.json" \
    --json-out "scratch/spec_claims/${suite}.json"
done
```

新增 claim：在对应 JSON 增加条目（`claim` / `code_excerpt` / `spec_quote`），再跑单套即可。

---

## 8. 局限

- Claim 覆盖的是抽检条款，不是 4.6 全书或 GL46CTS mustpass。  
- TypeSafe 只读你提供的 SPEC 摘录；摘录过窄会得到 `no_evidence` / 低 confidence——须人工用全文与代码复核（本文 §3）。  
- `supports` ≠「实现已完美」；仅表示「所声称行为不被该摘录否定 / 被要求或推荐」。  
- API key 仅允许环境变量注入；`.venv-typesafe/` 与 `scratch/` 不入库。


> 全子系统扩展见 [`docs/SPEC_FULL_AUDIT_2026-09-20.md`](SPEC_FULL_AUDIT_2026-09-20.md)。
