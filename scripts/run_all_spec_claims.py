#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
"""Run all scripts/spec_claims/*.json suites and write a markdown audit summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_DIR = ROOT / "scripts" / "spec_claims"
SCRATCH = ROOT / "scratch" / "spec_claims"
CHECKER = ROOT / "scripts" / "spec_claim_check.py"
DOC = ROOT / "docs" / f"SPEC_FULL_AUDIT_{date.today().isoformat()}.md"
DOC_HEADER_NOTE = (
    "规范权威源：`external/OpenGL-Registry/specs/gl/`"
    "（`glspec46.core.pdf` + `GLSLangSpec.4.60.html`）；"
    "证据语料由 `scripts/spec_registry_evidence.py` 从 Registry 抽取。"
)
PY = sys.executable

FOCUS = {
    "draw_command": "§10.4 Drawing",
    "vao": "§10.3 VAO / BindVertexBuffer",
    "texture_upload": "§8.5 / §8.19 TexImage/TexStorage",
    "glsl": "GLSL 4.60 + §7 / §11.2",
    "buffers": "§6 Buffers / Map / BindBufferRange",
    "shaders_programs": "§7 Shader/Program API",
    "uniforms": "§7.6 Uniforms / §8.26 Images",
    "framebuffers": "§9 / §17.4 / §18.3 FBO+Blit",
    "compute": "§19 Compute",
    "sync_fence": "§4.1 Sync / MemoryBarrier",
    "state_raster": "§13–17 Enable/Scissor/Stencil/LogicOp",
    "samplers": "§8.2 Sampler objects",
    "transform_feedback": "§13.3 XFB / DrawTransformFeedback",
    "queries": "§4.2 Queries",
    "pixel_ops": "§17–18 Clear/ReadBuffer",
    "copy_image": "§18.3.2 CopyImageSubData",
    "unimplemented_inventory": "Core stubs vs advertised 4.6",
}

VERDICT_KEYS = (
    "conforms",
    "violates",
    "unspecified",
    "no_evidence",
    "fabricated_quote",
)


def run_suite(path: Path) -> dict:
    out = SCRATCH / f"{path.stem}.json"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"\n======== {path.stem} ========", flush=True)
    subprocess.run(
        [PY, str(CHECKER), str(path), "--json-out", str(out)],
        cwd=ROOT,
        check=False,
    )
    if not out.exists():
        raise SystemExit(f"missing results for {path.stem}")
    return json.loads(out.read_text(encoding="utf-8"))


def claim_total(counts: dict) -> int:
    return sum(counts.get(k, 0) for k in VERDICT_KEYS)


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.stderr.write("TYPESAFE_API_KEY is not set.\n")
        return 2

    suites = sorted(CLAIMS_DIR.glob("*.json"))
    if not suites:
        sys.stderr.write(f"no suites in {CLAIMS_DIR}\n")
        return 2

    results = [run_suite(p) for p in suites]
    all_rows = []
    for suite in results:
        for row in suite["results"]:
            all_rows.append({**row, "suite": suite["suite"]})

    counts = Counter(r["verdict"] for r in all_rows)
    by_suite = {s["suite"]: s["counts"] for s in results}
    auto_violates = [
        r for r in all_rows if r["verdict"] == "violates" and r.get("auto")
    ]
    review_rows = [
        r
        for r in all_rows
        if (r["verdict"] in ("violates", "no_evidence") and not r.get("auto"))
        or (r["verdict"] == "no_evidence")
    ]
    # unique review list
    seen = set()
    review_unique = []
    for r in sorted(all_rows, key=lambda x: (x["suite"], x["id"])):
        if r["verdict"] in ("violates", "no_evidence") and not r.get("auto"):
            key = (r["suite"], r["id"])
            if key not in seen:
                seen.add(key)
                review_unique.append(r)
    unspecified = [r for r in all_rows if r["verdict"] == "unspecified"]
    needs_review = sum(1 for r in all_rows if not r.get("auto"))

    lines = [
        "# MGL OpenGL 4.6 Core 全子系统 SPEC 对照审计",
        "",
        f"**日期**：{date.today().isoformat()}  ",
        f"**规范**：{DOC_HEADER_NOTE}  ",
        "**方法**：`scripts/spec_claim_check.py` + `scripts/spec_claims/*.json`（TypeSafe Jev）  ",
        f"**模型**：{results[0].get('model', 'jev-latest')}  ",
        "**原始结果**：`scratch/spec_claims/*.json`  ",
        "",
        "> **覆盖定义**：对 MGL 实现的全部主要 OpenGL 子系统各建 claim suite，"
        "用 SPEC 错误/语义条款做高信号对照。这不是 4.6 全书逐句证明，"
        "也不是 GL46CTS 替代品；目标是系统级合规地图 + 可复跑违规清单。",
        "",
        "## 1. 子系统矩阵",
        "",
        "| 子系统 suite | SPEC 焦点 | claims | conforms | violates | unspecified | no_evidence | needs_review |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(by_suite):
        c = by_suite[name]
        lines.append(
            f"| `{name}` | {FOCUS.get(name, '')} | {claim_total(c)} | "
            f"{c.get('conforms', 0)} | {c.get('violates', 0)} | "
            f"{c.get('unspecified', 0)} | {c.get('no_evidence', 0)} | "
            f"{c.get('needs_review', 0)} |"
        )
    lines.append(
        f"| **合计** |  | **{len(all_rows)}** | **{counts.get('conforms', 0)}** | "
        f"**{counts.get('violates', 0)}** | **{counts.get('unspecified', 0)}** | "
        f"**{counts.get('no_evidence', 0)}** | **{needs_review}** |"
    )

    lines += ["", "## 2. Auto 违规（confidence ≥ 0.8，优先修）", ""]
    if not auto_violates:
        lines.append("（本轮无 auto 违规。）")
    else:
        lines += [
            "| suite | id | conf | claim |",
            "| --- | --- | ---: | --- |",
        ]
        for r in sorted(auto_violates, key=lambda x: (x["suite"], x["id"])):
            conf = r.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "-"
            claim = (r.get("claim") or "").replace("|", "\\|")
            lines.append(
                f"| `{r['suite']}` | `{r['id']}` | {conf_s} | {claim} |"
            )

    lines += ["", "## 3. Review 违规 / 摘录不足（需人工）", ""]
    if not review_unique:
        lines.append("（无）")
    else:
        lines += [
            "| suite | id | verdict | conf |",
            "| --- | --- | --- | ---: |",
        ]
        for r in review_unique:
            conf = r.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "-"
            lines.append(
                f"| `{r['suite']}` | `{r['id']}` | {r['verdict']} | {conf_s} |"
            )

    lines += ["", "## 4. Unspecified（SPEC 留空 / UB）", ""]
    if not unspecified:
        lines.append("（无）")
    else:
        for r in unspecified:
            lines.append(
                f"- `{r['suite']}/{r['id']}` — {r.get('claim', '')}"
            )

    lines += [
        "",
        "## 5. 复跑",
        "",
        "```bash",
        "export TYPESAFE_API_KEY='…'",
        ".venv-typesafe/bin/python scripts/run_all_spec_claims.py",
        "```",
        "",
        "## 6. 局限",
        "",
        "- Claim 抽检覆盖各子系统的高信号错误/语义条款，不是全书。",
        "- TypeSafe 只读提供的 SPEC 摘录；低 confidence / no_evidence 必须人工对照全文与代码。",
        "- `conforms` 只表示该 claim 不被摘录否定；不表示该子系统 CTS 全绿。",
        "- 与 `docs/SILENT_GAP_AUDIT_2026-09-17.md`、`docs/SPEC_CLAIM_AUDIT_2026-09-20.md` 互补。",
        "",
    ]

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {DOC}")
    print("totals:", dict(counts), "auto_violates=", len(auto_violates))
    if auto_violates:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
