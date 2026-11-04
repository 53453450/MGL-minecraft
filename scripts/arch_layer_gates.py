#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
"""Architecture layer gates (docs/MGL-architecture-design.md §9).

Phase 0 mode: freeze bad dependencies. Existing violations are recorded in
scripts/arch_layer_gates_baseline.txt. The gate fails only when a NEW
violation appears, or when a baseline entry is stale (count went down —
run --update after paying debt).

Rules:
  Lyr-FE-Metal      FE TUs must not include metal_ref / backend_objects /
                    *_backend accessor headers (G1 / ARCH-005/007).
  Lyr-FE-Compiler   FE + draw/batch/pass must not include mgl_glsl_* /
                    mgl_air_* except abi/artifact allowlist (G4 / ARCH-008).
  Lyr-Pass-Batch    mgl_render_pass_*.c must not call/include mglBatch* /
                    mgl_batch_* (G3 / ARCH-004).
  Lyr-Batch-Pass    mgl_batch_*.c must not include mgl_render_pass_manager_ops.h
                    (only pass_public.h) (ARCH-004/016).
  Lyr-No-Private    MGL/src must not #include/#import *Private* (ARCH-009;
                    Phase 0: baseline allow; Phase 8: empty baseline).
  Lyr-DualProxyDoc  Informational: hand assignment to core->activeState outside
                    mgl_renderer_core_state.c / documented setters (ARCH-003).

Usage:
  python3 scripts/arch_layer_gates.py
  python3 scripts/arch_layer_gates.py --update
  python3 scripts/arch_layer_gates.py --report
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "scripts", "arch_layer_gates_baseline.txt")
SRC = os.path.join(ROOT, "MGL", "src")
INC = os.path.join(ROOT, "MGL", "include")

FE_FILES = (
    "buffers.c",
    "textures.c",
    "framebuffers.c",
    "program.c",
    "shaders.c",
    "state.c",
    "get.c",
)

METAL_HEADERS = re.compile(
    r'#\s*include\s*[<"]([^>"]*(?:'
    r"mgl_metal_ref|"
    r"mgl_backend_objects|"
    r"mgl_buffer_backend|"
    r"mgl_texture_backend|"
    r"mgl_sampler_backend|"
    r"mgl_program_backend"
    r")\.h)[>\"]"
)

COMPILER_HEADERS = re.compile(
    r'#\s*include\s*[<"]([^>"]*(?:mgl_glsl_|mgl_air_)[^>"]*\.h)[>\"]'
)

COMPILER_ALLOW = {
    "mgl_shader_abi.h",
    "mgl_compile_artifact.h",
    "mgl_air_tess_abi.h",
}

PASS_BATCH = re.compile(
    r'#\s*include\s*[<"]([^>"]*mgl_batch[^>"]*)[>"]'
    r"|\bmglBatch[A-Za-z0-9_]*\s*\("
)

BATCH_PASS_OPS = re.compile(
    r'#\s*include\s*[<"]([^>"]*mgl_render_pass_manager_ops\.h)[>\"]'
)

PRIVATE_INC = re.compile(
    r'#\s*(?:include|import)\s*[<"]([^>"]*Private[^>"]*)[>\"]'
)

ACTIVE_STATE_ASSIGN = re.compile(
    r"(?:core|_core)\s*->\s*activeState\s*="
)

COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def scan_file(path: str, regex: re.Pattern) -> list[tuple[int, str]]:
    text = strip_comments(open(path, encoding="utf-8", errors="ignore").read())
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if regex.search(line):
            out.append((i, line.strip()))
    return out


def collect() -> dict[tuple[str, str], list[tuple[int, str]]]:
    hits: dict[tuple[str, str], list[tuple[int, str]]] = {}

    def add(rule: str, path: str, found: list[tuple[int, str]]) -> None:
        if found:
            hits[(rule, rel(path))] = found

    # Lyr-FE-Metal
    for name in FE_FILES:
        path = os.path.join(SRC, name)
        if os.path.exists(path):
            add("Lyr-FE-Metal", path, scan_file(path, METAL_HEADERS))

    # Lyr-FE-Compiler
    for name in sorted(os.listdir(SRC)):
        if not name.endswith((".c", ".cpp")):
            continue
        if not (
            name in FE_FILES
            or name.startswith("draw_")
            or name.startswith("mgl_batch_")
            or name.startswith("mgl_render_pass")
        ):
            continue
        path = os.path.join(SRC, name)
        found = []
        text = strip_comments(open(path, encoding="utf-8", errors="ignore").read())
        for i, line in enumerate(text.splitlines(), 1):
            m = COMPILER_HEADERS.search(line)
            if not m:
                continue
            base = os.path.basename(m.group(1))
            if base in COMPILER_ALLOW:
                continue
            found.append((i, line.strip()))
        add("Lyr-FE-Compiler", path, found)

    # Lyr-Pass-Batch
    for name in sorted(os.listdir(SRC)):
        if not (name.startswith("mgl_render_pass") and name.endswith(".c")):
            continue
        path = os.path.join(SRC, name)
        add("Lyr-Pass-Batch", path, scan_file(path, PASS_BATCH))

    # Lyr-Batch-Pass
    for name in sorted(os.listdir(SRC)):
        if not (name.startswith("mgl_batch_") and name.endswith((".c", ".cpp"))):
            continue
        path = os.path.join(SRC, name)
        add("Lyr-Batch-Pass", path, scan_file(path, BATCH_PASS_OPS))

    # Lyr-No-Private (src only; include umbrella is Phase 8)
    for name in sorted(os.listdir(SRC)):
        if not name.endswith((".c", ".cpp", ".m", ".mm", ".h", ".hpp")):
            continue
        path = os.path.join(SRC, name)
        add("Lyr-No-Private", path, scan_file(path, PRIVATE_INC))

    # Lyr-DualProxy hand writes (baseline allow core helpers)
    for dirpath, _, files in os.walk(os.path.join(ROOT, "MGL")):
        for name in files:
            if not name.endswith((".c", ".cpp", ".h", ".hpp")):
                continue
            path = os.path.join(dirpath, name)
            found = scan_file(path, ACTIVE_STATE_ASSIGN)
            # Always allow the dual-proxy helpers themselves.
            if os.path.basename(path) in (
                "mgl_renderer_core_state.c",
                "mgl_renderer_core_state.h",
            ):
                continue
            add("Lyr-DualProxy", path, found)

    return hits


def read_baseline() -> dict[tuple[str, str], int]:
    base: dict[tuple[str, str], int] = {}
    if not os.path.exists(BASELINE):
        return base
    for raw in open(BASELINE, encoding="utf-8"):
        raw = raw.split("#", 1)[0].strip()
        if not raw:
            continue
        rule, path, count = raw.split()
        base[(rule, path)] = int(count)
    return base


def write_baseline(hits: dict[tuple[str, str], list[tuple[int, str]]]) -> None:
    lines = [
        "# Architecture layer gates baseline (Phase 0: ban NEW violations).",
        "# Format: rule path count. Counts may only go down; --update after paying debt.",
        "# See docs/MGL-architecture-design.md §9.",
        "",
    ]
    for (rule, path), found in sorted(hits.items()):
        lines.append(f"{rule} {path} {len(found)}")
    with open(BASELINE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    hits = collect()
    if args.update:
        write_baseline(hits)
        total = sum(len(v) for v in hits.values())
        print(
            f"wrote {rel(BASELINE)}: {len(hits)} entries, {total} violations"
        )
        return 0
    if args.report:
        for (rule, path), found in sorted(hits.items()):
            for ln, text in found:
                print(f"{rule} {path}:{ln}: {text}")
        return 0

    base = read_baseline()
    bad = 0
    for key in sorted(set(hits) | set(base)):
        have = len(hits.get(key, ()))
        want = base.get(key, 0)
        rule, path = key
        if have > want:
            bad += 1
            print(f"NEW {rule} {path}: {have} violations, baseline {want}")
            for ln, text in hits.get(key, ()):
                print(f"    {path}:{ln}: {text}")
        elif have < want:
            bad += 1
            print(
                f"STALE {rule} {path}: baseline {want}, now {have} "
                f"(run --update after paying debt)"
            )
    if bad:
        return 1
    total = sum(base.values())
    print(
        f"arch layer gates ok: {total} baselined violations in {len(base)} entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
