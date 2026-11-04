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
                    Phase 8: also forbid leftover MGLRenderer*_Private.h /
                    mgl_renderer_ivars.h in the tree).
  Lyr-DualProxy     Ban assignment to core/_core->activeState anywhere
                    (ARCH-003 / Phase 2: single ctx->active_state).
  Lyr-Ports-FanIn   Count of MGL/src TUs that #include mgl_renderer_ports.h
                    must stay ≤ baseline (Phase 5: target ≤15).
  Lyr-Ext           MGL/extensions/** must not include mgl_renderer_ports.h /
                    backend_objects / mgl_glsl_parser / *Private* (G6).
  Lyr-Ctx-Fwd       Public/plan headers (api.h, buffer_plan, pass_public,
                    extension_hooks) must not include full glm_context.h (G5).
  Lyr-Ctx-FanIn     Count of MGL/src TUs that #include glm_context.h must
                    stay ≤ CTX_FANIN_LIMIT (G5 / Phase 1: ≤110).

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

PORTS_INCLUDE = re.compile(
    r'#\s*include\s*[<"]mgl_renderer_ports\.h[>"]'
)

PORTS_FANIN_LIMIT = 15
CTX_FANIN_LIMIT = 110

CTX_FULL_INCLUDE = re.compile(
    r'#\s*include\s*[<"]glm_context\.h[>"]'
)

# Public / plan headers that must stay on glm_context_fwd (G5 / Lyr-Ctx-Fwd).
CTX_FWD_HEADERS = (
    "mgl/api.h",
    "mgl_buffer_plan.h",
    "mgl_render_pass_public.h",
    "mgl_extension_hooks.h",
    "mgl_batch_public.h",
    "glm_state.h",
)

EXT_FORBIDDEN = re.compile(
    r'#\s*include\s*[<"]([^>"]*(?:'
    r"mgl_renderer_ports\.h|"
    r"mgl_backend_objects|"
    r"mgl_glsl_parser|"
    r"Private"
    r")[^>\"]*)[>\"]"
)

PRIVATE_LEFTOVER_NAMES = (
    "MGLRenderer_Private.h",
    "mgl_renderer_ivars.h",
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


def iter_src_files(*suffixes: str):
    for dirpath, _, files in os.walk(SRC):
        for name in sorted(files):
            if name.endswith(suffixes):
                yield os.path.join(dirpath, name)


def collect() -> dict[tuple[str, str], list[tuple[int, str]]]:
    hits: dict[tuple[str, str], list[tuple[int, str]]] = {}

    def add(rule: str, path: str, found: list[tuple[int, str]]) -> None:
        if found:
            hits[(rule, rel(path))] = found

    # Lyr-FE-Metal (FE may live under L4_context/ after Phase 6)
    for name in FE_FILES:
        matches = [
            p for p in iter_src_files(".c") if os.path.basename(p) == name
        ]
        for path in matches:
            add("Lyr-FE-Metal", path, scan_file(path, METAL_HEADERS))

    # Lyr-FE-Compiler
    for path in iter_src_files(".c", ".cpp"):
        name = os.path.basename(path)
        if not (
            name in FE_FILES
            or name.startswith("draw_")
            or name.startswith("mgl_batch_")
            or name.startswith("mgl_render_pass")
        ):
            continue
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
    for path in iter_src_files(".c"):
        if not os.path.basename(path).startswith("mgl_render_pass"):
            continue
        add("Lyr-Pass-Batch", path, scan_file(path, PASS_BATCH))

    # Lyr-Batch-Pass
    for path in iter_src_files(".c", ".cpp"):
        if not os.path.basename(path).startswith("mgl_batch_"):
            continue
        add("Lyr-Batch-Pass", path, scan_file(path, BATCH_PASS_OPS))

    # Lyr-No-Private (all src TUs + leftover Private filename ban)
    for path in iter_src_files(".c", ".cpp", ".m", ".mm", ".h", ".hpp"):
        add("Lyr-No-Private", path, scan_file(path, PRIVATE_INC))
    leftover: list[tuple[int, str]] = []
    for dirpath, _, files in os.walk(os.path.join(ROOT, "MGL")):
        for name in files:
            if name in PRIVATE_LEFTOVER_NAMES or (
                name.startswith("MGLRenderer+") and name.endswith("_Private.h")
            ):
                leftover.append((1, rel(os.path.join(dirpath, name))))
    if leftover:
        hits[("Lyr-No-Private", "MGL/tree")] = leftover

    # Lyr-DualProxy: Phase 2 forbids any core.activeState assignment.
    for dirpath, _, files in os.walk(os.path.join(ROOT, "MGL")):
        for name in files:
            if not name.endswith((".c", ".cpp", ".h", ".hpp", ".m", ".mm")):
                continue
            path = os.path.join(dirpath, name)
            add("Lyr-DualProxy", path, scan_file(path, ACTIVE_STATE_ASSIGN))

    # Lyr-Ports-FanIn: one synthetic entry whose "count" is TU fan-in.
    ports_hits: list[tuple[int, str]] = []
    for path in iter_src_files(".c", ".cpp"):
        found = scan_file(path, PORTS_INCLUDE)
        if found:
            ports_hits.append((found[0][0], rel(path)))
    if ports_hits:
        hits[("Lyr-Ports-FanIn", "MGL/src")] = [
            (ln, path) for ln, path in ports_hits
        ]

    # Lyr-Ext (G6): extensions tree must stay on hooks surface.
    ext_root = os.path.join(ROOT, "MGL", "extensions")
    if os.path.isdir(ext_root):
        for dirpath, _, files in os.walk(ext_root):
            for name in sorted(files):
                if not name.endswith((".c", ".cpp", ".h", ".hpp")):
                    continue
                path = os.path.join(dirpath, name)
                add("Lyr-Ext", path, scan_file(path, EXT_FORBIDDEN))

    # Lyr-Ctx-Fwd (G5): named public/plan headers must not pull full Rec.
    for rel_h in CTX_FWD_HEADERS:
        path = os.path.join(INC, rel_h)
        if os.path.isfile(path):
            add("Lyr-Ctx-Fwd", path, scan_file(path, CTX_FULL_INCLUDE))

    # Lyr-Ctx-FanIn (G5): src TU count including glm_context.h.
    ctx_hits: list[tuple[int, str]] = []
    for path in iter_src_files(".c", ".cpp"):
        found = scan_file(path, CTX_FULL_INCLUDE)
        if found:
            ctx_hits.append((found[0][0], rel(path)))
    if ctx_hits:
        hits[("Lyr-Ctx-FanIn", "MGL/src")] = [
            (ln, path) for ln, path in ctx_hits
        ]

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
    ports_fanin = len(hits.get(("Lyr-Ports-FanIn", "MGL/src"), ()))
    if ports_fanin > PORTS_FANIN_LIMIT:
        bad += 1
        print(
            f"FAIL Lyr-Ports-FanIn: {ports_fanin} TUs include "
            f"mgl_renderer_ports.h (limit {PORTS_FANIN_LIMIT})"
        )
    ctx_fanin = len(hits.get(("Lyr-Ctx-FanIn", "MGL/src"), ()))
    if ctx_fanin > CTX_FANIN_LIMIT:
        bad += 1
        print(
            f"FAIL Lyr-Ctx-FanIn: {ctx_fanin} TUs include "
            f"glm_context.h (limit {CTX_FANIN_LIMIT})"
        )
    if bad:
        return 1
    total = sum(base.values())
    print(
        f"arch layer gates ok: {total} baselined violations in {len(base)} entries; "
        f"ports fan-in={ports_fanin}/{PORTS_FANIN_LIMIT}; "
        f"ctx fan-in={ctx_fanin}/{CTX_FANIN_LIMIT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
