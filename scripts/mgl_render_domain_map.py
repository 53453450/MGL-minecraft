#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
"""Assign every mgl* definition in mgl_render.cpp to a domain.

Name-based, not line-band based: section 6.7.7 of
docs/C_LAYER_ARCHITECTURE_REVIEW.md showed domains are interleaved.
The first matching rule wins. `core` is the residual, not a real domain.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "MGL" / "src"
SOURCES = [
    SRC_DIR / "mgl_render.cpp",
    SRC_DIR / "mgl_render_binding.cpp",
    SRC_DIR / "mgl_render_buffer.cpp",
    SRC_DIR / "mgl_render_command.cpp",
    SRC_DIR / "mgl_render_draw.cpp",
    SRC_DIR / "mgl_render_lifecycle.cpp",
    SRC_DIR / "mgl_render_pixel.cpp",
    SRC_DIR / "mgl_render_query.cpp",
    SRC_DIR / "mgl_render_readback.cpp",
    SRC_DIR / "mgl_render_texture.cpp",
]
OUT = ROOT / "scripts" / "mgl_render_domain_map.json"

# (domain, name tokens). Longer / more specific domains are listed first.
RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "pixel_convert",
        (
            "Snorm",
            "Unorm",
            "Half",
            "Unpack",
            "Pack",
            "Expand",
            "Swizzle",
            "Integer",
            "Float",
        ),
    ),
    ("readback", ("Readback", "Clear", "GetBytes", "ReplaceRegion")),
    (
        "texture",
        (
            "Texture",
            "Pixel",
            "Texel",
            "Mip",
            "Image",
            "Sampler",
            "InternalFormat",
        ),
    ),
    ("binding", ("Binding", "Bind", "VertexAttrib", "Attrib", "Pipeline")),
    (
        "command",
        (
            "Command",
            "Encode",
            "Encoder",
            "Blit",
            "Fence",
            "Recovery",
            "Pass",
            "Dispatch",
        ),
    ),
    ("buffer", ("Buffer", "Staging")),
    ("draw", ("Draw", "Tess", "Primitive", "Cull", "Index")),
    ("query", ("Query", "Get", "Is", "Supports", "Plan")),
    (
        "lifecycle",
        ("Create", "Destroy", "Init", "Release", "Reset", "Retain", "Alloc", "Load"),
    ),
]


VERBS = {
    "Set",
    "Get",
    "Is",
    "Create",
    "Destroy",
    "Init",
    "Release",
    "Reset",
    "Retain",
    "Alloc",
    "Load",
    "Begin",
    "End",
    "Add",
    "Fill",
    "Build",
    "Use",
    "Make",
    "Note",
    "Mark",
    "Copy",
    "Read",
    "Write",
    "Wait",
    "Prefer",
    "Promote",
    "Repair",
    "Emulate",
    "Apply",
    "Update",
    "Ensure",
    "Find",
    "Has",
    "Can",
    "Should",
    "Needs",
    "Supports",
}

# Acronyms kept whole so GL/MTL do not split into single letters.
ACRONYMS = (
    "MSAA",
    "MTL",
    "PSO",
    "CPU",
    "GPU",
    "FBO",
    "VAO",
    "GLSL",
    "GL",
)


def tokens_of(name: str) -> list[str]:
    rest = name[len("mglRender") :] if name.startswith("mglRender") else name[3:]
    out: list[str] = []
    while rest:
        for acro in ACRONYMS:
            if rest.startswith(acro) and (len(rest) == len(acro) or rest[len(acro)].isupper()):
                out.append(acro)
                rest = rest[len(acro) :]
                break
        else:
            match = re.match(r"[A-Z][a-z0-9]*|[A-Z]+(?![a-z])", rest)
            if not match:
                return out
            out.append(match.group(0))
            rest = rest[len(match.group(0)) :]
    return out


def domain_of(name: str) -> str:
    toks = tokens_of(name)
    if "Swizzle" in toks or "Snorm" in toks or "Unorm" in toks:
        return "pixel_convert"
    body = [tok for tok in toks if tok not in VERBS] or toks
    for domain, keys in RULES:
        if any(tok in keys for tok in body[:4]):
            return domain
    if toks and toks[0] in ("Create", "Destroy", "Init", "Release", "Reset", "Retain", "Alloc", "Load"):
        return "lifecycle"
    if toks and toks[0] in ("Get", "Is", "Supports", "Plan"):
        return "query"
    return "core"


def definitions(text: str, source: str) -> list[dict]:
    lines = text.splitlines()
    found: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        if line[:1] in " \t#":
            continue
        match = re.search(r"\b(mgl\w+)\s*\(", line)
        if not match:
            continue
        name = match.group(1)
        if line.startswith(name):
            continue
        # Declaration (semicolon before brace) vs definition.
        window = "\n".join(lines[i : i + 12])
        brace = window.find("{")
        semi = window.find(";")
        if brace < 0 or (semi >= 0 and semi < brace):
            continue
        if name in seen:
            continue
        seen.add(name)
        found.append(
            {
                "symbol": name,
                "file": source,
                "line": i + 1,
                "domain": domain_of(name),
                "tokens": tokens_of(name)[:4],
            }
        )
    return found


def main() -> None:
    funcs: list[dict] = []
    seen: set[str] = set()
    for path in SOURCES:
        for row in definitions(path.read_text(encoding="utf-8", errors="replace"), path.name):
            if row["symbol"] in seen:
                continue
            seen.add(row["symbol"])
            funcs.append(row)
    counts = Counter(row["domain"] for row in funcs)
    by_file = Counter(row["file"] for row in funcs)
    payload = {
        "source": [path.name for path in SOURCES],
        "method": "name tokens after skipping verbs; Swizzle/Snorm/Unorm forced to pixel_convert; residual is core",
        "rules": [{"domain": domain, "tokens": list(keys)} for domain, keys in RULES],
        "counts": dict(counts),
        "by_file": dict(by_file),
        "function_count": len(funcs),
        "functions": funcs,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"functions={len(funcs)} wrote {OUT.relative_to(ROOT)}")
    for domain, count in counts.most_common():
        print(f"  {count:4} {domain}")
    for name, count in by_file.most_common():
        print(f"  {count:4} {name}")


if __name__ == "__main__":
    main()
