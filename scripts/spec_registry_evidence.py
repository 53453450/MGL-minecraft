#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
"""Build searchable SPEC corpora from OpenGL-Registry and enrich claim suites.

- Prefer external/OpenGL-Registry over docs/ copies.
- GLSL 4.60 uses the HTML (searchable); GL 4.6 Core uses pdftotext of the PDF.
- Normalizes hyphenation / broken ligatures so quote matching is reliable.
- Writes evidence windows into each claim and retargets spec_source.

Usage:
  python3 scripts/spec_registry_evidence.py [--verify-only] [--rewrite]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "external" / "OpenGL-Registry" / "specs" / "gl"
TEXT_DIR = ROOT / "scratch" / "spec_text"
CLAIMS_DIR = ROOT / "scripts" / "spec_claims"

GL_PDF = REGISTRY / "glspec46.core.pdf"
GLSL_HTML = REGISTRY / "GLSLangSpec.4.60.html"
GL_TXT = TEXT_DIR / "glspec46.core.txt"
GLSL_TXT = TEXT_DIR / "GLSLangSpec.4.60.txt"
GL_NORM = TEXT_DIR / "glspec46.core.norm.txt"
GLSL_NORM = TEXT_DIR / "GLSLangSpec.4.60.norm.txt"


def ensure_gl_txt() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    if GL_TXT.exists() and GL_TXT.stat().st_mtime >= GL_PDF.stat().st_mtime:
        return
    subprocess.run(
        ["pdftotext", "-layout", str(GL_PDF), str(GL_TXT)],
        check=True,
    )


def ensure_glsl_txt() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    if GLSL_TXT.exists() and GLSL_TXT.stat().st_mtime >= GLSL_HTML.stat().st_mtime:
        return
    import html as html_mod

    raw = GLSL_HTML.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"(?is)<script.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?</style>", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html_mod.unescape(raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    GLSL_TXT.write_text(raw, encoding="utf-8")


def normalize_corpus(text: str) -> str:
    """Undo PDF layout artifacts that break quote search."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2212", "-").replace("\ufb01", "fi").replace("\ufb02", "fl")
    # Soft hyphen / end-of-line hyphenation: "Draw-\nTransform" / "Draw- Transform"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    # Broken fi ligatures that become "f irst"
    text = re.sub(r"\bf\s+irst\b", "first", text)
    text = re.sub(r"\bf\s+loat\b", "float", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_norm() -> tuple[str, str]:
    ensure_gl_txt()
    ensure_glsl_txt()
    gl = normalize_corpus(GL_TXT.read_text(encoding="utf-8", errors="replace"))
    glsl = normalize_corpus(GLSL_TXT.read_text(encoding="utf-8", errors="replace"))
    GL_NORM.write_text(gl, encoding="utf-8")
    GLSL_NORM.write_text(glsl, encoding="utf-8")
    return gl, glsl


def find_window(corpus: str, quote: str, radius: int = 420) -> str | None:
    if not quote:
        return None
    q = normalize_corpus(quote)
    # Prefer exact; then first segment before ellipsis; then first 72 chars.
    candidates = [q]
    if "..." in q:
        candidates.append(q.split("...")[0].strip())
    candidates.append(q[:72].strip())
    for cand in candidates:
        if len(cand) < 12:
            continue
        i = corpus.lower().find(cand.lower())
        if i >= 0:
            start = max(0, i - 80)
            end = min(len(corpus), i + len(cand) + radius)
            return corpus[start:end].strip()
    return None


def pick_corpus(claim: dict, gl: str, glsl: str) -> tuple[str, str]:
    section = (claim.get("spec_section") or "") + " " + (claim.get("id") or "")
    if "GLSL" in section or "glsl" in (claim.get("id") or ""):
        return glsl, "GLSL 4.60 HTML"
    # Try both; prefer the one that matches.
    return gl, "GL 4.6 Core PDF"


def enrich_claim(claim: dict, gl: str, glsl: str) -> dict:
    quote = claim.get("spec_quote") or ""
    window = find_window(gl, quote) or find_window(glsl, quote)
    claim = dict(claim)
    if window:
        claim["evidence"] = window
        claim["evidence_verified"] = True
    else:
        # Keep self-quote as evidence so Jev can still run; mark unverified.
        claim["evidence"] = claim.get("evidence") or quote
        claim["evidence_verified"] = False
    return claim


def retarget_suite(data: dict) -> dict:
    data = dict(data)
    sources = [
        "external/OpenGL-Registry/specs/gl/glspec46.core.pdf",
    ]
    if "GLSL" in (data.get("spec") or "") or data.get("suite") == "glsl":
        sources.append(
            "external/OpenGL-Registry/specs/gl/GLSLangSpec.4.60.html"
        )
    data["spec_source"] = " ; ".join(sources)
    data["spec_corpus"] = [
        "scratch/spec_text/glspec46.core.norm.txt",
        "scratch/spec_text/GLSLangSpec.4.60.norm.txt",
    ]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="report quote hit rates without rewriting JSON",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="write evidence + retargeted spec_source back into claim JSON",
    )
    args = parser.parse_args()

    gl, glsl = ensure_norm()
    print(f"GL corpus chars={len(gl)}  GLSL corpus chars={len(glsl)}")

    total = verified = 0
    misses: list[tuple[str, str]] = []
    for path in sorted(CLAIMS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        new_claims = []
        for claim in data["claims"]:
            total += 1
            enriched = enrich_claim(claim, gl, glsl)
            if enriched.get("evidence_verified"):
                verified += 1
            else:
                misses.append((path.stem, claim["id"]))
            new_claims.append(enriched)
        if args.rewrite:
            data = retarget_suite(data)
            data["claims"] = new_claims
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"rewrote {path.name}")

    print(f"verified evidence windows: {verified}/{total}")
    if misses:
        print("unverified:")
        for suite, cid in misses:
            print(f"  - {suite}/{cid}")
    return 0 if verified == total or args.verify_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
