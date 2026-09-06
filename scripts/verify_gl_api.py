#!/usr/bin/env python3
"""Verify handwritten gl_core.c against GL 4.6 core commands from pinned gl.xml.

The generated inventory is the source of truth for which GL 4.6 core
commands exist. Overlay lists known extras (compat leftovers) and
documented missing entries. This script does not generate mgl* bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GL_XML = os.path.join(ROOT, "external", "OpenGL-Registry", "xml", "gl.xml")
GL_CORE = os.path.join(ROOT, "MGL", "src", "gl_core.c")
LOCK = os.path.join(ROOT, "MGL", "generated", "registry.lock")
OUT_COMMANDS = os.path.join(ROOT, "MGL", "generated", "gl46_core_commands.txt")
OUT_ABI = os.path.join(ROOT, "MGL", "generated", "gl46_abi.inc")
OVERLAY = os.path.join(ROOT, "MGL", "generated", "gl_api_overlay.txt")

CORE_EXPORT_RE = re.compile(
    r"^(?:[\w\s\*]+)\bgl([A-Za-z0-9_]+)\s*\(",
    re.M,
)


def read_lock_commit() -> str:
    commit = ""
    with open(LOCK, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("commit="):
                commit = line.split("=", 1)[1].strip()
    return commit


def parse_overlay(path: str) -> tuple[set[str], set[str]]:
    extra: set[str] = set()
    missing: set[str] = set()
    if not os.path.exists(path):
        return extra, missing
    section = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                section = line[:-1]
                continue
            name = line[2:] if line.startswith("gl") else line
            if section == "extra":
                extra.add(name)
            elif section == "missing":
                missing.add(name)
    return extra, missing


def gl46_core_commands(gl_xml: str) -> list[str]:
    tree = ET.parse(gl_xml)
    root = tree.getroot()
    required: set[str] = set()
    removed: set[str] = set()
    for feature in root.findall("feature"):
        if feature.get("api") != "gl":
            continue
        try:
            number = float(feature.get("number") or "0")
        except ValueError:
            continue
        if number > 4.6 + 1e-9:
            continue
        for req in feature.findall("require"):
            for cmd in req.findall("command"):
                name = cmd.get("name")
                if name:
                    required.add(name)
        for rem in feature.findall("remove"):
            profile = rem.get("profile")
            if profile and profile != "core":
                continue
            for cmd in rem.findall("command"):
                name = cmd.get("name")
                if name:
                    removed.add(name)
    core = sorted(name[2:] for name in required - removed if name.startswith("gl"))
    return core


def gl_core_exports(path: str) -> set[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return set(CORE_EXPORT_RE.findall(text))


def write_generated(commands: list[str], commit: str, digest: str) -> None:
    os.makedirs(os.path.dirname(OUT_COMMANDS), exist_ok=True)
    with open(OUT_COMMANDS, "w", encoding="utf-8") as fh:
        fh.write(f"# GL 4.6 core commands from gl.xml {commit} sha256={digest}\n")
        for name in commands:
            fh.write(f"gl{name}\n")
    with open(OUT_ABI, "w", encoding="utf-8") as fh:
        fh.write("/* Generated from pinned gl.xml. Do not edit. */\n")
        fh.write(f"/* registry {commit} sha256={digest} */\n")
        fh.write("#define MGL_GL46_CORE_COMMANDS(M) \\\n")
        for i, name in enumerate(commands):
            slash = " \\" if i + 1 < len(commands) else ""
            fh.write(f"    M({name}){slash}\n")
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite generated inventory")
    args = parser.parse_args()

    if not os.path.isfile(GL_XML):
        print(f"error: missing {GL_XML}; run scripts/fetch_opengl_registry.sh", file=sys.stderr)
        return 2
    commit = read_lock_commit()
    with open(GL_XML, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    commands = gl46_core_commands(GL_XML)
    extra, missing = parse_overlay(OVERLAY)
    exported = gl_core_exports(GL_CORE)
    wanted = set(commands)

    if args.write or not os.path.exists(OUT_COMMANDS):
        write_generated(commands, commit, digest)
    else:
        with open(OUT_COMMANDS, encoding="utf-8") as fh:
            existing = [ln.strip()[2:] for ln in fh if ln.startswith("gl")]
        if existing != commands:
            print("error: MGL/generated/gl46_core_commands.txt stale; run with --write", file=sys.stderr)
            return 1

    unexplained_missing = sorted(wanted - exported - missing)
    unexplained_extra = sorted(exported - wanted - extra)
    ok = not unexplained_missing and not unexplained_extra
    print(f"gl.xml {commit} sha256={digest}")
    print(f"GL 4.6 core commands: {len(commands)}")
    print(f"gl_core.c exports: {len(exported)}")
    print(f"overlay extra={len(extra)} missing={len(missing)}")
    if unexplained_missing:
        print("missing from gl_core.c (not in overlay):")
        for name in unexplained_missing[:40]:
            print(f"  gl{name}")
        if len(unexplained_missing) > 40:
            print(f"  ... {len(unexplained_missing) - 40} more")
    if unexplained_extra:
        print("extra in gl_core.c (not in overlay):")
        for name in unexplained_extra[:40]:
            print(f"  gl{name}")
        if len(unexplained_extra) > 40:
            print(f"  ... {len(unexplained_extra) - 40} more")
    if ok:
        print("verify-gl-api: ok")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
