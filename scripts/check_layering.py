#!/usr/bin/env python3
"""Layering lint for MGL (docs/AUDIT_LAYERING_2026-10-17.md, section 5.3).

Four rules, each a count of offending source lines per file:

  C1  L0 forwarding files must not mention Metal handles or release them:
      `mtl_*` identifiers, `MTL::`, `mglSafeReleaseMetalObj`.
  C2  L3 backend files must not read GL object fields or branch on GL enums:
      `->samples`, `->target`, `is_render_target`, `dirty_bits`, `GL_[A-Z]*`.
  C3  Plan files (*_plan.c) must not include mgl_render_internal.h or use
      `STATE(`.
  C4  GL object type headers must not declare `void *mtl_*` fields.

Comments are ignored.  scripts/layering_baseline.txt records the counts that
existed when the check was added ("rule path count").  A file may not exceed
its baseline, and a baseline that is higher than the real count is stale, so
the numbers only go down.  Run with --update after paying down violations to
rewrite the baseline; --report prints every offending line.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'MGL' / 'src'
INC = ROOT / 'MGL' / 'include'
BASELINE = ROOT / 'scripts' / 'layering_baseline.txt'

# L0: GL entry points and their object tables.
L0_FILES = [
    'gl_core.c', 'mgl_draw_entry.c', 'mgl_texture_entries.c',
    'draw_buffers.c', 'rendering.c', 'textures.c', 'buffers.c', 'samplers.c',
    'framebuffers.c', 'program.c', 'uniforms.c', 'tex_param.c',
    'vertex_arrays.c', 'vertex_buffers.c', 'shaders.c', 'state.c', 'get.c',
    'fence.c', 'compute.c',
]
C1_RE = re.compile(r'\bmtl_\w+|MTL::|\bmglSafeReleaseMetalObj\b')

# L3: Metal encoding.
L3_GLOBS = ['mgl_render_*.cpp', 'mgl_batch_*_encode.c']
L3_FILES = ['mgl_pso_build_ops.c']
C2_RE = re.compile(
    r'->samples\b|->target\b|\bis_render_target\b|\bdirty_bits\b'
    r'|(?<![A-Za-z0-9_])GL_[A-Z]')

# L2: plans.
PLAN_GLOB = '*_plan.c'
C3_RE = re.compile(r'#\s*include\s*[<"]mgl_render_internal\.h[>"]|\bSTATE\s*\(')

# GL object type headers.
TYPE_HEADERS = [
    'mgl_types_buffer.h', 'mgl_types_framebuffer.h', 'mgl_types_program.h',
    'mgl_types_state.h', 'mgl_types_sync.h', 'mgl_types_texture.h',
    'mgl_types_vertex.h',
]
C4_RE = re.compile(r'\bvoid\s*\*\s*mtl_\w+\s*;')

COMMENT_RE = re.compile(
    r'/\*.*?\*/|//[^\n]*', re.S)


def strip_comments(text):
    return COMMENT_RE.sub(lambda m: '\n' * m.group(0).count('\n'), text)


def scan(path, regex):
    text = strip_comments(path.read_text(errors='replace'))
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)
            if regex.search(line)]


def targets():
    for name in L0_FILES:
        yield 'C1', SRC / name, C1_RE
    l3 = {p for g in L3_GLOBS for p in SRC.glob(g)}
    l3 |= {SRC / n for n in L3_FILES}
    for p in sorted(l3):
        yield 'C2', p, C2_RE
    for p in sorted(SRC.glob(PLAN_GLOB)):
        yield 'C3', p, C3_RE
    for name in TYPE_HEADERS:
        yield 'C4', INC / name, C4_RE


def collect():
    hits = {}
    for rule, path, regex in targets():
        if not path.exists():
            sys.exit(f'check_layering: missing file {path.relative_to(ROOT)}')
        found = scan(path, regex)
        if found:
            hits[(rule, str(path.relative_to(ROOT)))] = found
    return hits


def read_baseline():
    base = {}
    for raw in BASELINE.read_text().splitlines():
        raw = raw.split('#', 1)[0].strip()
        if raw:
            rule, path, count = raw.split()
            base[(rule, path)] = int(count)
    return base


def write_baseline(hits):
    lines = [
        '# Layering violations present when scripts/check_layering.py was added.',
        '# Format: rule path count.  Counts may only go down; see the script.',
        '',
    ]
    for (rule, path), found in sorted(hits.items()):
        lines.append(f'{rule} {path} {len(found)}')
    BASELINE.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true',
                    help='rewrite the baseline from the current tree')
    ap.add_argument('--report', action='store_true',
                    help='print every offending line')
    args = ap.parse_args()

    hits = collect()
    if args.update:
        write_baseline(hits)
        print(f'wrote {BASELINE.relative_to(ROOT)}: {len(hits)} entries, '
              f'{sum(len(v) for v in hits.values())} lines')
        return 0
    if args.report:
        for (rule, path), found in sorted(hits.items()):
            for line_no, text in found:
                print(f'{rule} {path}:{line_no}: {text}')
        return 0

    base = read_baseline() if BASELINE.exists() else {}
    bad = 0
    for key in sorted(set(hits) | set(base)):
        have = len(hits.get(key, ()))
        want = base.get(key, 0)
        rule, path = key
        if have > want:
            bad += 1
            print(f'{rule} {path}: {have} violations, baseline {want}')
            for line_no, text in hits[key]:
                print(f'    {path}:{line_no}: {text}')
        elif have < want:
            bad += 1
            print(f'{rule} {path}: stale baseline {want}, now {have} '
                  f'(run --update)')
    if bad:
        return 1
    total = sum(base.values())
    print(f'layering ok: {total} baselined violations in {len(base)} entries')
    return 0


if __name__ == '__main__':
    sys.exit(main())
