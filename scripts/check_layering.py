#!/usr/bin/env python3
"""Layering lint for MGL (docs/AUDIT_LAYERING_2026-10-17.md, section 5.3).

Six rules, each a count of offending source lines per file:

  C1  L0 forwarding files must not mention Metal handles or release them:
      `mtl_*` identifiers, `MTL::`, `mglSafeReleaseMetalObj`.
  C2  L3 backend files must not read GL object fields or branch on GL enums:
      `->samples`, `->target`, `is_render_target`, `dirty_bits`, `GL_[A-Z]*`.
  C3  Plan files (*_plan.c) must not include mgl_render_internal.h or use
      `STATE(`.
  C4  GL object type headers must not declare `void *mtl_*` fields.
  C5  Only the drivers that change renderer state hold MGLRendererStateAreas;
      everything else takes the read-only MGLRendererStateView.  The count of
      areas references per file may not grow, and a new file starts at zero.
  C6  Only the blit plan may assign a row-order field
      (`src_metal_y`/`dst_metal_y`/`scaled_dst_metal_y`/`blit_needs_flip`).
      Consumers read them; deriving one elsewhere recreates the second
      orientation source the GL-order work removed.

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
    'draw_buffers.c', 'rendering.c', 'buffers.c', 'samplers.c',
    'framebuffers.c', 'program.c', 'uniforms.c', 'tex_param.c',
    'vertex_arrays.c', 'vertex_buffers.c', 'shaders.c', 'state.c', 'get.c',
    'fence.c', 'compute.c',
]
C1_RE = re.compile(r'\bmtl_\w+|MTL::|\bmglSafeReleaseMetalObj\b')

# L3: Metal encoding.
L3_GLOBS = ['mgl_render_*.cpp', 'mgl_batch_*_encode.c']
L3_FILES = ['mgl_pso_build_ops.c', 'mgl_renderer_backend.cpp',
            'mgl_renderer_ports.c', 'mgl_texture_bind.c']
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

# Renderer state: the mutable areas (see mgl_renderer_ports.h).
C5_RE = re.compile(r'\bMGLRendererStateAreas\b|\bmglRendererFillStateAreas\b')

# Blit plan ownership (C6).  The plan is the only place allowed to derive a
# row-order field from the per-side storage bits.
C6_RE = re.compile(
    r'(?:->|\.)\s*(?:src_metal_y|dst_metal_y|scaled_dst_metal_y|blit_needs_flip)'
    r'\s*=')
C6_OWNER = 'mgl_render_command.cpp'

COMMENT_RE = re.compile(
    r'/\*.*?\*/|//[^\n]*', re.S)


def strip_comments(text):
    return COMMENT_RE.sub(lambda m: '\n' * m.group(0).count('\n'), text)


# A function that legitimately owns writable state areas marks its declaration
# with this comment; C5 stops counting its Area/fill lines.  The driver side
# (and the port setters that write through the areas) is where the areas are
# supposed to live, so those lines are not violations to shrink.
EXEMPT_MARK = 'C5-EXEMPT'


def exempt_ranges(path):
    """Line ranges of functions carrying a C5-EXEMPT marker."""
    try:
        lines = path.read_text(errors='replace').splitlines()
    except OSError:
        return []
    ranges = []
    starts = [i for i, l in enumerate(lines)
              if l.startswith(('void ', 'int ', 'bool ', 'static ')) or
                 (i == 0 and l.strip())]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        if any(EXEMPT_MARK in l for l in lines[start:end]):
            ranges.append((start + 1, end))
    return ranges


def scan(path, regex):
    text = strip_comments(path.read_text(errors='replace'))
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)
            if regex.search(line)]


def find_sources(name):
    matches = sorted(SRC.rglob(name))
    return matches or [SRC / name]


def targets():
    for name in L0_FILES:
        for p in find_sources(name):
            yield 'C1', p, C1_RE
    for p in sorted(SRC.rglob('mgl_textures*.c')):
        yield 'C1', p, C1_RE
    l3 = {p for g in L3_GLOBS for p in SRC.rglob(g)}
    for name in L3_FILES:
        l3.update(find_sources(name))
    for p in sorted(l3):
        yield 'C2', p, C2_RE
    for p in sorted(SRC.rglob(PLAN_GLOB)):
        yield 'C3', p, C3_RE
    for name in TYPE_HEADERS:
        yield 'C4', INC / name, C4_RE
    for p in sorted(SRC.rglob('*')):
        if p.suffix in ('.c', '.cpp', '.m', '.mm'):
            yield 'C5', p, C5_RE
            if p.name != C6_OWNER:
                yield 'C6', p, C6_RE


def collect():
    hits = {}
    for rule, path, regex in targets():
        if not path.exists():
            sys.exit(f'check_layering: missing file {path.relative_to(ROOT)}')
        found = scan(path, regex)
        if rule == 'C5':
            exempt = exempt_ranges(path)
            found = [(ln, txt) for ln, txt in found
                     if EXEMPT_MARK not in txt and
                     not any(a <= ln <= b for a, b in exempt)]
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
