#!/usr/bin/env python3
"""Fail when a function name is defined in more than one MGL/src file.

Static copies of the same helper in several translation units are invisible
to the linker and to review, and drift apart over time.  For every name
reported:

  * same body in each file  -> keep one definition, declare it in a header;
  * different bodies        -> make them agree, or rename them apart.

Before adding a function, grep MGL/src and MGL/include for an existing
equivalent.  New names follow this prefix ledger:

  mglXxx(GLMContext ctx, ...)   GL entry implementation (glXxx dispatch)
  mgl<Domain>*                  implementation inside one domain
                                (mglBuffer*, mglTexture*, mglBatch*, ...)
  mgl<Domain>*ForOwner          forwards to a domain object held by an owner
  mglRender*                    render backend API (mgl_render_api_*.h)
  mglRenderer*                  renderer-level entry taking void *renderer
  mglBinding*                   GL-facing binding logic (mgl_binding_*.c)

scripts/c_duplicate_baseline.txt lists names that were already duplicated
when this check was added.  Remove a name there once it is deduplicated;
the check fails on stale entries so the list only shrinks.

gl_es.c is skipped: it is the ES-profile counterpart of gl_core.c and the
two are never linked together.
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / 'scripts' / 'c_duplicate_baseline.txt'
SKIP = {'gl_es.c'}
SUFFIXES = {'.c', '.cpp', '.m', '.mm'}
KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'defined'}

DEF_RE = re.compile(
    r'^(?!\s)(?!#)(?!typedef\b)(?:[A-Za-z_][\w*\s]*?[\s*])?'
    r'([A-Za-z_]\w*)\s*\(([^;{}()]*(?:\([^;{}()]*\)[^;{}()]*)*)\)'
    r'\s*(?:const\s*)?\{',
    re.M)
COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)


def definitions():
    defs = collections.defaultdict(list)
    for path in sorted((ROOT / 'MGL' / 'src').iterdir()):
        if path.suffix not in SUFFIXES or path.name in SKIP:
            continue
        text = path.read_text(errors='replace')
        text = COMMENT_RE.sub(lambda m: '\n' * m.group(0).count('\n'), text)
        for m in DEF_RE.finditer(text):
            name = m.group(1)
            if name in KEYWORDS:
                continue
            line = text.count('\n', 0, m.start()) + 1
            defs[name].append((path.relative_to(ROOT), line))
    return defs


def main():
    baseline = set()
    for raw in BASELINE.read_text().splitlines():
        raw = raw.split('#', 1)[0].strip()
        if raw:
            baseline.add(raw)

    dups = {name: locs for name, locs in definitions().items()
            if len({str(p) for p, _ in locs}) > 1}

    new = sorted(set(dups) - baseline)
    stale = sorted(baseline - set(dups))
    for name in new:
        where = ' '.join(f'{p}:{line}' for p, line in dups[name])
        print(f'duplicate definition: {name}: {where}')
    for name in stale:
        print(f'no longer duplicated, remove from {BASELINE.name}: {name}')
    if new or stale:
        print(__doc__)
        return 1
    print(f'check-c-duplicates: OK ({len(dups)} baselined)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
