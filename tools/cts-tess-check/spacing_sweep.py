#!/usr/bin/env python3
"""Score MGL's quad domain against the CTS vertex-spacing check, offline.

    ./spacing_sweep.py [--spacing equal|fractional_odd] [--inner i0 i1] [--outer o0 o1 o2 o3]

Without arguments it replicates the level set that
`TessellationShaderUtils::getTessellationLevelSetForPrimitiveMode(QUADS,
GL_MAX_TESS_GEN_LEVEL, INNER_AND_OUTER_LEVELS_USE_DIFFERENT_VALUES)` builds and
reports how many runs of
`KHR-GL46.tessellation_shader.vertex.vertex_spacing_primitive_mode_quads_vs_mode_*`
the CTS check accepts.  `cts_spacing_sim.py` is a faithful re-implementation of
`getEdgesForQuadsTessellation()` + `verifyEdges()`, and the point sets come from
`dump_quad_points` (built by `build.sh`), so no CTS build or Metal device is
needed.

Observed result (2026-09-12, MGL @ 8b1efdd):

    fractional_odd: 48/48 runs rejected, always on the inner-quad edge whose
                    level is 32 ("expected 29, found 31"); the level-64 edges
                    (clamped to 63) match at 61 = 61.
    equal:          144/144 runs accepted.

See docs/CTS_TESS_REMAINING_2026-09-10.md for the arithmetic behind that.
"""
import argparse
import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "cts_spacing_sim.py")
DUMP = os.path.join(HERE, "dump_quad_points")
GL_EQUAL = 0x0202
GL_FRACTIONAL_ODD = 0x8E7B
GL_FRACTIONAL_EVEN = 0x8E7C
SPACING = {"equal": GL_EQUAL, "fractional_odd": GL_FRACTIONAL_ODD,
           "fractional_even": GL_FRACTIONAL_EVEN}
BASE_VALUES = [1, 32, 64]           # CTS base_values minus the -1 entry
MAX_LEVEL = 64


def cts_runs(spacing):
    """Level set of the CTS case, after its own `inner <= 1` skip for FO."""
    runs = []
    for inner in itertools.product(BASE_VALUES, repeat=2):
        if inner[0] == inner[1]:
            continue
        if spacing == "fractional_odd" and (inner[0] <= 1 or inner[1] <= 1):
            continue
        for outer in itertools.product(BASE_VALUES, repeat=4):
            if (outer[0] == outer[1] or outer[1] == outer[2]
                    or outer[2] == outer[3]):
                continue
            runs.append((inner, outer))
    return runs


def check(spacing, inner, outer, verbose=False):
    dump = [DUMP, "4", str(SPACING[spacing]), str(inner[0]), str(inner[1])]
    dump += [str(v) for v in outer]
    pts = subprocess.run(dump, capture_output=True, text=True).stdout
    if not pts.strip():
        return None, ""
    sim = [sys.executable, SIM, spacing, str(inner[0]), str(inner[1])]
    sim += [str(v) for v in outer]
    res = subprocess.run(sim, input=pts, capture_output=True, text=True)
    if verbose:
        sys.stdout.write(res.stdout)
    return "RESULT: PASS" in res.stdout, res.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", default=None)
    ap.add_argument("--inner", nargs=2, type=float)
    ap.add_argument("--outer", nargs=4, type=float)
    args = ap.parse_args()

    if not os.path.exists(DUMP):
        sys.exit("missing %s - run ./build.sh first" % DUMP)

    if args.inner and args.outer:
        spacing = args.spacing or "fractional_odd"
        ok, out = check(spacing, args.inner, args.outer, verbose=True)
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    for spacing in ([args.spacing] if args.spacing
                    else ["equal", "fractional_odd"]):
        runs = cts_runs(spacing)
        rejected = []
        for inner, outer in runs:
            ok, out = check(spacing, inner, outer)
            if ok is False:
                rejected.append((inner, outer, out))
        print("%-15s runs=%3d rejected=%3d" % (spacing, len(runs),
                                               len(rejected)))
        for inner, outer, out in rejected[:2]:
            print("  inner=%s outer=%s" % (inner, outer))
            for line in out.splitlines():
                if "expect" in line:
                    print("     " + line.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
