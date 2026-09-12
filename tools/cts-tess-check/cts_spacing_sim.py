#!/usr/bin/env python3
"""Simulate CTS TessellationShaderVertexSpacing on MGL's point-mode quad domain.

Faithful re-implementation of:
  - getTessellationLevelAfterVertexSpacing (fractional odd/even, equal)
  - getEdgesForQuadsTessellation (2-iteration corner extraction + edge grouping)
  - verifyEdges (segment-delta counting against the expected counter)
Reads the point set on stdin (first line = count, then "x y" per line).
"""
import sys
import math
from collections import namedtuple

EPS = 1e-3
MAXLEVEL = 64

Pt = namedtuple("Pt", "x y")


def level_after_spacing(spacing, level, maxlevel=MAXLEVEL):
    """CTS getTessellationLevelAfterVertexSpacing: (clamped, clamped_and_rounded)."""
    if spacing == "equal":
        level = min(max(level, 1.0), float(maxlevel))
        return level, float(int(math.ceil(level) + 0.5))
    if spacing == "fe":
        level = min(max(level, 2.0), float(maxlevel))
        t = int(math.ceil(level) + 0.5)
        if t % 2 != 0:
            t += 1
        return level, float(t)
    # fractional odd
    level = min(max(level, 1.0), float(maxlevel - 1))
    t = int(math.ceil(level) + 0.5)
    if t % 2 != 1:
        t += 1
    return level, float(t)


def on_line(a, b, p):
    dx, dy = a.x - b.x, a.y - b.y
    den = math.sqrt(dx * dx + dy * dy)
    if den == 0.0:
        return False
    d = abs(dy * p.x - dx * p.y + a.x * b.y - b.x * a.y) / den
    return abs(d) < EPS


def same(a, b):
    return abs(a.x - b.x) < EPS and abs(a.y - b.y) < EPS


def farthest(coords, sx, sy):
    best, bd = None, -1.0
    for c in coords:
        dx, dy = c.x - 0.5, c.y - 0.5
        if not (dx * sx >= 0.0 and dy * sy >= 0.0):
            continue
        d = math.sqrt(dx * dx + dy * dy)
        if d > bd:
            bd, best = d, c
    return best


def erase_one(coords, p):
    for i, c in enumerate(coords):
        if same(c, p):
            del coords[i]
            return True
    return False


def collect_edge(coords, start, end):
    """CTS edge grouping for one corner pair: points on the line, closer to
    `start` than `end` is, de-duplicated, then sorted by distance from start."""
    dx, dy = end.x - start.x, end.y - start.y
    len_sq = dx * dx + dy * dy
    pts = [start]
    if not same(start, end):
        pts.append(end)
    for c in coords:
        if same(c, start) or same(c, end):
            continue
        if not on_line(start, end, c):
            continue
        ddx, ddy = c.x - start.x, c.y - start.y
        if ddx * ddx + ddy * ddy >= len_sq:
            continue
        if any(same(k, c) for k in pts):
            continue
        pts.append(c)
    pts.sort(key=lambda p: (p.x - start.x) ** 2 + (p.y - start.y) ** 2)
    return pts


def quads_edges(coords, spacing, inner, outer):
    coords = list(coords)
    n_iter = 2 if inner[0] > 1.0 else 1
    edges = []
    for _ in range(n_iter):
        tl = farthest(coords, -1, +1)
        tr = farthest(coords, +1, +1)
        bl = farthest(coords, -1, -1)
        br = farthest(coords, +1, -1)
        if tl is None or tr is None or bl is None or br is None:
            break
        for c in (tl, tr, bl, br):
            erase_one(coords, c)
        # tl_tr, tr_br, br_bl, bl_tl
        e0 = collect_edge(coords, tl, tr)
        e1 = collect_edge(coords, tr, br)
        e2 = collect_edge(coords, br, bl)
        e3 = collect_edge(coords, bl, tl)
        for e in (e0, e1, e2, e3):
            for p in e:
                erase_one(coords, p)
        edges.extend([(e0, outer[3]), (e1, outer[2]), (e2, outer[1]), (e3, outer[0])])
    return edges


def verify(points, spacing, inner, outer):
    """Returns (ok, list of (edge_index, len, segments, distinct, expected, msg))."""
    # iteration-1 edge descriptors use inner-based levels; CTS adds them after
    # the first four, so rebuild with the right level per iteration.
    coords = list(points)
    n_iter = 2 if inner[0] > 1.0 else 1
    all_edges = []
    for it in range(n_iter):
        tl = farthest(coords, -1, +1)
        tr = farthest(coords, +1, +1)
        bl = farthest(coords, -1, -1)
        br = farthest(coords, +1, -1)
        if tl is None or tr is None or bl is None or br is None:
            return False, [(-1, 0, 0, 0, 0, "could not find 4 corners in iteration %d" % it)]
        for c in (tl, tr, bl, br):
            erase_one(coords, c)
        raw = [collect_edge(coords, tl, tr), collect_edge(coords, tr, br),
               collect_edge(coords, br, bl), collect_edge(coords, bl, tl)]
        if it == 0:
            lv = list(outer[3::-1])  # tl_tr<-outer[3], tr_br<-outer[2], br_bl<-outer[1], bl_tl<-outer[0]
            outer_lv = list(outer[3::-1])
        else:
            lv = [inner[0] - 2.0, inner[1] - 2.0, inner[0] - 2.0, inner[1] - 2.0]
            outer_lv = list(outer[3::-1])
        for k in range(4):
            for p in raw[k]:
                erase_one(coords, p)
            all_edges.append((raw[k], lv[k], outer_lv[k]))

    report = []
    ok = True
    for n_edge, (pts, tess_level, outermost) in enumerate(all_edges):
        if len(pts) < 2:
            continue
        deltas = []
        for i in range(len(pts) - 1):
            dx = pts[i].x - pts[i + 1].x
            dy = pts[i].y - pts[i + 1].y
            d = math.sqrt(dx * dx + dy * dy)
            for ent in deltas:
                if abs(ent[0] - d) < EPS:
                    ent[1] += 1
                    break
            else:
                deltas.append([d, 1])
        _, clamped_rounded = level_after_spacing(spacing, tess_level)
        if spacing == "equal":
            # single delta of length edge_len/clamped_rounded
            edge_len = math.sqrt((pts[0].x - pts[-1].x) ** 2 + (pts[0].y - pts[-1].y) ** 2)
            exp = edge_len / clamped_rounded if clamped_rounded else 0
            good = len(deltas) == 1 and abs(deltas[0][0] - exp) <= EPS
            msg = "equal: expect 1 delta %.6g, got %s" % (exp, deltas)
        else:
            expected_counter = clamped_rounded
            if n_edge >= 4:
                expected_counter = clamped_rounded - 2.0 * (n_edge // 4)
            if tess_level <= 0.0:
                expected_counter = 1.0
            if len(deltas) == 1:
                good = deltas[0][1] == int(expected_counter)
                msg = "expect %d segments, got %d (delta %.6g)" % (
                    int(expected_counter), deltas[0][1], deltas[0][0])
            elif len(deltas) == 2:
                c = sorted(e[1] for e in deltas)
                good = c == [2, int(expected_counter) - 2]
                msg = "2-delta expect {2,%d} got %s" % (int(expected_counter) - 2, c)
            else:
                good = False
                msg = "expect <=2 distinct deltas, got %d (%s)" % (len(deltas), deltas)
        if not good:
            ok = False
        report.append((n_edge, len(pts), deltas, clamped_rounded, expected_counter if spacing != "equal" else None, msg))
    return ok, report


def load(stream):
    lines = [l for l in stream.read().split("\n") if l.strip()]
    n = int(lines[0])
    pts = []
    for l in lines[1:1 + n]:
        a, b = l.split()
        pts.append(Pt(float(a), float(b)))
    return pts


def main():
    spacing = sys.argv[1]
    inner = (float(sys.argv[2]), float(sys.argv[3]))
    outer = tuple(float(v) for v in sys.argv[4:8])
    pts = load(sys.stdin)
    ok, rep = verify(pts, spacing, inner, outer)
    print("points=%d spacing=%s inner=%s outer=%s" % (len(pts), spacing, inner, outer))
    for n_edge, n, deltas, cr, exp, msg in rep:
        print("  edge%-2d pts=%-4d clamped_rounded=%-4g %s" % (n_edge, n, cr, msg))
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
