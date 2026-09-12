# Offline tessellation domain checkers

Tools that apply the specification rules — and a faithful re-implementation of
the relevant CTS checks — directly to MGL's domain generator
(`MGL/src/mgl_tess_domain_gen.c` + `mgl_tess_factor_normalize.c`), so the
tessellator can be iterated on without a Metal device or a CTS build.

```sh
./build.sh                      # builds rule34 + dump_quad_points
./rule34 <spacing> <o0> <o1> <o2> <o3> <i0> <i1>
python3 spacing_sweep.py        # scores every run of the CTS vertex-spacing case
```

`spacing` takes the GL enum value: `0` equal, `1` fractional_even,
`2` fractional_odd.

## Rule 3 / Rule 4 checker

Source of truth: `external/OpenGL-Registry/extensions/ARB/ARB_tessellation_shader.txt`,
Appendix A.X (Tessellation Invariance).

* **Rule 3** — "For quad tessellation, if the subdivision generates a vertex with
  coordinates of `(x,0)` or `(0,x)`, it will also generate a vertex with
  coordinates of **exactly** `(1-x,0)` or `(0,1-x)`."
* **Rule 4** — "For quad tessellation, if vertices at `(x,0)` and `(1-x,0)` are
  generated when subdividing the `v==0` edge, vertices must be generated at
  `(0,x)` and `(0,1-x)` when subdividing an otherwise identical `u==0` edge."

```sh
$ ./rule34 0 7 7 7 7 4 5
spacing=0 outer=(7,7,7,7) inner=(4,5) points=40 bottom=6 left=6
  RULE3 miss: 1-0.571428537=0x3edb6db8 not on bottom edge
  ...
  RULE3 violations=3  RULE4 violations=0  => FAIL
```

**RULE4 is clean** since commit `6d205b4`: every edge now reads its position out
of the same function (mirrored *index* for the edges that run against the u/v
axes, barycentric components taken from that function instead of rebuilt with
`1 - u - v`), so the value sets of the `v==0` and `u==0` edges are identical.
`KHR-GL46.tessellation_shader.tessellation_invariance.invariance_rule4` passes.

The **RULE3** misses this checker still reports are stricter than the CTS case:
`{k/7}` is not closed under `x -> 1-x` in float (see below), so a checker that
demands the exact complement of *every* position always finds misses, while
`invariance_rule3` compares with a `1e-4` epsilon
(`esextcTessellationShaderInvariance.cpp:1540`) and passes. Close the set only
if something needs it; publishing `1-x` for every `x` on both edges would break
the "`<n>` segments" contract and the `vertex_spacing` cases.

## Marker-triangle check (`inner_tessellation_level_rounding`)

```sh
cc -O1 -IMGL/include -IMGL/include/GL -o marker marker_check.c \
   MGL/src/mgl_tess_domain_gen.c MGL/src/mgl_tess_factor_normalize.c -lm
./marker
```

Reproduces the CTS "marker triangle" search
(`esextcTessellationShaderQuads.cpp`) for `inner=(1,3)`, `outer=3`,
`fractional_odd`. It prints the unique x/y sets and whether a triangle with a
full-width base exists at the second y level.

This passes in CTS since `6d205b4`: with an inner level of one treated as its
limit (`f` = clamped level = 1, `n` = 3, so `n-<f>` = 2 and the two additional
segments have length zero) the subdivision publishes `0, 0, 1, 1`, the inner
mesh columns land exactly on the rectangle edges, and the boundary-strip joins
produce the expected marker triangles.

## CTS vertex-spacing check, offline

`cts_spacing_sim.py` re-implements `getEdgesForQuadsTessellation()` and
`verifyEdges()` from `esextcTessellationShaderVertexSpacing.cpp`;
`spacing_sweep.py` feeds it every run of
`KHR-GL46.tessellation_shader.vertex.vertex_spacing_primitive_mode_quads_vs_mode_*`
that `getTessellationLevelSetForPrimitiveMode(QUADS, 64,
INNER_AND_OUTER_LEVELS_USE_DIFFERENT_VALUES)` produces.

```
$ python3 spacing_sweep.py
equal           runs=144 rejected=  0
fractional_odd  runs= 48 rejected= 48
```

The `equal_spacing` sibling passes on all 144 runs, i.e. MGL matches the CTS
model whenever CTS applies it consistently. Every `fractional_odd` run is
rejected, always on the inner-quad edge whose level is 32, always
`expected 29, found 31`; the level-64 edges (clamped to 63) match at `61 = 61`.

That is a CTS-side accounting problem, not a driver bug: with the level clamp at
63 the FO branch's expectation `FOround(clamp(L-2)) - 2` coincides with the
grid-model truth `FOround(clamp(L)) - 2` for `L = 64` (both 63 - 2 = 61) but not
for `L = 32` (31 - 2 = 29 vs 33 - 2 = 31), while the equal-spacing branch of the
same test expects `m - 2` and is satisfied. Making the FO run pass would need an
interior mesh that contradicts the grid §2.X.2.2 describes ("The boundary of the
region covered by these triangles forms an inner rectangle, the edges of which
are subdivided by the grid vertices that lie on the edge") and would break the
equal-spacing run. See `docs/CTS_TESS_REMAINING_2026-09-10.md` §4 (round 28).

Single run, verbose:

```sh
python3 spacing_sweep.py --spacing fractional_odd --inner 32 64 --outer 1 32 1 32
```
