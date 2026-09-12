# Offline Rule 3 / Rule 4 checker

`KHR-GL46.tessellation_shader.tessellation_invariance.invariance_rule4` compares
tessellation coordinates with **exact float equality**, so the only practical way
to iterate on the domain generator is a checker that applies the specification
rules directly instead of running the whole CTS case.

Source of truth: `external/OpenGL-Registry/extensions/ARB/ARB_tessellation_shader.txt`,
Appendix A.X (Tessellation Invariance).

* **Rule 3** — "For quad tessellation, if the subdivision generates a vertex with
  coordinates of `(x,0)` or `(0,x)`, it will also generate a vertex with
  coordinates of **exactly** `(1-x,0)` or `(0,1-x)`."
* **Rule 4** — "For quad tessellation, if vertices at `(x,0)` and `(1-x,0)` are
  generated when subdividing the `v==0` edge, vertices must be generated at
  `(0,x)` and `(0,1-x)` when subdividing an otherwise identical `u==0` edge."

## Usage

```sh
./build.sh
./rule34 <spacing> <o0> <o1> <o2> <o3> <i0> <i1>
```

`spacing` takes the GL enum value: `0` equal, `1` fractional_even,
`2` fractional_odd.

The current state of the failing CTS configuration:

```sh
$ ./rule34 0 7 7 7 7 4 5
spacing=0 outer=(7,7,7,7) inner=(4,5) points=40 bottom=6 left=6
  RULE3 miss: 1-0.857142866=0x3e124924 not on bottom edge
  RULE4 miss: x=0.142857149 0x3e124925 -> (0,x)=N (0,1-x)=Y
  RULE3 violations=3  RULE4 violations=3  => FAIL
```

## What the violations mean

For a 7-segment edge the subdivided positions are the exact float values `k/7`.
That set is **not closed** under `x -> 1-x`: `1 - 3/7` is `0x3f124924`, one ulp
away from `4/7 = 0x3f124925`. A symmetric set around `0.5` needs an odd number of
elements, and `{k/7}` does not have one, so satisfying Rule 3 requires publishing
additional positions.

Publishing the complement of every interior position (behind
`MGL_TESS_RULE3_CLOSURE`) drives **Rule 3 to zero violations** but leaves Rule 4
unsatisfied: the `v==0` edge gets its positions from the outer tessellation level
while the `u==0` edge gets its from the inner mesh subdivision, so the two edges
still publish different float values for the same coordinate.
