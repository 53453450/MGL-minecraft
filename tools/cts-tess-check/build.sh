#!/bin/sh
# Offline Rule 3 / Rule 4 checker for quad tessellation edges.
#
#   ./build.sh && ./rule34 <spacing> <o0> <o1> <o2> <o3> <i0> <i1>
#
# spacing: 0 = equal, 2 = fractional_odd, 1 = fractional_even (GL enum values)
# Verifies ARB_tessellation_shader Appendix A.X Rule 3 and Rule 4 directly on
# MGL's domain output, without running CTS.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
clang -O1 -I "$root/MGL/include" -I "$root/MGL/include/GL" \
  -o "$here/rule34" "$here/rule34_check.c" \
  "$root/MGL/src/mgl_tess_domain_gen.c" \
  "$root/MGL/src/mgl_tess_factor_normalize.c" -lm
echo "built $here/rule34"
