#!/bin/bash
# Compatibility entry point retained for downstream build scripts.
#
# P4/P4.5 used to audit the dual-path callback bridge. That bridge and its
# census header were removed by the single-path renderer migration, so the
# terminal P5 checker now owns the active renderer invariants.
set -u

script_dir=$(cd "$(dirname "$0")" && pwd)
exec "$script_dir/check_p5_metalcpp.sh"
