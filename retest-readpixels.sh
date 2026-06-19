#!/bin/zsh
# Quick re-test harness for the glReadPixels readback rework.
# Usage: ./retest-readpixels.sh [sample|full]
#   sample (default): /tmp/rp-sample.txt (~25 cases, fast ~1min)
#   full:             /tmp/rp-full.txt  (~5653 cases)
set -e
MODE="${1:-sample}"
case "$MODE" in
  sample) LIST=/tmp/rp-sample.txt ;;
  full)   LIST=/tmp/rp-full.txt ;;
  *) echo "usage: $0 [sample|full]"; exit 1 ;;
esac

CTSDIR=/Users/fterward/VK-GL-CTS-build-mgl-target/external/openglcts/modules
OUT=/Users/fterward/VK-GL-CTS-build-mgl-target/mgl-rp-$MODE-$(date +%Y%m%d-%H%M%S)

echo "Running $MODE ($(<"$LIST" wc -l | tr -d ' ') cases) -> $OUT"
python3 -u /Users/fterward/VK-GL-CTS-build-mgl-target/run_mgl_cts_cases.py \
  --glcts "$CTSDIR"/glcts --caselist "$LIST" --workdir "$CTSDIR" \
  --outdir "$OUT" --dyld-library-path /Users/fterward/MGL-minecraft \
  --timeout 45 2>&1 | tail -1

echo
echo "=== aggregate ==="
python3 - "$OUT/summary.tsv" <<'PY'
import sys,collections
c=collections.Counter()
for l in open(sys.argv[1]):
    p=l.rstrip('\n').split('\t')
    if len(p)>=3 and p[0]!='index': c[p[2]]+=1
for k in ('pass','fail','crash','not_supported','timeout'):
    print(f"  {k:14} {c.get(k,0)}")
PY

# For the single-case deep dive, break down the failure modes (compressed_red_format_red).
echo
echo "=== compressed_red_format_red failure-mode breakdown (if present) ==="
Q="$OUT/qpa/fail"/*.compressed_red_format_red.qpa 2>/dev/null
Q=$(ls "$OUT"/qpa/*/*.compressed_red_format_red.qpa 2>/dev/null | head -1)
[ -n "$Q" ] || Q=$(find "$OUT" -name '*compressed_red_format_red.qpa' 2>/dev/null | head -1)
if [ -n "$Q" ]; then
  echo "  qpa: $Q"
  for pat in "Error during glGetTexImage" "Gradient comparison failed during ReadPixels" "Gradient comparison failed during GetTexImage" "Valid format used but glReadPixels failed" "Invalid format used but glReadPixels succeeded"; do
    printf "  %3d  %s\n" "$(grep -c "$pat" "$Q" 2>/dev/null)" "$pat"
  done
else
  echo "  (compressed_red_format_red not in this run)"
fi
