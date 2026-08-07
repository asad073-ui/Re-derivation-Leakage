#!/usr/bin/env bash
# Days 3-5: the C0/C1/C2/C3 grid, 5 seeds each.
#
# C1 runs before C3 because the pre-registered gate is C3 - C1 and you want the floor
# in hand before you look at the treatment.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-5}"

for C in C0 C1 C2 C3; do
    echo
    echo "=================== $C ==================="
    python -m rdl.cli run-condition --condition "configs/conditions/${C}.yaml" --seeds "$SEEDS" "$@"
done

echo
echo "=== report ==="
python -m rdl.cli make-report

echo
echo "Now read docs/00_preregistration.md and apply the gate:"
echo "  SysRecall@5(C3, store) - SysRecall@5(C1) >= 20 points, AND laundering_rate(C3) >= 0.5,"
echo "  with non-overlapping 95% CIs over $SEEDS seeds."
echo "  Confound gate: the C3 result MUST survive under always_delegate."
echo "  Kill (day 5): C3 - C1 < 10 points -> re-scope."
