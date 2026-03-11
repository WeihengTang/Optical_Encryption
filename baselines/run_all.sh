#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Run all baselines sequentially and print the final comparison table.
# Must be run from the MetaLens project root, e.g.:
#   bash baselines/run_all.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Step 1 / 5 — Export classical baseline metrics"
echo "============================================================"
python baselines/export_classical_metrics.py

echo ""
echo "============================================================"
echo " Step 2 / 5 — DPIR (non-blind deep, knows PSF)"
echo "============================================================"
python baselines/run_dpir.py

echo ""
echo "============================================================"
echo " Step 3 / 5 — NAFNet (blind adversary, zero-shot)"
echo "============================================================"
python baselines/run_nafnet.py

echo ""
echo "============================================================"
echo " Step 4 / 5 — Restormer (blind adversary, zero-shot)"
echo "============================================================"
python baselines/run_restormer.py

echo ""
echo "============================================================"
echo " Step 5 / 5 — Collect and print results"
echo "============================================================"
python baselines/collect_results.py
