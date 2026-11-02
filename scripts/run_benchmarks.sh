#!/usr/bin/env bash
set -euo pipefail

# GraphRAG-QPP: Self-Contained Benchmark Runner
# Usage:
#   ./scripts/run_benchmarks.sh --mode mock          (Fast CPU test, ~10s)
#   ./scripts/run_benchmarks.sh --mode colab --pilot (20-query GPU validation, ~3m)
#   ./scripts/run_benchmarks.sh --mode colab --full  (1,000-query full run, ~65m)

MODE="mock"
LME_QUERIES=20
MH_QUERIES=20
OUTPUT_DIR="artifacts/benchmark_run"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --pilot)
      LME_QUERIES=10
      MH_QUERIES=10
      OUTPUT_DIR="artifacts/pilot_v2"
      shift
      ;;
    --full)
      LME_QUERIES=500
      MH_QUERIES=500
      OUTPUT_DIR="artifacts/full"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_COLAB_SECRETS=1
export VERTEX_PRODUCT=COLAB_ENTERPRISE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "Running GraphRAG-QPP Benchmark"
echo "Mode:             $MODE"
echo "LongMemEval-S:    $LME_QUERIES queries"
echo "Multi-Hop:        $MH_QUERIES queries"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="

if command -v uv &>/dev/null && [ -f "pyproject.toml" ]; then
  PYTHON_BIN="uv run python"
elif [ -f ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

$PYTHON_BIN scripts/run_benchmarks.py \
  --mode "$MODE" \
  --max-queries-lme "$LME_QUERIES" \
  --max-queries-multihop "$MH_QUERIES" \
  --output-dir "$OUTPUT_DIR" \
  --cleanup-stage context_filtered \
  --tau-sweep "0.0,0.05,0.10,0.15,0.20"

echo "Benchmark run complete! Results written to $OUTPUT_DIR"
