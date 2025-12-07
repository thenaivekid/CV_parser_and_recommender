# !/bin/bash
# Run Recommendation Benchmark Script to evaluate retrieval performance single stage vs 2-stage(taking top k semantically similar jobs only)
set -euo pipefail
source .venv/bin/activate
PYTHONPATH=. python src/benchmark_retrieval.py --sample-size 5