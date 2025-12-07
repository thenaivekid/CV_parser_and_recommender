#!/bin/bash
# Evaluation Pipeline: Generate GT → Evaluate Recommendations

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=.

echo "=========================================="
echo "Recommendation Evaluation Pipeline"
echo "=========================================="

# Parse args
GENERATE_GT=""
GT_FILE="data/evaluation/ground_truth.json"
TOP_K="1 5 10 20"
BATCH_SIZE=100
WORKERS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-gt) GENERATE_GT="true"; shift ;;
        --ground-truth) GT_FILE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--generate-gt] [--ground-truth FILE] [--top-k \"5 10 20\"] [--batch-size N] [--workers N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Step 1: Generate GT if needed
if [ -n "$GENERATE_GT" ]; then
    echo ""
    echo "Step 1: Generating Ground Truth"
    echo "----------------------------------------"
    python3 src/generate_ground_truth.py --output "$GT_FILE"
fi

# Check GT exists
if [ ! -f "$GT_FILE" ]; then
    echo "Error: Ground truth not found: $GT_FILE"
    echo "Run with --generate-gt first"
    exit 1
fi

# Step 2: Evaluate
echo ""
echo "Step 2: Evaluating Recommendations"
echo "----------------------------------------"
python3 src/evaluate_recommendations.py \
    --ground-truth "$GT_FILE" \
    --top-k $TOP_K \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
