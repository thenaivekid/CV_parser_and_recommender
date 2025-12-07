#!/bin/bash

# Recommendation Evaluation Pipeline Script
# Generates ground truth, runs evaluation, and displays quality metrics

set -e

echo "=========================================="
echo "Recommendation Evaluation Pipeline"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.

# Default values
CONFIG="configurations/config.yaml"
GT_OUTPUT_DIR="data/evaluation"
EVAL_OUTPUT_DIR="data/performance_reports"
SPLIT="test"
TOP_K=20
K_VALUES="5 10 20"
REGENERATE=""
NO_BASELINES=""
GENERATE_GT=""
GT_FORMAT="both"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --generate-ground-truth)
            GENERATE_GT="true"
            shift
            ;;
        --gt-format)
            GT_FORMAT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --k-values)
            K_VALUES="$2"
            shift 2
            ;;
        --regenerate)
            REGENERATE="--regenerate"
            shift
            ;;
        --no-baselines)
            NO_BASELINES="--no-baselines"
            shift
            ;;
        --output-dir)
            EVAL_OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE                Path to configuration file (default: configurations/config.yaml)"
            echo "  --generate-ground-truth      Generate ground truth dataset before evaluation"
            echo "  --gt-format FORMAT          Ground truth format: csv, json, or both (default: both)"
            echo "  --split SPLIT               Data split to evaluate: train, val, or test (default: test)"
            echo "  --top-k N                   Number of recommendations to evaluate (default: 20)"
            echo "  --k-values \"K1 K2 K3\"       K values for metrics (default: \"5 10 20\")"
            echo "  --regenerate                Regenerate recommendations instead of using existing ones"
            echo "  --no-baselines              Skip baseline evaluations (faster)"
            echo "  --output-dir DIR            Output directory for evaluation reports (default: data/performance_reports)"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Generate ground truth and run evaluation on test set"
            echo "  $0 --generate-ground-truth --split test"
            echo ""
            echo "  # Evaluate with top-10 recommendations only"
            echo "  $0 --top-k 10 --k-values \"5 10\""
            echo ""
            echo "  # Evaluate validation set without baselines (faster)"
            echo "  $0 --split val --no-baselines"
            echo ""
            echo "  # Regenerate recommendations and evaluate (don't use cached)"
            echo "  $0 --regenerate --split test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Generate ground truth if requested
if [ -n "$GENERATE_GT" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Generating Ground Truth Dataset"
    echo "=========================================="
    
    python3 src/generate_ground_truth.py \
        --config "$CONFIG" \
        --output-dir "$GT_OUTPUT_DIR" \
        --format "$GT_FORMAT"
    
    if [ $? -ne 0 ]; then
        echo "Error: Ground truth generation failed"
        exit 1
    fi
    
    echo ""
    echo "Ground truth generated successfully!"
fi

# Find the latest ground truth file
GT_CSV="$GT_OUTPUT_DIR/ground_truth_latest.csv"

if [ ! -f "$GT_CSV" ]; then
    echo "Error: Ground truth file not found at $GT_CSV"
    echo "Please run with --generate-ground-truth first, or provide a ground truth CSV file"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running Evaluation"
echo "=========================================="
echo "Ground Truth: $GT_CSV"
echo "Split: $SPLIT"
echo "Top-K: $TOP_K"
echo "K Values: $K_VALUES"
echo "Regenerate: ${REGENERATE:-No}"
echo "Skip Baselines: ${NO_BASELINES:-No}"
echo ""

# Build evaluation command
CMD="python3 src/evaluate_recommendations.py \
    --config $CONFIG \
    --ground-truth $GT_CSV \
    --split $SPLIT \
    --top-k $TOP_K \
    --k-values $K_VALUES \
    --output-dir $EVAL_OUTPUT_DIR"

if [ -n "$REGENERATE" ]; then
    CMD="$CMD $REGENERATE"
fi

if [ -n "$NO_BASELINES" ]; then
    CMD="$CMD $NO_BASELINES"
fi

# Run evaluation
eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to: $EVAL_OUTPUT_DIR"
echo ""
echo "To view the latest evaluation report:"
echo "  ls -lt $EVAL_OUTPUT_DIR/evaluation_*.json | head -n 1"
echo ""
echo "To generate a performance dashboard:"
echo "  ./scripts/generate_dashboard.sh"
echo ""
