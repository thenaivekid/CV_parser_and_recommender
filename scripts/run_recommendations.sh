#!/bin/bash

# Generate Job Recommendations Script

set -e

echo "=========================================="
echo "Job Recommendation Generator"
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
CANDIDATE_ID=""
TOP_K=""
OUTPUT_DIR="data/recommendations"
NO_SAVE_DB=""
CONFIG="configurations/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --candidate-id)
            CANDIDATE_ID="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-save-db)
            NO_SAVE_DB="--no-save-db"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --candidate-id ID    Generate recommendations for specific candidate only"
            echo "  --top-k N           Number of top recommendations per candidate (default: all)"
            echo "  --output-dir DIR    Directory to save JSON output files (default: data/recommendations)"
            echo "  --no-save-db        Do not save recommendations to database"
            echo "  --config FILE       Path to configuration file (default: configurations/config.yaml)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Generate recommendations for all candidates"
            echo "  $0"
            echo ""
            echo "  # Generate top 10 recommendations for all candidates"
            echo "  $0 --top-k 10"
            echo ""
            echo "  # Generate recommendations for specific candidate"
            echo "  $0 --candidate-id ENGINEERING_10030015"
            echo ""
            echo "  # Generate and save to custom directory"
            echo "  $0 --output-dir my_recommendations --top-k 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 src/generate_recommendations.py --config $CONFIG"

if [ -n "$CANDIDATE_ID" ]; then
    CMD="$CMD --candidate-id $CANDIDATE_ID"
fi

if [ -n "$TOP_K" ]; then
    CMD="$CMD --top-k $TOP_K"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$NO_SAVE_DB" ]; then
    CMD="$CMD $NO_SAVE_DB"
fi

# Run the command
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Recommendation generation complete!"
echo "=========================================="
