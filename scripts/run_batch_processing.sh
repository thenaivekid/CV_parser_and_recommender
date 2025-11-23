#!/bin/bash
# Batch Processing Script for CV Parser and Recommender System

set -e  # Exit on error

echo "=========================================="
echo "CV Parser & Recommender - Batch Processing"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with required environment variables"
    exit 1
fi

# Check database connection
echo "🔍 Checking database connection..."
if ! docker exec cv-job-pgvector pg_isready -U cv_user -d cv_job_db > /dev/null 2>&1; then
    echo "❌ Database not ready!"
    echo "Please start the database:"
    echo "  docker-compose up -d"
    exit 1
fi

echo "✓ Database connection OK"
echo ""

# Run the batch processor
echo "🚀 Starting batch processing..."
echo ""

python src/process_resumes.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Batch processing completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Batch processing failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
