#!/bin/bash
# Run job PDF processing script
# Usage: ./scripts/run_job_pdf_processing.sh /path/to/job/pdfs [--force] [--verbose] [--provider azure|gemini]

# Activate virtual environment
source .venv/bin/activate

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Job PDFs directory path is required"
    echo "Usage: $0 /path/to/job/pdfs [--force] [--verbose] [--provider azure|gemini]"
    echo ""
    echo "Examples:"
    echo "  $0 data/sample_jobs_pdf"
    echo "  $0 data/sample_jobs_pdf --force"
    echo "  $0 data/sample_jobs_pdf --verbose"
    echo "  $0 data/sample_jobs_pdf --provider gemini"
    exit 1
fi

# Run the job PDF processing script with all arguments
python src/process_jobs_pdf.py "$@"
