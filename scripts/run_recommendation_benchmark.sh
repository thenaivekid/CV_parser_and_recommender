# !/bin/bash
# Run Recommendation Benchmark Script to evaluate retrieval performance single stage vs 2-stage(taking top k semantically similar jobs only)
PYTHONPATH=/workspaces/CV_parser_and_recommender:$PYTHONPATH python src/benchmark_retrieval.py --sample-size 5