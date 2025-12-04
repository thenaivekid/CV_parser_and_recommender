"""
Benchmarking Script for Two-Stage vs Single-Stage Retrieval
Compares performance, latency, and throughput of both approaches
"""
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import statistics

from src.config import config
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.performance_monitor import get_monitor, set_monitor, PerformanceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_single_candidate(
    db_manager: DatabaseManager,
    candidate_id: str,
    use_two_stage: bool,
    stage1_top_k: int = 50,
    stage1_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Benchmark recommendation generation for a single candidate
    
    Returns:
        Dictionary with timing and statistics
    """
    result = {
        'candidate_id': candidate_id,
        'mode': 'two-stage' if use_two_stage else 'single-stage',
        'success': False,
        'error': None
    }
    
    try:
        # Get candidate
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            result['error'] = 'Candidate not found'
            return result
        
        # Initialize engine
        engine = RecommendationEngine(
            use_two_stage=use_two_stage,
            stage1_top_k=stage1_top_k,
            stage1_threshold=stage1_threshold
        )
        
        # STAGE 1: Vector Retrieval (timed separately)
        stage1_start = time.time()
        if use_two_stage:
            jobs = db_manager.get_top_k_jobs_by_similarity(
                candidate_id, stage1_top_k, stage1_threshold
            )
        else:
            jobs = db_manager.get_all_jobs_with_similarity_for_candidate(candidate_id)
        stage1_time = time.time() - stage1_start
        
        result['stage1_retrieval_time_ms'] = stage1_time * 1000
        result['jobs_retrieved'] = len(jobs)
        
        if not jobs:
            result['error'] = 'No jobs found'
            return result
        
        # STAGE 2: Full Scoring (timed separately)
        stage2_start = time.time()
        recommendations = engine.rank_jobs_for_candidate(
            candidate=candidate,
            jobs_with_similarity=jobs,
            top_k=None  # Get all for benchmarking
        )
        stage2_time = time.time() - stage2_start
        
        result['stage2_scoring_time_ms'] = stage2_time * 1000
        result['total_time_ms'] = (stage1_time + stage2_time) * 1000
        result['recommendations_count'] = len(recommendations['recommendations'])
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error benchmarking {candidate_id}: {e}")
    
    return result


def run_benchmark(
    db_manager: DatabaseManager,
    sample_size: int = 10,
    stage1_top_k: int = 50,
    stage1_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing single-stage vs two-stage retrieval
    
    Args:
        db_manager: Database manager
        sample_size: Number of candidates to benchmark (default: 10)
        stage1_top_k: Top-K for two-stage filtering
        stage1_threshold: Similarity threshold for two-stage
    
    Returns:
        Benchmark results with statistics
    """
    logger.info("=" * 80)
    logger.info("RETRIEVAL BENCHMARKING: Single-Stage vs Two-Stage")
    logger.info("=" * 80)
    
    # Get sample candidates
    all_candidate_ids = db_manager.get_all_candidate_ids()
    sample_candidates = all_candidate_ids[:min(sample_size, len(all_candidate_ids))]
    job_count = db_manager.get_job_count()
    
    logger.info(f"Sample size: {len(sample_candidates)} candidates")
    logger.info(f"Total jobs: {job_count}")
    logger.info(f"Two-stage config: top_k={stage1_top_k}, threshold={stage1_threshold}")
    logger.info("-" * 80)
    
    # Benchmark SINGLE-STAGE
    logger.info("\n🔍 BENCHMARKING SINGLE-STAGE RETRIEVAL...")
    single_stage_results = []
    single_stage_start = time.time()
    
    for idx, candidate_id in enumerate(sample_candidates, 1):
        logger.info(f"  [{idx}/{len(sample_candidates)}] Processing {candidate_id}...")
        result = benchmark_single_candidate(
            db_manager, candidate_id, use_two_stage=False
        )
        single_stage_results.append(result)
    
    single_stage_total = time.time() - single_stage_start
    
    # Benchmark TWO-STAGE
    logger.info("\n🚀 BENCHMARKING TWO-STAGE RETRIEVAL...")
    two_stage_results = []
    two_stage_start = time.time()
    
    for idx, candidate_id in enumerate(sample_candidates, 1):
        logger.info(f"  [{idx}/{len(sample_candidates)}] Processing {candidate_id}...")
        result = benchmark_single_candidate(
            db_manager, candidate_id, use_two_stage=True,
            stage1_top_k=stage1_top_k, stage1_threshold=stage1_threshold
        )
        two_stage_results.append(result)
    
    two_stage_total = time.time() - two_stage_start
    
    # Calculate statistics
    def calc_stats(results: List[Dict]) -> Dict[str, float]:
        successful = [r for r in results if r['success']]
        if not successful:
            return {}
        
        total_times = [r['total_time_ms'] for r in successful]
        stage1_times = [r['stage1_retrieval_time_ms'] for r in successful]
        stage2_times = [r['stage2_scoring_time_ms'] for r in successful]
        jobs_retrieved = [r['jobs_retrieved'] for r in successful]
        
        return {
            'avg_total_ms': statistics.mean(total_times),
            'median_total_ms': statistics.median(total_times),
            'min_total_ms': min(total_times),
            'max_total_ms': max(total_times),
            'std_total_ms': statistics.stdev(total_times) if len(total_times) > 1 else 0,
            'avg_stage1_ms': statistics.mean(stage1_times),
            'avg_stage2_ms': statistics.mean(stage2_times),
            'avg_jobs_retrieved': statistics.mean(jobs_retrieved),
            'successful_count': len(successful),
            'failed_count': len(results) - len(successful)
        }
    
    single_stats = calc_stats(single_stage_results)
    two_stage_stats = calc_stats(two_stage_results)
    
    # Calculate improvements
    if single_stats and two_stage_stats:
        speedup = single_stats['avg_total_ms'] / two_stage_stats['avg_total_ms']
        time_saved_pct = (1 - two_stage_stats['avg_total_ms'] / single_stats['avg_total_ms']) * 100
        jobs_reduction_pct = (1 - two_stage_stats['avg_jobs_retrieved'] / single_stats['avg_jobs_retrieved']) * 100
        
        # Throughput calculations
        single_throughput = len(sample_candidates) / single_stage_total * 60  # candidates/min
        two_stage_throughput = len(sample_candidates) / two_stage_total * 60
    else:
        speedup = 0
        time_saved_pct = 0
        jobs_reduction_pct = 0
        single_throughput = 0
        two_stage_throughput = 0
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)
    
    logger.info("\n📊 SINGLE-STAGE RETRIEVAL:")
    logger.info(f"  Total time: {single_stage_total:.2f}s")
    logger.info(f"  Avg per candidate: {single_stats.get('avg_total_ms', 0):.2f}ms")
    logger.info(f"  Median: {single_stats.get('median_total_ms', 0):.2f}ms")
    logger.info(f"  Range: {single_stats.get('min_total_ms', 0):.2f}ms - {single_stats.get('max_total_ms', 0):.2f}ms")
    logger.info(f"  Std dev: {single_stats.get('std_total_ms', 0):.2f}ms")
    logger.info(f"  Avg Stage 1 (DB query): {single_stats.get('avg_stage1_ms', 0):.2f}ms")
    logger.info(f"  Avg Stage 2 (scoring): {single_stats.get('avg_stage2_ms', 0):.2f}ms")
    logger.info(f"  Avg jobs evaluated: {single_stats.get('avg_jobs_retrieved', 0):.1f}")
    logger.info(f"  Throughput: {single_throughput:.2f} candidates/min")
    logger.info(f"  Success rate: {single_stats.get('successful_count', 0)}/{len(sample_candidates)}")
    
    logger.info("\n🚀 TWO-STAGE RETRIEVAL:")
    logger.info(f"  Total time: {two_stage_total:.2f}s")
    logger.info(f"  Avg per candidate: {two_stage_stats.get('avg_total_ms', 0):.2f}ms")
    logger.info(f"  Median: {two_stage_stats.get('median_total_ms', 0):.2f}ms")
    logger.info(f"  Range: {two_stage_stats.get('min_total_ms', 0):.2f}ms - {two_stage_stats.get('max_total_ms', 0):.2f}ms")
    logger.info(f"  Std dev: {two_stage_stats.get('std_total_ms', 0):.2f}ms")
    logger.info(f"  Avg Stage 1 (DB query): {two_stage_stats.get('avg_stage1_ms', 0):.2f}ms")
    logger.info(f"  Avg Stage 2 (scoring): {two_stage_stats.get('avg_stage2_ms', 0):.2f}ms")
    logger.info(f"  Avg jobs evaluated: {two_stage_stats.get('avg_jobs_retrieved', 0):.1f}")
    logger.info(f"  Throughput: {two_stage_throughput:.2f} candidates/min")
    logger.info(f"  Success rate: {two_stage_stats.get('successful_count', 0)}/{len(sample_candidates)}")
    
    logger.info("\n✨ PERFORMANCE IMPROVEMENT:")
    logger.info(f"  Speedup: {speedup:.2f}x faster")
    logger.info(f"  Time saved: {time_saved_pct:.1f}%")
    logger.info(f"  Jobs reduction: {jobs_reduction_pct:.1f}%")
    logger.info(f"  Throughput gain: {two_stage_throughput - single_throughput:.2f} candidates/min")
    
    logger.info("\n📈 SCALABILITY PROJECTION (for 1000 CVs × 500 jobs):")
    single_projected = (single_stats.get('avg_total_ms', 0) / 1000) * 1000  # seconds
    two_stage_projected = (two_stage_stats.get('avg_total_ms', 0) / 1000) * 1000
    logger.info(f"  Single-stage: ~{single_projected:.1f}s ({single_projected/60:.1f} min)")
    logger.info(f"  Two-stage: ~{two_stage_projected:.1f}s ({two_stage_projected/60:.1f} min)")
    logger.info(f"  Time saved at scale: ~{(single_projected - two_stage_projected)/60:.1f} min")
    
    logger.info("\n" + "=" * 80)
    
    # Save detailed results
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sample_size': len(sample_candidates),
            'total_jobs': job_count,
            'stage1_top_k': stage1_top_k,
            'stage1_threshold': stage1_threshold
        },
        'single_stage': {
            'total_time_seconds': single_stage_total,
            'stats': single_stats,
            'throughput_per_min': single_throughput,
            'detailed_results': single_stage_results
        },
        'two_stage': {
            'total_time_seconds': two_stage_total,
            'stats': two_stage_stats,
            'throughput_per_min': two_stage_throughput,
            'detailed_results': two_stage_results
        },
        'improvement': {
            'speedup': speedup,
            'time_saved_percent': time_saved_pct,
            'jobs_reduction_percent': jobs_reduction_pct,
            'throughput_gain': two_stage_throughput - single_throughput
        }
    }
    
    # Save to file
    output_dir = Path('data/performance_reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {output_file}")
    
    return benchmark_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark single-stage vs two-stage retrieval'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10,
        help='Number of candidates to benchmark (default: 10)'
    )
    parser.add_argument(
        '--stage1-k',
        type=int,
        default=10,
        help='Stage 1 top-K filter (default: 10)'
    )
    parser.add_argument(
        '--stage1-threshold',
        type=float,
        default=0.3,
        help='Stage 1 similarity threshold (default: 0.3)'
    )
    parser.add_argument(
        '--config',
        default='configurations/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize database
    try:
        db_manager = DatabaseManager(config.database)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Run benchmark
    try:
        results = run_benchmark(
            db_manager=db_manager,
            sample_size=args.sample_size,
            stage1_top_k=args.stage1_k,
            stage1_threshold=args.stage1_threshold
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db_manager.close()
    
    logger.info("\n✅ Benchmark completed successfully!")
