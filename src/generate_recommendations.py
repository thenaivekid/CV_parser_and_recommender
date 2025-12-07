"""
Generate Job Recommendations for Candidates
Matches all candidates with all jobs and saves recommendations to database
"""
import sys
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.config import config
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.performance_monitor import get_monitor, set_monitor, PerformanceMonitor
from src.dashboard_generator import DashboardGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_candidate_worker(
    candidate_id: str,
    db_config: dict,
    engine_weights: Optional[Dict[str, float]],
    use_two_stage: bool,
    stage1_top_k: int,
    stage1_threshold: float,
    top_k: Optional[int],
    save_to_db: bool,
    output_dir: Optional[str],
    skip_existing: bool = True,
    redis_config: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Worker function to process a single candidate in parallel.
    Each worker gets its own database connection (thread-safe).
    
    Supports TWO-STAGE RETRIEVAL:
    - Stage 1: Fast vector similarity filtering (top-K jobs)
    - Stage 2: Full scoring (skills, experience, education) on filtered jobs only
    
    Supports REDIS CACHING for production API use:
    - When redis_config provided: Use Redis-cached embeddings + NumPy similarity
    - When redis_config is None: Fall back to pgvector similarity (backward compatible)
    
    Can SKIP existing recommendations to avoid redundant computation.
    
    Args:
        candidate_id: Candidate identifier
        db_config: Database configuration for creating connection
        engine_weights: Recommendation engine weights
        use_two_stage: Enable two-stage retrieval optimization
        stage1_top_k: Number of jobs to retrieve in Stage 1
        stage1_threshold: Minimum similarity threshold for Stage 1
        top_k: Number of top recommendations to save
        save_to_db: Whether to save to database
        output_dir: Output directory for JSON files
        skip_existing: Skip jobs that already have recommendations (default: True)
        redis_config: Redis configuration dict (if None, use pgvector only)
        
    Returns:
        Result dictionary with statistics
    """
    result = {
        'candidate_id': candidate_id,
        'success': False,
        'recommendations_count': 0,
        'saved_to_db': 0,
        'jobs_evaluated': 0,
        'jobs_skipped': 0,
        'retrieval_mode': 'redis-cached' if redis_config else ('two-stage' if use_two_stage else 'single-stage'),
        'skip_existing': skip_existing,
        'error': None
    }
    
    # Each worker creates its own DB connection (thread-safe)
    db_manager = None
    cache_manager = None
    try:
        db_manager = DatabaseManager(db_config)
        
        # Initialize cache manager if Redis is enabled
        if redis_config:
            from src.cache_manager import CachedRecommendationEngine
            cache_manager = CachedRecommendationEngine(db_manager, redis_config)
        
        engine = RecommendationEngine(
            weights=engine_weights,
            use_two_stage=use_two_stage,
            stage1_top_k=stage1_top_k,
            stage1_threshold=stage1_threshold,
            cache_manager=cache_manager
        )
        
        # Get candidate data
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            result['error'] = 'Candidate not found'
            return result
        
        candidate_name = candidate.get('name', 'Unknown')
        
        # CHOOSE RETRIEVAL STRATEGY: Redis-cached OR pgvector
        if cache_manager and redis_config.get('use_for_similarity', False):
            # REDIS PATH: Get CV embedding, use cached job embeddings + NumPy similarity
            cv_embedding = db_manager.get_candidate_embedding(candidate_id)
            if cv_embedding is None:
                result['error'] = 'Candidate embedding not found'
                return result
            
            # Use cache manager for similarity computation
            jobs_with_similarity = cache_manager.get_similar_jobs(
                cv_embedding=cv_embedding,
                top_k=stage1_top_k,
                similarity_threshold=stage1_threshold
            )
            result['jobs_skipped'] = 0  # Redis doesn't support skip_existing yet
        else:
            # PGVECTOR PATH: Traditional database similarity (backward compatible)
            if skip_existing:
                # OPTIMIZED: Get only jobs WITHOUT existing recommendations
                jobs_with_similarity = db_manager.get_jobs_without_recommendations(
                    candidate_id=candidate_id,
                    use_two_stage=use_two_stage,
                    stage1_top_k=stage1_top_k,
                    stage1_threshold=stage1_threshold
                )
                # Calculate skipped count
                if use_two_stage:
                    total_retrieved = min(stage1_top_k, db_manager.get_job_count())
                else:
                    total_retrieved = db_manager.get_job_count()
                result['jobs_skipped'] = total_retrieved - len(jobs_with_similarity)
            else:
                # STANDARD: Get all jobs (will overwrite existing)
                if use_two_stage:
                    # TWO-STAGE: Stage 1 - Get only top-K most similar jobs (FAST)
                    jobs_with_similarity = db_manager.get_top_k_jobs_by_similarity(
                        candidate_id=candidate_id,
                        top_k=stage1_top_k,
                        similarity_threshold=stage1_threshold
                    )
                else:
                    # SINGLE-STAGE: Get ALL jobs with similarity (SLOWER for large datasets)
                    jobs_with_similarity = db_manager.get_all_jobs_with_similarity_for_candidate(
                        candidate_id
                    )
                result['jobs_skipped'] = 0
        
        result['jobs_evaluated'] = len(jobs_with_similarity)
        
        if not jobs_with_similarity:
            if skip_existing and result['jobs_skipped'] > 0:
                # All jobs already have recommendations
                result['success'] = True
                result['error'] = None
                logger.info(
                    f"✓ [SKIP] {candidate_id} ({candidate_name}): "
                    f"All {result['jobs_skipped']} jobs already have recommendations"
                )
            else:
                result['error'] = 'No jobs with embeddings found'
            return result
        
        logger.debug(
            f"Candidate {candidate_id} - Processing {len(jobs_with_similarity)} jobs, "
            f"Skipped {result['jobs_skipped']} existing recommendations"
        )
        
        # Stage 2: Full scoring (skills, experience, education) on filtered jobs
        recommendations = engine.rank_jobs_for_candidate(
            candidate=candidate,
            jobs_with_similarity=jobs_with_similarity,
            top_k=top_k
        )
        
        result['recommendations_count'] = len(recommendations['recommendations'])
        
        # Save to database using BATCH insert (efficient)
        if save_to_db and recommendations['recommendations']:
            batch_recs = [
                {
                    'candidate_id': candidate_id,
                    'job_id': rec['job_id'],
                    'match_score': rec['match_score'],
                    'skills_match': rec['matching_factors']['skills_match'],
                    'experience_match': rec['matching_factors']['experience_match'],
                    'education_match': rec['matching_factors']['education_match'],
                    'semantic_similarity': rec['matching_factors']['semantic_similarity'],
                    'matched_skills': rec['matched_skills'],
                    'missing_skills': rec['missing_skills'],
                    'explanation': rec['explanation']
                }
                for rec in recommendations['recommendations']
            ]
            
            saved_count = db_manager.save_recommendations_batch(batch_recs)
            result['saved_to_db'] = saved_count
        
        # Save to file if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"{candidate_id}_recommendations.json"
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
        
        result['success'] = True
        
        if result['jobs_skipped'] > 0:
            logger.info(
                f"✓ [{result['retrieval_mode']}] Processed {candidate_id} ({candidate_name}): "
                f"{result['recommendations_count']} new recommendations, "
                f"{result['jobs_skipped']} skipped (already exist)"
            )
        else:
            logger.info(
                f"✓ [{result['retrieval_mode']}] Processed {candidate_id} ({candidate_name}): "
                f"{result['recommendations_count']} recommendations from {result['jobs_evaluated']} jobs evaluated"
            )
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"✗ Error processing {candidate_id}: {e}")
    
    finally:
        if db_manager:
            db_manager.close()
    
    return result


def generate_all_recommendations(
    db_manager: DatabaseManager,
    engine: RecommendationEngine,
    top_k: Optional[int] = None,
    save_to_db: bool = True,
    output_dir: Optional[str] = None,
    max_workers: int = 10,
    use_two_stage: bool = True,
    stage1_top_k: int = 50,
    stage1_threshold: float = 0.3,
    skip_existing: bool = True,
    redis_config: Optional[dict] = None
) -> dict:
    """
    Generate recommendations for all candidates using OPTIMIZED PARALLEL approach:
    - Batch read candidate IDs from database
    - Process multiple candidates in parallel using ThreadPoolExecutor
    - Each worker has its own database connection (thread-safe)
    - Use PostgreSQL pgvector OR Redis-cached embeddings for similarity computation
    - Batch save to database (efficient)
    - TWO-STAGE RETRIEVAL: Stage 1 filters, Stage 2 full scoring
    - SKIP EXISTING: Avoid recalculating existing recommendations
    
    Args:
        db_manager: Database manager instance (only for initial queries)
        engine: Recommendation engine instance (for getting weights)
        top_k: Number of top recommendations per candidate (None = all)
        save_to_db: Whether to save recommendations to database
        output_dir: Directory to save JSON output files (optional)
        max_workers: Number of parallel worker threads (default: 10)
        use_two_stage: Enable two-stage retrieval (default: True)
        stage1_top_k: Number of jobs to filter in Stage 1 (default: 50)
        stage1_threshold: Minimum similarity for Stage 1 (default: 0.3)
        skip_existing: Skip jobs with existing recommendations (default: True)
        redis_config: Redis configuration dict (if None, use pgvector only)
        
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 80)
    if redis_config and redis_config.get('use_for_similarity', False):
        logger.info("Starting PARALLEL REDIS-CACHED recommendation generation")
        logger.info(f"Using Redis for similarity computation (NumPy + cached embeddings)")
    else:
        logger.info(f"Starting PARALLEL {'TWO-STAGE' if use_two_stage else 'SINGLE-STAGE'} recommendation generation")
        if use_two_stage:
            logger.info(f"Stage 1: Filter top-{stage1_top_k} jobs (threshold >= {stage1_threshold})")
            logger.info(f"Stage 2: Full scoring on filtered jobs only")
        logger.info("Using PostgreSQL pgvector for similarity computation")
    logger.info(f"Using {max_workers} parallel workers")
    logger.info(f"Skip existing: {'YES - Only compute new recommendations' if skip_existing else 'NO - Recalculate all'}")
    logger.info("=" * 80)
    
    # Get candidate IDs only (memory efficient - no full data)
    logger.info("Retrieving candidate IDs...")
    candidate_ids = db_manager.get_all_candidate_ids()
    logger.info(f"Found {len(candidate_ids)} candidates")
    
    # Get job count for stats
    job_count = db_manager.get_job_count()
    logger.info(f"Found {job_count} jobs")
    
    if not candidate_ids:
        logger.error("No candidates found in database")
        return {'error': 'No candidates found'}
    
    if job_count == 0:
        logger.error("No jobs found in database")
        return {'error': 'No jobs found'}
    
    # Start session to track THIS run
    monitor = get_monitor()
    jobs_per_candidate = stage1_top_k if use_two_stage else job_count
    total_pairs = len(candidate_ids) * jobs_per_candidate
    session_metadata = {
        'max_workers': max_workers,
        'top_k': top_k,
        'total_candidates': len(candidate_ids),
        'total_jobs': job_count,
        'use_two_stage': use_two_stage,
        'stage1_top_k': stage1_top_k,
        'stage1_threshold': stage1_threshold,
        'skip_existing': skip_existing,
        'estimated_pairs_evaluated': total_pairs
    }
    monitor.start_session('recommendation_generation', metadata=session_metadata)
    
    # Initialize statistics
    stats = {
        'total_candidates': len(candidate_ids),
        'total_jobs': job_count,
        'processed_candidates': 0,
        'successful_candidates': 0,
        'failed_candidates': 0,
        'total_recommendations': 0,
        'total_jobs_evaluated': 0,
        'total_jobs_skipped': 0,
        'saved_to_db': 0,
        'max_workers': max_workers,
        'use_two_stage': use_two_stage,
        'skip_existing': skip_existing,
        'stage1_top_k': stage1_top_k if use_two_stage else 'N/A',
        'elapsed_time_seconds': 0
    }
    
    # Get DB config and engine weights for workers
    db_config = db_manager.db_config
    engine_weights = engine.weights
    
    logger.info(f"\nStarting parallel processing with {max_workers} workers...")
    logger.info("-" * 80)
    
    # Process candidates in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_candidate = {
            executor.submit(
                process_candidate_worker,
                candidate_id,
                db_config,
                engine_weights,
                use_two_stage,
                stage1_top_k,
                stage1_threshold,
                top_k,
                save_to_db,
                output_dir,
                skip_existing,
                redis_config
            ): candidate_id
            for candidate_id in candidate_ids
        }
        
        # Collect results as they complete
        start_time = time.time()
        for idx, future in enumerate(as_completed(future_to_candidate), 1):
            candidate_id = future_to_candidate[future]
            try:
                result = future.result()
                
                stats['processed_candidates'] += 1
                
                if result['success']:
                    stats['successful_candidates'] += 1
                    stats['total_recommendations'] += result['recommendations_count']
                    stats['total_jobs_evaluated'] += result.get('jobs_evaluated', 0)
                    stats['total_jobs_skipped'] += result.get('jobs_skipped', 0)
                    stats['saved_to_db'] += result['saved_to_db']
                else:
                    stats['failed_candidates'] += 1
                    logger.warning(
                        f"Failed to process {candidate_id}: {result.get('error', 'Unknown error')}"
                    )
                
                # Progress indicator and system snapshot every 10 candidates
                if stats['processed_candidates'] % 10 == 0:
                    elapsed = time.time() - start_time
                    # Record system snapshot
                    monitor = get_monitor()
                    monitor.record_system_snapshot(
                        active_workers=max_workers,
                        throughput_per_min=(stats['processed_candidates'] / elapsed) * 60
                    )
                    
                    avg_jobs = stats['total_jobs_evaluated'] / max(stats['processed_candidates'], 1)
                    avg_skipped = stats['total_jobs_skipped'] / max(stats['processed_candidates'], 1)
                    logger.info(
                        f"Progress: {stats['processed_candidates']}/{len(candidate_ids)} "
                        f"({stats['processed_candidates']/len(candidate_ids)*100:.1f}%) "
                        f"- Success: {stats['successful_candidates']}, "
                        f"Failed: {stats['failed_candidates']}, "
                        f"Avg jobs/candidate: {avg_jobs:.1f}, "
                        f"Avg skipped/candidate: {avg_skipped:.1f}"
                    )
                
            except Exception as e:
                stats['failed_candidates'] += 1
                logger.error(f"Exception processing {candidate_id}: {e}")
    
    stats['elapsed_time_seconds'] = time.time() - start_time
    
    # End session with actual processed counts
    monitor.end_session(
        items_processed=stats['total_jobs_evaluated'],  # Total CV×Job pairs actually evaluated
        items_success=stats['saved_to_db'],             # Successfully saved
        items_failed=stats['failed_candidates'] * (stage1_top_k if use_two_stage else job_count),
        items_skipped=stats['total_jobs_skipped']       # Skipped existing recommendations
    )
    
    logger.info("=" * 80)
    logger.info("PARALLEL RECOMMENDATION GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total candidates: {stats['total_candidates']}")
    logger.info(f"Total jobs in DB: {stats['total_jobs']}")
    logger.info(f"Retrieval mode: {'TWO-STAGE' if use_two_stage else 'SINGLE-STAGE'}")
    if use_two_stage:
        logger.info(f"  Stage 1 filter: top-{stage1_top_k} jobs per candidate")
        logger.info(f"  Avg jobs evaluated: {stats['total_jobs_evaluated']/max(stats['successful_candidates'], 1):.1f}")
    logger.info(f"Skip existing: {skip_existing}")
    if skip_existing:
        logger.info(f"  Total skipped (existing): {stats['total_jobs_skipped']}")
        logger.info(f"  Avg skipped per candidate: {stats['total_jobs_skipped']/max(stats['successful_candidates'], 1):.1f}")
        efficiency_pct = (stats['total_jobs_skipped'] / max(stats['total_jobs_evaluated'] + stats['total_jobs_skipped'], 1)) * 100
        logger.info(f"  Computation saved: {efficiency_pct:.1f}%")
    logger.info(f"Processed candidates: {stats['processed_candidates']}")
    logger.info(f"  ✓ Successful: {stats['successful_candidates']}")
    logger.info(f"  ✗ Failed: {stats['failed_candidates']}")
    logger.info(f"Total recommendations generated: {stats['total_recommendations']}")
    logger.info(f"Total jobs evaluated: {stats['total_jobs_evaluated']}")
    logger.info(f"Saved to database: {stats['saved_to_db']}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Elapsed time: {stats['elapsed_time_seconds']:.2f} seconds")
    logger.info(f"Throughput: {stats['successful_candidates']/stats['elapsed_time_seconds']:.2f} candidates/sec")
    logger.info("=" * 80)
    
    return stats


def generate_recommendations_for_candidate(
    db_manager: DatabaseManager,
    engine: RecommendationEngine,
    candidate_id: str,
    top_k: Optional[int] = 10,
    save_to_db: bool = True,
    output_file: Optional[str] = None
) -> dict:
    """
    Generate recommendations for a specific candidate using OPTIMIZED approach
    
    Args:
        db_manager: Database manager instance
        engine: Recommendation engine instance
        candidate_id: Candidate identifier
        top_k: Number of top recommendations to return
        save_to_db: Whether to save to database
        output_file: Optional output file path
        
    Returns:
        Recommendations dictionary
    """
    logger.info(f"Generating recommendations for candidate: {candidate_id}")
    logger.info("Using PostgreSQL pgvector for similarity computation")
    
    # Get candidate
    candidate = db_manager.get_candidate(candidate_id)
    if not candidate:
        logger.error(f"Candidate {candidate_id} not found")
        return {'error': 'Candidate not found'}
    
    # Get ALL jobs with pre-computed similarity from PostgreSQL
    logger.info("Fetching jobs with pre-computed similarities from database...")
    jobs_with_similarity = db_manager.get_all_jobs_with_similarity_for_candidate(
        candidate_id
    )
    
    if not jobs_with_similarity:
        logger.error(
            f"No jobs with embeddings found for candidate {candidate_id}"
        )
        return {'error': 'No jobs with embeddings found'}
    
    logger.info(f"Retrieved {len(jobs_with_similarity)} jobs with similarities")
    
    # Generate recommendations (semantic similarity already computed!)
    recommendations = engine.rank_jobs_for_candidate(
        candidate=candidate,
        jobs_with_similarity=jobs_with_similarity,
        top_k=top_k
    )
    
    logger.info(f"Generated {len(recommendations['recommendations'])} recommendations")
    
    # Save to database using batch insert
    if save_to_db and recommendations['recommendations']:
        # Prepare batch data
        batch_recs = [
            {
                'candidate_id': candidate_id,
                'job_id': rec['job_id'],
                'match_score': rec['match_score'],
                'skills_match': rec['matching_factors']['skills_match'],
                'experience_match': rec['matching_factors']['experience_match'],
                'education_match': rec['matching_factors']['education_match'],
                'semantic_similarity': rec['matching_factors']['semantic_similarity'],
                'matched_skills': rec['matched_skills'],
                'missing_skills': rec['missing_skills'],
                'explanation': rec['explanation']
            }
            for rec in recommendations['recommendations']
        ]
        
        saved_count = db_manager.save_recommendations_batch(batch_recs)
        logger.info(f"Batch saved {saved_count} recommendations to database")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        logger.info(f"Saved recommendations to {output_file}")
    
    return recommendations


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate job recommendations for candidates'
    )
    parser.add_argument(
        '--candidate-id',
        help='Generate recommendations for specific candidate only'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Number of top recommendations per candidate (default: all)'
    )
    parser.add_argument(
        '--no-save-db',
        action='store_true',
        help='Do not save recommendations to database'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save JSON output files'
    )
    parser.add_argument(
        '--output-file',
        help='Output file for single candidate recommendations'
    )
    parser.add_argument(
        '--config',
        default='configurations/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--weights',
        help='Custom weights as JSON: {"skills":0.4,"experience":0.3,"education":0.1,"semantic":0.2}'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel worker threads for processing candidates (default: 10)'
    )
    parser.add_argument(
        '--single-stage',
        action='store_true',
        help='Use single-stage retrieval (evaluate ALL jobs). Default is two-stage for efficiency.'
    )
    parser.add_argument(
        '--stage1-k',
        type=int,
        default=50,
        help='Stage 1: Number of top jobs to filter (default: 50). Only used in two-stage mode.'
    )
    parser.add_argument(
        '--stage1-threshold',
        type=float,
        default=0.3,
        help='Stage 1: Minimum similarity threshold 0.0-1.0 (default: 0.3). Only used in two-stage mode.'
    )
    parser.add_argument(
        '--force-recalculate',
        action='store_true',
        help='Force recalculation of existing recommendations (default: skip existing)'
    )
    
    args = parser.parse_args()

    # Parse custom weights if provided
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
            logger.info(f"Using custom weights: {weights}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid weights JSON: {e}")
            sys.exit(1)
    
    # Two-stage retrieval settings
    use_two_stage = not args.single_stage
    stage1_top_k = args.stage1_k
    stage1_threshold = args.stage1_threshold
    skip_existing = not args.force_recalculate
    
    # Initialize components
    try:
        db_manager = DatabaseManager(config.database)
        
        # Initialize Redis cache manager if enabled
        cache_manager = None
        redis_config = None
        if hasattr(config, 'redis') and config.redis.get('enabled', False):
            logger.info("Redis caching enabled")
            try:
                from src.cache_manager import CachedRecommendationEngine
                redis_config = config.redis
                cache_manager = CachedRecommendationEngine(db_manager, redis_config)
                logger.info(f"✓ Connected to Redis at {redis_config['host']}:{redis_config['port']}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to pgvector.")
                redis_config = None
                cache_manager = None
        else:
            logger.info("Redis caching disabled (using pgvector only)")
        
        engine = RecommendationEngine(
            weights=weights,
            use_two_stage=use_two_stage,
            stage1_top_k=stage1_top_k,
            stage1_threshold=stage1_threshold,
            cache_manager=cache_manager
        )
        
        # Initialize performance monitor
        monitor = PerformanceMonitor(db_manager)
        set_monitor(monitor)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Generate recommendations
    try:
        if args.candidate_id:
            # Single candidate
            recommendations = generate_recommendations_for_candidate(
                db_manager=db_manager,
                engine=engine,
                candidate_id=args.candidate_id,
                top_k=args.top_k,
                save_to_db=not args.no_save_db,
                output_file=args.output_file
            )
            
            # Print top 3 recommendations
            if 'recommendations' in recommendations:
                print("\n" + "=" * 80)
                print(f"TOP RECOMMENDATIONS FOR: {recommendations['candidate_name']}")
                print("=" * 80)
                for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                    print(f"\n{i}. {rec['job_title']} at {rec['company']}")
                    print(f"   Match Score: {rec['match_score']:.2%}")
                    print(f"   Skills: {rec['matching_factors']['skills_match']:.2%} | "
                          f"Experience: {rec['matching_factors']['experience_match']:.2%} | "
                          f"Education: {rec['matching_factors']['education_match']:.2%} | "
                          f"Semantic: {rec['matching_factors']['semantic_similarity']:.2%}")
                    print(f"   {rec['explanation']}")
                print("\n" + "=" * 80)
        else:
            # All candidates (parallel processing with two-stage option)
            stats = generate_all_recommendations(
                db_manager=db_manager,
                engine=engine,
                top_k=args.top_k,
                save_to_db=not args.no_save_db,
                output_dir=args.output_dir,
                max_workers=args.workers,
                use_two_stage=use_two_stage,
                stage1_top_k=stage1_top_k,
                stage1_threshold=stage1_threshold,
                skip_existing=skip_existing,
                redis_config=redis_config
            )
            
            if 'error' in stats:
                logger.error(f"Recommendation generation failed: {stats['error']}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during recommendation generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        db_manager.close()
        
    logger.info("Done!")


if __name__ == '__main__':
    main()
