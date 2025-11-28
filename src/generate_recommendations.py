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
    top_k: Optional[int],
    save_to_db: bool,
    output_dir: Optional[str]
) -> Dict[str, Any]:
    """
    Worker function to process a single candidate in parallel.
    Each worker gets its own database connection (thread-safe).
    
    Args:
        candidate_id: Candidate identifier
        db_config: Database configuration for creating connection
        engine_weights: Recommendation engine weights
        top_k: Number of top recommendations
        save_to_db: Whether to save to database
        output_dir: Output directory for JSON files
        
    Returns:
        Result dictionary with statistics
    """
    result = {
        'candidate_id': candidate_id,
        'success': False,
        'recommendations_count': 0,
        'saved_to_db': 0,
        'error': None
    }
    
    # Each worker creates its own DB connection (thread-safe)
    db_manager = None
    try:
        db_manager = DatabaseManager(db_config)
        engine = RecommendationEngine(weights=engine_weights)
        
        # Get candidate data
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            result['error'] = 'Candidate not found'
            return result
        
        candidate_name = candidate.get('name', 'Unknown')
        
        # Get ALL jobs with pre-computed similarity from PostgreSQL
        # Uses pgvector's optimized C code and IVFFLAT index
        jobs_with_similarity = db_manager.get_all_jobs_with_similarity_for_candidate(
            candidate_id
        )
        
        if not jobs_with_similarity:
            result['error'] = 'No jobs with embeddings found'
            return result
        
        # Generate recommendations (Python does skills/exp/edu scoring only)
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
        logger.info(
            f"✓ Processed {candidate_id} ({candidate_name}): "
            f"{result['recommendations_count']} recommendations"
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
    max_workers: int = 10
) -> dict:
    """
    Generate recommendations for all candidates using OPTIMIZED PARALLEL approach:
    - Batch read candidate IDs from database
    - Process multiple candidates in parallel using ThreadPoolExecutor
    - Each worker has its own database connection (thread-safe)
    - Use PostgreSQL pgvector for similarity computation (fast)
    - Batch save to database (efficient)
    
    Args:
        db_manager: Database manager instance (only for initial queries)
        engine: Recommendation engine instance (for getting weights)
        top_k: Number of top recommendations per candidate (None = all)
        save_to_db: Whether to save recommendations to database
        output_dir: Directory to save JSON output files (optional)
        max_workers: Number of parallel worker threads (default: 10)
        
    Returns:
        Statistics dictionary
    """
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("Starting PARALLEL OPTIMIZED recommendation generation")
    logger.info(f"Using {max_workers} parallel workers")
    logger.info("Using PostgreSQL pgvector for similarity computation")
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
    
    # Initialize statistics
    stats = {
        'total_candidates': len(candidate_ids),
        'total_jobs': job_count,
        'processed_candidates': 0,
        'successful_candidates': 0,
        'failed_candidates': 0,
        'total_recommendations': 0,
        'saved_to_db': 0,
        'max_workers': max_workers,
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
                top_k,
                save_to_db,
                output_dir
            ): candidate_id
            for candidate_id in candidate_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_candidate):
            candidate_id = future_to_candidate[future]
            try:
                result = future.result()
                
                stats['processed_candidates'] += 1
                
                if result['success']:
                    stats['successful_candidates'] += 1
                    stats['total_recommendations'] += result['recommendations_count']
                    stats['saved_to_db'] += result['saved_to_db']
                else:
                    stats['failed_candidates'] += 1
                    logger.warning(
                        f"Failed to process {candidate_id}: {result.get('error', 'Unknown error')}"
                    )
                
                # Progress indicator
                if stats['processed_candidates'] % 10 == 0:
                    logger.info(
                        f"Progress: {stats['processed_candidates']}/{len(candidate_ids)} "
                        f"({stats['processed_candidates']/len(candidate_ids)*100:.1f}%) "
                        f"- Success: {stats['successful_candidates']}, "
                        f"Failed: {stats['failed_candidates']}"
                    )
                
            except Exception as e:
                stats['failed_candidates'] += 1
                logger.error(f"Exception processing {candidate_id}: {e}")
    
    elapsed_time = time.time() - start_time
    stats['elapsed_time_seconds'] = round(elapsed_time, 2)
    
    logger.info("=" * 80)
    logger.info("PARALLEL RECOMMENDATION GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total candidates: {stats['total_candidates']}")
    logger.info(f"Total jobs: {stats['total_jobs']}")
    logger.info(f"Processed candidates: {stats['processed_candidates']}")
    logger.info(f"  ✓ Successful: {stats['successful_candidates']}")
    logger.info(f"  ✗ Failed: {stats['failed_candidates']}")
    logger.info(f"Total recommendations generated: {stats['total_recommendations']}")
    logger.info(f"Saved to database: {stats['saved_to_db']}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Total time: {stats['elapsed_time_seconds']} seconds")
    logger.info(f"Average time per candidate: {stats['elapsed_time_seconds']/max(stats['processed_candidates'], 1):.2f} seconds")
    logger.info(f"Throughput: {stats['processed_candidates']/elapsed_time:.2f} candidates/second")
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
    
    # Initialize components
    try:
        db_manager = DatabaseManager(config.database)
        engine = RecommendationEngine(weights=weights)
        
        # Initialize performance monitor
        monitor = PerformanceMonitor(db_manager)
        set_monitor(monitor)
        
        # Set dataset context
        num_cvs = db_manager.get_candidate_count()
        num_jobs = db_manager.get_job_count()
        monitor.set_dataset_context(num_cvs=num_cvs, num_jobs=num_jobs)
        
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
            # All candidates (parallel processing)
            stats = generate_all_recommendations(
                db_manager=db_manager,
                engine=engine,
                top_k=args.top_k,
                save_to_db=not args.no_save_db,
                output_dir=args.output_dir,
                max_workers=args.workers
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
        
        # Generate performance dashboard
        logger.info("\n📊 Generating performance dashboard...")
        try:
            output_dir_path = Path("data/performance_reports")
            dashboard_gen = DashboardGenerator()
            report = dashboard_gen.generate_report(output_dir_path)
            logger.info(f"✓ Performance dashboard saved to {output_dir_path}")
            logger.info(f"  View: {output_dir_path}/performance_dashboard_*.html")
        except Exception as e:
            logger.warning(f"Could not generate dashboard: {e}")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
