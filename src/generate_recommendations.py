"""
Generate Job Recommendations for Candidates
Matches all candidates with all jobs and saves recommendations to database
"""
import sys
import logging
import json
from pathlib import Path
from typing import Optional

from src.config import config
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _ensure_list(embedding):
    """
    Ensure embedding is a list of floats
    
    Args:
        embedding: Embedding as string or list
        
    Returns:
        List of floats
    """
    if isinstance(embedding, str):
        # Remove brackets and split by comma
        embedding = embedding.strip('[]')
        return [float(x) for x in embedding.split(',')]
    elif isinstance(embedding, list):
        return [float(x) for x in embedding]
    return embedding


def generate_all_recommendations(
    db_manager: DatabaseManager,
    engine: RecommendationEngine,
    top_k: Optional[int] = None,
    save_to_db: bool = True,
    output_dir: Optional[str] = None
) -> dict:
    """
    Generate recommendations for all candidates
    
    Args:
        db_manager: Database manager instance
        engine: Recommendation engine instance
        top_k: Number of top recommendations per candidate (None = all)
        save_to_db: Whether to save recommendations to database
        output_dir: Directory to save JSON output files (optional)
        
    Returns:
        Statistics dictionary
    """
    logger.info("Starting recommendation generation...")
    
    # Retrieve all candidates and jobs
    logger.info("Retrieving candidates from database...")
    candidates = db_manager.get_all_candidates() # TODO: make it more memory efficient for large datasets
    logger.info(f"Found {len(candidates)} candidates")
    
    logger.info("Retrieving jobs from database...")
    jobs = db_manager.get_all_jobs()  # TODO: make it more memory efficient for large datasets
    logger.info(f"Found {len(jobs)} jobs")
    
    if not candidates:
        logger.error("No candidates found in database")
        return {'error': 'No candidates found'}
    
    if not jobs:
        logger.error("No jobs found in database")
        return {'error': 'No jobs found'}
    
    # Retrieve all embeddings
    logger.info("Retrieving embeddings...")
    candidate_embeddings = db_manager.get_all_candidate_embeddings() # TODO: make it more memory efficient for large datasets
    job_embeddings = db_manager.get_all_job_embeddings()
    
    # Convert embeddings to lists if they're strings
    candidate_embeddings = {
        cid: _ensure_list(emb) for cid, emb in candidate_embeddings.items()
    }
    job_embeddings = {
        jid: _ensure_list(emb) for jid, emb in job_embeddings.items()
    }
    
    logger.info(f"Found {len(candidate_embeddings)} candidate embeddings") # TODO: make it more memory efficient for large datasets
    logger.info(f"Found {len(job_embeddings)} job embeddings")
    
    # Generate recommendations for each candidate
    stats = {
        'total_candidates': len(candidates),
        'total_jobs': len(jobs),
        'processed_candidates': 0,
        'total_recommendations': 0,
        'saved_to_db': 0,
        'saved_to_files': 0
    }
    
    all_recommendations = []
    
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        candidate_name = candidate.get('name', 'Unknown')
        
        logger.info(f"Processing candidate: {candidate_id} ({candidate_name})")
        
        # Get candidate embedding
        candidate_embedding = candidate_embeddings.get(candidate_id)
        
        if not candidate_embedding:
            logger.warning(f"No embedding found for candidate {candidate_id}, skipping")
            continue
        
        # Generate recommendations
        try:
            recommendations = engine.rank_jobs_for_candidate(
                candidate=candidate,
                candidate_embedding=candidate_embedding,
                jobs=jobs,
                job_embeddings=job_embeddings,
                top_k=top_k
            )
            
            all_recommendations.append(recommendations)
            stats['processed_candidates'] += 1
            stats['total_recommendations'] += len(recommendations['recommendations'])
            
            logger.info(
                f"Generated {len(recommendations['recommendations'])} recommendations "
                f"for {candidate_id}"
            )
            
            # Save to database
            if save_to_db:
                saved_count = 0
                for rec in recommendations['recommendations']:
                    success = db_manager.save_recommendation(
                        candidate_id=candidate_id,
                        job_id=rec['job_id'],
                        match_score=rec['match_score'],
                        skills_match=rec['matching_factors']['skills_match'],
                        experience_match=rec['matching_factors']['experience_match'],
                        education_match=rec['matching_factors']['education_match'],
                        semantic_similarity=rec['matching_factors']['semantic_similarity'],
                        matched_skills=rec['matched_skills'],
                        missing_skills=rec['missing_skills'],
                        explanation=rec['explanation']
                    )
                    if success:
                        saved_count += 1
                
                stats['saved_to_db'] += saved_count
                logger.info(f"Saved {saved_count} recommendations to database")
            
            # Save to file if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                output_file = output_path / f"{candidate_id}_recommendations.json"
                with open(output_file, 'w') as f:
                    json.dump(recommendations, f, indent=2)
                
                stats['saved_to_files'] += 1
                logger.info(f"Saved recommendations to {output_file}")
        
        except Exception as e:
            logger.error(f"Error processing candidate {candidate_id}: {e}")
            continue
    
    logger.info("=" * 80)
    logger.info("RECOMMENDATION GENERATION COMPLETE")
    logger.info(f"Total candidates: {stats['total_candidates']}")
    logger.info(f"Total jobs: {stats['total_jobs']}")
    logger.info(f"Processed candidates: {stats['processed_candidates']}")
    logger.info(f"Total recommendations generated: {stats['total_recommendations']}")
    logger.info(f"Saved to database: {stats['saved_to_db']}")
    logger.info(f"Saved to files: {stats['saved_to_files']}")
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
    Generate recommendations for a specific candidate
    
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
    
    # Get candidate
    candidate = db_manager.get_candidate(candidate_id)
    if not candidate:
        logger.error(f"Candidate {candidate_id} not found")
        return {'error': 'Candidate not found'}
    
    # Get candidate embedding
    candidate_embedding = db_manager.get_candidate_embedding(candidate_id)
    if not candidate_embedding:
        logger.error(f"No embedding found for candidate {candidate_id}")
        return {'error': 'Candidate embedding not found'}
    
    # Convert embedding to list if it's a string
    candidate_embedding = _ensure_list(candidate_embedding)
    
    # Get all jobs and embeddings
    jobs = db_manager.get_all_jobs()
    job_embeddings = db_manager.get_all_job_embeddings()
    
    # Convert job embeddings to lists if they're strings
    job_embeddings = {
        jid: _ensure_list(emb) for jid, emb in job_embeddings.items()
    }
    
    logger.info(f"Found {len(jobs)} jobs with {len(job_embeddings)} embeddings")
    
    # Generate recommendations
    recommendations = engine.rank_jobs_for_candidate(
        candidate=candidate,
        candidate_embedding=candidate_embedding,
        jobs=jobs,
        job_embeddings=job_embeddings,
        top_k=top_k
    )
    
    logger.info(f"Generated {len(recommendations['recommendations'])} recommendations")
    
    # Save to database
    if save_to_db:
        saved_count = 0
        for rec in recommendations['recommendations']:
            success = db_manager.save_recommendation(
                candidate_id=candidate_id,
                job_id=rec['job_id'],
                match_score=rec['match_score'],
                skills_match=rec['matching_factors']['skills_match'],
                experience_match=rec['matching_factors']['experience_match'],
                education_match=rec['matching_factors']['education_match'],
                semantic_similarity=rec['matching_factors']['semantic_similarity'],
                matched_skills=rec['matched_skills'],
                missing_skills=rec['missing_skills'],
                explanation=rec['explanation']
            )
            if success:
                saved_count += 1
        
        logger.info(f"Saved {saved_count} recommendations to database")
    
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
            # All candidates
            stats = generate_all_recommendations(
                db_manager=db_manager,
                engine=engine,
                top_k=args.top_k,
                save_to_db=not args.no_save_db,
                output_dir=args.output_dir
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
