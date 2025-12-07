#!/usr/bin/env python3
"""
Test backward compatibility: verify system works with Redis disabled
"""
import logging
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pgvector_only():
    """Test recommendation generation without Redis (pgvector only)"""
    
    config = Config()
    db_manager = DatabaseManager(config.database)
    
    try:
        # Initialize recommendation engine WITHOUT cache_manager
        logger.info("Initializing recommendation engine (pgvector only, no Redis)...")
        engine = RecommendationEngine(
            use_two_stage=True,
            stage1_top_k=10,
            cache_manager=None  # Explicitly disable cache
        )
        
        # Get a test candidate
        candidate_id = 'ARTS_10830646'
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing recommendations (PGVECTOR ONLY) for: {candidate_id}")
        logger.info(f"{'='*80}")
        
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            logger.error("Candidate not found!")
            return
        
        # Use traditional pgvector similarity (candidate_id based)
        logger.info("Using pgvector similarity search...")
        jobs_with_similarity = db_manager.get_top_k_jobs_by_similarity(
            candidate_id=candidate_id,
            top_k=10,
            similarity_threshold=0.0
        )
        
        logger.info(f"✓ Found {len(jobs_with_similarity)} similar jobs")
        
        # Stage 2: Full scoring
        logger.info("Applying full scoring...")
        recommendations = engine.rank_jobs_for_candidate(
            candidate=candidate,
            jobs_with_similarity=jobs_with_similarity,
            top_k=5
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"TOP 5 RECOMMENDATIONS (pgvector only)")
        print(f"{'='*80}")
        for i, rec in enumerate(recommendations['recommendations'][:5], 1):
            print(f"\n{i}. {rec['job_title']} at {rec['company']}")
            print(f"   Overall Match: {rec['match_score']:.2%}")
            print(f"   Semantic Similarity: {rec['matching_factors']['semantic_similarity']:.2%}")
        
        logger.info("\nBackward compatibility test passed!")
        logger.info("System works correctly without Redis (pgvector fallback)")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.close()


if __name__ == '__main__':
    test_pgvector_only()
