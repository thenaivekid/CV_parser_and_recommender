#!/usr/bin/env python3
"""
Test script for Redis-cached recommendation generation
"""
import logging
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.cache_manager import CachedRecommendationEngine
from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_redis_recommendations():
    """Test recommendation generation with Redis caching"""
    
    config = Config()
    db_manager = DatabaseManager(config.database)
    
    try:
        # Initialize Redis cache manager
        logger.info("Initializing Redis cache manager...")
        redis_config = config.redis
        cache_manager = CachedRecommendationEngine(db_manager, redis_config)
        
        # Initialize recommendation engine with cache
        logger.info("Initializing recommendation engine...")
        engine = RecommendationEngine(
            use_two_stage=False,
            cache_manager=cache_manager
        )
        
        # Get a test candidate
        candidate_id = 'ARTS_10830646'
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing recommendations for candidate: {candidate_id}")
        logger.info(f"{'='*80}")
        
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            logger.error("Candidate not found!")
            return
        
        # Get CV embedding
        cv_embedding = db_manager.get_candidate_embedding(candidate_id)
        if not cv_embedding:
            logger.error("Candidate embedding not found!")
            return
        
        logger.info(f"✓ Retrieved CV embedding (dimension: {len(cv_embedding)})")
        
        # Use Redis cache to get similar jobs
        logger.info("\nUsing Redis-cached similarity search...")
        jobs_with_similarity = cache_manager.get_similar_jobs(
            cv_embedding=cv_embedding,
            top_k=10,
            similarity_threshold=0.0
        )
        
        logger.info(f"✓ Found {len(jobs_with_similarity)} similar jobs")
        
        # Stage 2: Full scoring with RecommendationEngine
        logger.info("\nApplying full scoring (skills, experience, education)...")
        recommendations = engine.rank_jobs_for_candidate(
            candidate=candidate,
            jobs_with_similarity=jobs_with_similarity,
            top_k=5
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"TOP 5 RECOMMENDATIONS (Redis-cached)")
        print(f"{'='*80}")
        for i, rec in enumerate(recommendations['recommendations'][:5], 1):
            print(f"\n{i}. {rec['job_title']} at {rec['company']}")
            print(f"   Job ID: {rec['job_id']}")
            print(f"   Overall Match: {rec['match_score']:.2%}")
            print(f"   Factors:")
            print(f"     - Skills:    {rec['matching_factors']['skills_match']:.2%}")
            print(f"     - Experience: {rec['matching_factors']['experience_match']:.2%}")
            print(f"     - Education:  {rec['matching_factors']['education_match']:.2%}")
            print(f"     - Semantic:   {rec['matching_factors']['semantic_similarity']:.2%}")
            print(f"   Matched Skills: {', '.join(rec['matched_skills'][:3])}")
            if rec['missing_skills']:
                print(f"   Missing Skills: {', '.join(rec['missing_skills'][:3])}")
        
        # Get cache stats
        stats = cache_manager.get_stats()
        print(f"\n{'='*80}")
        print(f"REDIS CACHE STATISTICS")
        print(f"{'='*80}")
        print(f"Connected: {stats.get('connected')}")
        print(f"Job embeddings cached: {stats.get('job_embeddings_cached')}")
        print(f"Cache TTL: {stats.get('job_embeddings_ttl')}s")
        print(f"Total keys: {stats.get('total_keys')}")
        print(f"Memory used: {stats.get('used_memory_human')}")
        
        logger.info("\n✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.close()


if __name__ == '__main__':
    test_redis_recommendations()
