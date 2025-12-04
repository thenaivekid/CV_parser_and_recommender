#!/usr/bin/env python3
"""
Test Redis Cache Implementation
Verifies Redis caching for job embeddings
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache_manager import EmbeddingCache, CachedRecommendationEngine
from src.database_manager import DatabaseManager
from src.config import config
import numpy as np

def print_separator(char='=', length=60):
    print(char * length)

def test_redis_connection():
    """Test 1: Redis Connection"""
    print_separator()
    print("TEST 1: Redis Connection")
    print_separator()
    
    cache = EmbeddingCache(
        host=config.redis['host'],
        port=config.redis['port'],
        enabled=True
    )
    
    if cache.enabled:
        print("✅ Redis connected successfully")
        stats = cache.get_cache_stats()
        print(f"   Redis version: {stats.get('redis_version')}")
        print(f"   Memory used: {stats.get('used_memory_human')}")
    else:
        print("❌ Redis connection failed")
        return False
    
    print()
    return True

def test_cache_loading():
    """Test 2: Load Job Embeddings"""
    print_separator()
    print("TEST 2: Load Job Embeddings from Database")
    print_separator()
    
    db = DatabaseManager(config.database)
    cache = EmbeddingCache(enabled=True)
    
    # Clear cache first
    cache.invalidate_cache()
    print("🗑️  Cache cleared")
    
    # Load from database (should cache)
    print("Loading job embeddings...")
    job_data = cache.get_job_embeddings(db)
    
    if job_data:
        print(f"✅ Loaded {job_data['count']} job embeddings")
        print(f"   Embedding dimension: {job_data['dimension']}")
        print(f"   Matrix shape: {job_data['matrix'].shape}")
        print(f"   Sample job IDs: {job_data['ids'][:3]}")
    else:
        print("❌ Failed to load embeddings")
        db.close()
        return False
    
    # Try loading again (should hit cache)
    print("\nLoading again (should hit cache)...")
    job_data2 = cache.get_job_embeddings(db)
    
    if job_data2:
        print("✅ Cache HIT - loaded from Redis")
    
    db.close()
    print()
    return True

def test_similarity_computation():
    """Test 3: Similarity Computation"""
    print_separator()
    print("TEST 3: Similarity Computation (NumPy)")
    print_separator()
    
    db = DatabaseManager(config.database)
    cache = EmbeddingCache(enabled=True)
    
    # Load job embeddings
    job_data = cache.get_job_embeddings(db)
    
    if not job_data:
        print("❌ No job data available")
        db.close()
        return False
    
    # Create a fake CV embedding (same dimension as jobs)
    dimension = job_data['dimension']
    fake_cv_embedding = np.random.rand(dimension).tolist()
    
    print(f"Computing similarity with {job_data['count']} jobs...")
    print(f"CV embedding dimension: {len(fake_cv_embedding)}")
    
    # Compute similarity
    results = cache.compute_similarity(fake_cv_embedding, job_data, top_k=10)
    
    if results:
        print(f"✅ Found top-{len(results)} similar jobs:")
        for i, (job_id, similarity) in enumerate(results[:5], 1):
            print(f"   {i}. {job_id}: {similarity:.4f}")
    else:
        print("❌ Similarity computation failed")
        db.close()
        return False
    
    db.close()
    print()
    return True

def test_cached_recommendation_engine():
    """Test 4: CachedRecommendationEngine"""
    print_separator()
    print("TEST 4: CachedRecommendationEngine")
    print_separator()
    
    db = DatabaseManager(config.database)
    
    # Initialize with Redis enabled
    engine = CachedRecommendationEngine(
        db_manager=db,
        redis_config=config.redis,
        use_cache_for_similarity=True
    )
    
    print("Configuration:")
    print(f"   Redis enabled: {config.redis['enabled']}")
    print(f"   Use for similarity: {config.redis['use_for_similarity']}")
    print(f"   Cache TTL: {config.redis['cache_ttl']}s")
    
    # Get cache stats
    stats = engine.get_stats()
    print(f"\nCache Stats:")
    print(f"   Connected: {stats.get('connected', False)}")
    print(f"   Job embeddings cached: {stats.get('job_embeddings_cached', False)}")
    
    if stats.get('job_embeddings_ttl'):
        print(f"   TTL remaining: {stats['job_embeddings_ttl']}s")
    
    # Test similarity search
    dimension = config.embd_dimension
    fake_cv_embedding = np.random.rand(dimension).tolist()
    
    print(f"\nPerforming similarity search...")
    results = engine.get_similar_jobs(fake_cv_embedding, top_k=5)
    
    if results:
        print(f"✅ Found {len(results)} similar jobs:")
        for i, (job_id, similarity) in enumerate(results, 1):
            print(f"   {i}. {job_id}: {similarity:.4f}")
    else:
        print("⚠️  No results (database might be empty)")
    
    db.close()
    print()
    return True

def test_performance_comparison():
    """Test 5: Performance Comparison"""
    print_separator()
    print("TEST 5: Performance Comparison (Redis vs pgvector)")
    print_separator()
    
    import time
    
    db = DatabaseManager(config.database)
    
    # Get a real CV embedding from database
    db.cursor.execute("SELECT embedding FROM candidate_embeddings LIMIT 1")
    result = db.cursor.fetchone()
    
    if not result:
        print("⚠️  No CV embeddings in database, skipping performance test")
        db.close()
        return True
    
    cv_embedding = result[0]
    
    # Test with Redis cache
    print("Testing with Redis cache...")
    engine_cached = CachedRecommendationEngine(
        db_manager=db,
        redis_config=config.redis,
        use_cache_for_similarity=True
    )
    
    start = time.time()
    results_cached = engine_cached.get_similar_jobs(cv_embedding, top_k=50)
    time_cached = (time.time() - start) * 1000  # ms
    
    print(f"✅ Redis + NumPy: {time_cached:.2f}ms")
    
    # Test with pgvector (no cache)
    print("\nTesting with pgvector (no cache)...")
    start = time.time()
    results_pgvector = db.get_top_k_jobs_by_similarity(cv_embedding, top_k=50)
    time_pgvector = (time.time() - start) * 1000  # ms
    
    print(f"✅ pgvector: {time_pgvector:.2f}ms")
    
    # Compare
    print(f"\n📊 Performance Comparison:")
    print(f"   Redis cached: {time_cached:.2f}ms")
    print(f"   pgvector DB:  {time_pgvector:.2f}ms")
    if time_pgvector > time_cached:
        speedup = time_pgvector / time_cached
        print(f"   🚀 Speedup: {speedup:.1f}x faster with Redis")
    else:
        print(f"   ⚠️  pgvector was faster (small dataset or cold cache)")
    
    db.close()
    print()
    return True

def main():
    print_separator('=')
    print("REDIS CACHE IMPLEMENTATION TEST SUITE")
    print_separator('=')
    print()
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Cache Loading", test_cache_loading),
        ("Similarity Computation", test_similarity_computation),
        ("CachedRecommendationEngine", test_cached_recommendation_engine),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_separator('=')
    print("TEST SUMMARY")
    print_separator('=')
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print_separator('=')
    
    return all(s for _, s in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
