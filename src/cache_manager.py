"""
Redis Cache Manager for Embedding-Based Recommendation System
Provides high-performance caching and similarity search using Redis + NumPy
"""
import redis
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Manages Redis cache for job embeddings with NumPy-based similarity search
    
    Features:
    - Cache job embeddings in Redis (avoid DB queries)
    - Fast similarity computation using NumPy vectorization
    - Automatic cache invalidation and refresh
    - Graceful fallback to database on cache miss
    """
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize Redis cache manager
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            ttl: Cache TTL in seconds (default: 1 hour)
            enabled: Enable/disable caching
        """
        self.enabled = enabled
        self.ttl = ttl
        
        if not self.enabled:
            logger.info("Redis caching disabled")
            return
        
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Redis connected: {host}:{port}")
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            logger.warning("Falling back to database-only mode")
            self.enabled = False
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            self.enabled = False
    
    def get_job_embeddings(self, db_manager) -> Optional[Dict[str, Any]]:
        """
        Get job embeddings from cache or database
        
        Args:
            db_manager: DatabaseManager instance for fallback
            
        Returns:
            Dictionary with 'ids' and 'matrix' keys, or None if error
        """
        if not self.enabled:
            return None
        
        cache_key = "job_embeddings_v1"
        
        try:
            # Try cache first
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                logger.debug("Cache HIT: job embeddings loaded from Redis")
                return pickle.loads(cached_data)
            
            # Cache miss - load from database
            logger.info("Cache MISS: loading job embeddings from database...")
            job_data = self._load_from_database(db_manager)
            
            if job_data:
                # Cache for future requests
                self.redis.setex(
                    cache_key,
                    self.ttl,
                    pickle.dumps(job_data)
                )
                logger.info(f"Cached {len(job_data['ids'])} job embeddings (TTL: {self.ttl}s)")
            
            return job_data
            
        except Exception as e:
            logger.error(f"Cache error: {e}")
            return None
    
    def _load_from_database(self, db_manager) -> Optional[Dict[str, Any]]:
        """
        Load job embeddings from database
        
        Args:
            db_manager: DatabaseManager instance
            
        Returns:
            Dictionary with job IDs and embedding matrix
        """
        try:
            # Query all job embeddings
            query = """
                SELECT je.job_id, je.embedding
                FROM job_embeddings je
                JOIN jobs j ON je.job_id = j.job_id
                ORDER BY je.job_id
            """
            
            db_manager.cursor.execute(query)
            results = db_manager.cursor.fetchall()
            
            if not results:
                logger.warning("No job embeddings found in database")
                return None
            
            # Convert to numpy arrays for fast computation
            job_ids = []
            embeddings = []
            
            for row in results:
                job_ids.append(row[0])
                embedding = row[1]
                # PostgreSQL vector type is returned as string, parse it
                if isinstance(embedding, str):
                    import ast
                    embedding = ast.literal_eval(embedding)
                embeddings.append(embedding)
            
            # Convert PostgreSQL arrays to numpy matrix
            job_matrix = np.array(embeddings, dtype=np.float32)
            
            logger.info(f"Loaded {len(job_ids)} job embeddings from database")
            logger.debug(f"Embedding matrix shape: {job_matrix.shape}")
            
            return {
                'ids': job_ids,
                'matrix': job_matrix,
                'count': len(job_ids),
                'dimension': job_matrix.shape[1] if len(job_matrix.shape) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from database: {e}")
            return None
    
    def compute_similarity(
        self, 
        cv_embedding: List[float],
        job_data: Dict[str, Any],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Compute cosine similarity between CV and all jobs using NumPy
        
        Args:
            cv_embedding: CV embedding vector
            job_data: Job data from cache (ids + matrix)
            top_k: Number of top results to return
            
        Returns:
            List of (job_id, similarity_score) tuples, sorted by similarity
        """
        try:
            # Convert CV embedding to numpy array
            cv_vector = np.array(cv_embedding, dtype=np.float32).reshape(1, -1)
            
            # Compute cosine similarity with all jobs at once (vectorized)
            # This is MUCH faster than iterating
            similarities = cosine_similarity(cv_vector, job_data['matrix'])[0]
            
            # Get top-k indices (argsort returns ascending, so reverse)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Create result list
            results = [
                (job_data['ids'][idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            logger.debug(f"Computed similarity with {len(job_data['ids'])} jobs, returning top-{top_k}")
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return []
    
    def invalidate_cache(self):
        """
        Invalidate job embeddings cache
        Call this when jobs are added/updated/deleted
        """
        if not self.enabled:
            return
        
        try:
            deleted = self.redis.delete("job_embeddings_v1")
            if deleted:
                logger.info("Cache invalidated: job embeddings")
            else:
                logger.debug("Cache key not found (already empty)")
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {'enabled': False}
        
        try:
            info = self.redis.info()
            
            # Check if job embeddings are cached
            cache_key = "job_embeddings_v1"
            cached = self.redis.exists(cache_key)
            ttl = self.redis.ttl(cache_key) if cached else None
            
            # Get cached data size
            cache_size = self.redis.memory_usage(cache_key) if cached else 0
            
            return {
                'enabled': True,
                'connected': True,
                'redis_version': info.get('redis_version'),
                'used_memory_human': info.get('used_memory_human'),
                'job_embeddings_cached': bool(cached),
                'job_embeddings_ttl': ttl,
                'job_embeddings_size_bytes': cache_size,
                'total_keys': self.redis.dbsize()
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'enabled': True, 'connected': False, 'error': str(e)}
    
    def close(self):
        """Close Redis connection"""
        if self.enabled and self.redis:
            try:
                self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


class CachedRecommendationEngine:
    """
    Recommendation engine with Redis caching support
    
    Automatically uses Redis cache if available, falls back to pgvector otherwise
    """
    
    def __init__(
        self,
        db_manager,
        redis_config: Optional[Dict[str, Any]] = None,
        use_cache_for_similarity: bool = True
    ):
        """
        Initialize cached recommendation engine
        
        Args:
            db_manager: DatabaseManager instance
            redis_config: Redis configuration dict
            use_cache_for_similarity: Use cached embeddings for similarity computation
        """
        self.db_manager = db_manager
        self.use_cache_for_similarity = use_cache_for_similarity
        
        # Initialize cache
        if redis_config and redis_config.get('enabled', False):
            self.cache = EmbeddingCache(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                ttl=redis_config.get('cache_ttl', 3600),
                enabled=True
            )
        else:
            logger.info("Redis caching disabled by configuration")
            self.cache = EmbeddingCache(enabled=False)
    
    def get_similar_jobs(
        self,
        cv_embedding: List[float],
        top_k: int = 50,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get similar jobs using cache or database
        
        Args:
            cv_embedding: CV embedding vector
            top_k: Number of top results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of job dictionaries with 'semantic_similarity' field
        """
        # Try Redis cache first
        if self.cache.enabled and self.use_cache_for_similarity:
            job_data = self.cache.get_job_embeddings(self.db_manager)
            
            if job_data:
                # Get similarity scores
                similarity_results = self.cache.compute_similarity(cv_embedding, job_data, top_k)
                
                # Apply threshold filter
                if similarity_threshold > 0:
                    similarity_results = [(job_id, sim) for job_id, sim in similarity_results if sim >= similarity_threshold]
                
                # Fetch full job details from database for each job_id
                jobs_with_similarity = []
                for job_id, similarity in similarity_results:
                    job = self.db_manager.get_job(job_id)
                    if job:
                        job['semantic_similarity'] = similarity
                        jobs_with_similarity.append(job)
                
                logger.debug(f"✅ Used Redis cache for similarity search (mode: fast)")
                return jobs_with_similarity
        
        logger.warning("Redis cache unavailable, cannot compute similarity without candidate_id")
        return []
    
    def invalidate_cache(self):
        """Invalidate cache when jobs change"""
        self.cache.invalidate_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_cache_stats()
