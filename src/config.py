"""
Configuration loader for CV Parser and Recommender System
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Load and provide access to configuration settings"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configurations" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def llm_parser(self) -> str:
        """LLM provider: gemini or azure"""
        return self._config.get('llm_parser', 'gemini')
    
    @property
    def embd_model(self) -> str:
        """Embedding model name"""
        return self._config.get('embd_model', 'sentence-transformers/all-mpnet-base-v2')
    
    @property
    def embd_dimension(self) -> int:
        """Embedding vector dimension"""
        return self._config.get('embd_dimension', 768)
    
    @property
    def num_professions(self) -> int:
        """Number of professions to process"""
        return self._config.get('num_professions', 5)
    
    @property
    def num_cv_per_profession(self) -> int:
        """Number of CVs per profession"""
        return self._config.get('num_cv_per_profession', 5)
    
    @property
    def professions(self) -> list:
        """List of professions"""
        return self._config.get('professions', [])
    
    @property
    def resume_base_path(self) -> Path:
        """Base path for resume dataset"""
        return Path(self._config.get('resume_base_path', ''))
    
    @property
    def job_base_path(self) -> Path:
        """Base path for resume dataset"""
        return Path(self._config.get('jobs_base_path', ''))
    
    @property
    def database(self) -> Dict[str, Any]:
        """Database configuration from environment variables"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'cv_job_db'),
            'user': os.getenv('POSTGRES_USER', 'cv_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'cv_password_123')
        }
    
    @property
    def num_workers(self) -> int:
        """Number of parallel workers"""
        return self._config.get('batch_processing', {}).get('num_workers', 4)
    
    @property
    def chunk_size(self) -> int:
        """Chunk size for batch processing"""
        return self._config.get('batch_processing', {}).get('chunk_size', 5)
    
    @property
    def redis(self) -> Dict[str, Any]:
        """Redis cache configuration"""
        redis_config = self._config.get('redis', {})
        return {
            'enabled': redis_config.get('enabled', False),
            'host': redis_config.get('host', 'localhost'),
            'port': redis_config.get('port', 6379),
            'db': redis_config.get('db', 0),
            'cache_ttl': redis_config.get('cache_ttl', 3600),
            'use_for_similarity': redis_config.get('use_for_similarity', True)
        }


# Global config instance
config = Config()
