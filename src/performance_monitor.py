"""
Performance Monitor - Centralized timing and metrics collection
Provides decorators, context managers, and database persistence
"""
import time
import functools
import logging
import psutil
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

from src.metrics_collector import (
    PerformanceMetric,
    QueryMetric,
    SystemMetric,
    get_global_buffer
)

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Centralized performance monitoring"""
    
    def __init__(self, db_manager=None):
        """
        Initialize performance monitor
        
        Args:
            db_manager: Optional database manager for persistence
        """
        self.db_manager = db_manager
        self.buffer = get_global_buffer()
        self._dataset_context = {
            'num_cvs': None,
            'num_jobs': None
        }
    
    def set_dataset_context(self, num_cvs: Optional[int] = None, num_jobs: Optional[int] = None):
        """
        Set dataset size context for metrics
        
        Args:
            num_cvs: Number of CVs being processed
            num_jobs: Number of jobs in system
        """
        if num_cvs is not None:
            self._dataset_context['num_cvs'] = num_cvs
        if num_jobs is not None:
            self._dataset_context['num_jobs'] = num_jobs
        
        logger.info(f"Dataset context: {self._dataset_context['num_cvs']} CVs, {self._dataset_context['num_jobs']} jobs")
    
    def record_performance(
        self,
        operation_type: str,
        duration_seconds: float,
        success: bool,
        entity_id: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a performance metric
        
        Args:
            operation_type: Type of operation (e.g., 'cv_parsing', 'embedding_generation')
            duration_seconds: Duration in seconds
            success: Whether operation succeeded
            entity_id: Optional identifier for the entity being processed
            error_message: Optional error message
            metadata: Optional additional metadata
        """
        metric = PerformanceMetric(
            operation_type=operation_type,
            entity_id=entity_id,
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            dataset_size_cvs=self._dataset_context['num_cvs'],
            dataset_size_jobs=self._dataset_context['num_jobs']
        )
        
        self.buffer.add_performance_metric(metric)
        
        # Optionally persist to database immediately
        if self.db_manager and hasattr(self.db_manager, 'insert_performance_metric'):
            try:
                self.db_manager.insert_performance_metric(metric)
            except Exception as e:
                logger.debug(f"Failed to persist metric to DB: {e}")
    
    def record_query(
        self,
        query_type: str,
        duration_ms: float,
        rows_affected: int,
        index_used: bool = False
    ):
        """
        Record a database query metric
        
        Args:
            query_type: Type of query
            duration_ms: Duration in milliseconds
            rows_affected: Number of rows affected
            index_used: Whether an index was used
        """
        metric = QueryMetric(
            query_type=query_type,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            index_used=index_used
        )
        
        self.buffer.add_query_metric(metric)
    
    def record_system_snapshot(
        self,
        active_workers: int = 0,
        throughput_per_min: float = 0.0
    ):
        """
        Record system resource usage snapshot
        
        Args:
            active_workers: Number of active worker processes
            throughput_per_min: Current throughput
        """
        try:
            process = psutil.Process()
            
            metric = SystemMetric(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                disk_io_mb=0.0,  # Simplified for now
                active_workers=active_workers,
                throughput_per_min=throughput_per_min,
                dataset_size_cvs=self._dataset_context['num_cvs'],
                dataset_size_jobs=self._dataset_context['num_jobs']
            )
            
            self.buffer.add_system_metric(metric)
            
        except Exception as e:
            logger.debug(f"Failed to record system snapshot: {e}")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def set_monitor(monitor: PerformanceMonitor):
    """Set global performance monitor"""
    global _global_monitor
    _global_monitor = monitor


@contextmanager
def track_time(
    operation_type: str,
    entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking operation time
    
    Usage:
        with track_time('cv_parsing', entity_id='CV123'):
            parse_cv()
    
    Args:
        operation_type: Type of operation
        entity_id: Optional entity identifier
        metadata: Optional metadata
    """
    start_time = time.time()
    success = True
    error_msg = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        duration = time.time() - start_time
        monitor = get_monitor()
        monitor.record_performance(
            operation_type=operation_type,
            duration_seconds=duration,
            success=success,
            entity_id=entity_id,
            error_message=error_msg,
            metadata=metadata
        )


def track_performance(operation_type: str, include_args: bool = False):
    """
    Decorator for tracking function performance
    
    Usage:
        @track_performance('cv_parsing')
        def parse_cv(cv_path):
            ...
    
    Args:
        operation_type: Type of operation
        include_args: Whether to include function arguments in metadata
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            entity_id = None
            metadata = {}
            
            # Try to extract entity_id from common argument names
            if args and hasattr(args[0], '__class__'):
                # Skip 'self' if it's a method
                func_args = args[1:] if len(args) > 1 else args
            else:
                func_args = args
            
            # Look for common ID patterns
            for arg in func_args:
                if isinstance(arg, str) and any(x in str(arg).lower() for x in ['id', 'path', 'file']):
                    entity_id = str(arg)[:100]  # Limit length
                    break
            
            if include_args:
                metadata['args_count'] = len(args)
                metadata['kwargs_keys'] = list(kwargs.keys())
            
            start_time = time.time()
            success = True
            error_msg = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration = time.time() - start_time
                monitor = get_monitor()
                monitor.record_performance(
                    operation_type=operation_type,
                    duration_seconds=duration,
                    success=success,
                    entity_id=entity_id,
                    error_message=error_msg,
                    metadata=metadata
                )
        
        return wrapper
    return decorator


@contextmanager
def track_query(query_type: str):
    """
    Context manager for tracking database query performance
    
    Usage:
        with track_query('get_candidate'):
            result = cursor.execute(query)
    
    Args:
        query_type: Type of query
    """
    start_time = time.time()
    
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        monitor = get_monitor()
        monitor.record_query(
            query_type=query_type,
            duration_ms=duration_ms,
            rows_affected=0,  # Could be enhanced to capture actual row count
            index_used=False  # Could be enhanced with EXPLAIN analysis
        )


class PerformanceTimer:
    """Simple timer for measuring elapsed time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        Get elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
