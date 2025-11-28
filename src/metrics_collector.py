"""
Metrics Collector - Lightweight data classes for performance metrics
Handles in-memory buffering and JSON export
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation_type: str
    entity_id: Optional[str]
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Dataset size context for extrapolation
    dataset_size_cvs: Optional[int] = None
    dataset_size_jobs: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class QueryMetric:
    """Database query performance metric"""
    query_type: str
    duration_ms: float
    rows_affected: int
    index_used: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SystemMetric:
    """System resource usage snapshot"""
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    active_workers: int
    throughput_per_min: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Dataset context
    dataset_size_cvs: Optional[int] = None
    dataset_size_jobs: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MetricsBuffer:
    """In-memory buffer for metrics before database write"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize metrics buffer
        
        Args:
            max_size: Maximum buffer size before auto-flush
        """
        self.max_size = max_size
        self.performance_metrics: List[PerformanceMetric] = []
        self.query_metrics: List[QueryMetric] = []
        self.system_metrics: List[SystemMetric] = []
        self._lock = False
    
    def add_performance_metric(self, metric: PerformanceMetric):
        """Add performance metric to buffer"""
        self.performance_metrics.append(metric)
        if len(self.performance_metrics) >= self.max_size:
            logger.warning(f"Performance metrics buffer full ({self.max_size}), consider flushing")
    
    def add_query_metric(self, metric: QueryMetric):
        """Add query metric to buffer"""
        self.query_metrics.append(metric)
        if len(self.query_metrics) >= self.max_size:
            logger.warning(f"Query metrics buffer full ({self.max_size}), consider flushing")
    
    def add_system_metric(self, metric: SystemMetric):
        """Add system metric to buffer"""
        self.system_metrics.append(metric)
    
    def get_all_performance_metrics(self) -> List[PerformanceMetric]:
        """Get all performance metrics"""
        return self.performance_metrics.copy()
    
    def get_all_query_metrics(self) -> List[QueryMetric]:
        """Get all query metrics"""
        return self.query_metrics.copy()
    
    def get_all_system_metrics(self) -> List[SystemMetric]:
        """Get all system metrics"""
        return self.system_metrics.copy()
    
    def clear(self):
        """Clear all buffers"""
        self.performance_metrics.clear()
        self.query_metrics.clear()
        self.system_metrics.clear()
    
    def export_to_json(self, output_path: Path) -> bool:
        """
        Export all metrics to JSON file
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_metrics': [m.to_dict() for m in self.performance_metrics],
                'query_metrics': [m.to_dict() for m in self.query_metrics],
                'system_metrics': [m.to_dict() for m in self.system_metrics],
                'summary': {
                    'total_performance_metrics': len(self.performance_metrics),
                    'total_query_metrics': len(self.query_metrics),
                    'total_system_metrics': len(self.system_metrics)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported metrics to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics to JSON: {e}")
            return False


class MetricsAggregator:
    """Calculate statistics from collected metrics"""
    
    @staticmethod
    def calculate_stats(values: List[float]) -> Dict[str, float]:
        """
        Calculate statistical measures
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with min, max, mean, median, p95, p99
        """
        if not values:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'count': 0
            }
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'min': sorted_values[0],
            'max': sorted_values[-1],
            'mean': sum(values) / n,
            'median': sorted_values[n // 2],
            'p50': sorted_values[n // 2],
            'p95': sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
            'p99': sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
            'count': n
        }
    
    @staticmethod
    def aggregate_by_operation(metrics: List[PerformanceMetric]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics by operation type
        
        Args:
            metrics: List of performance metrics
            
        Returns:
            Dictionary mapping operation_type to statistics
        """
        by_operation: Dict[str, List[float]] = {}
        
        for metric in metrics:
            if metric.operation_type not in by_operation:
                by_operation[metric.operation_type] = []
            by_operation[metric.operation_type].append(metric.duration_seconds)
        
        return {
            op_type: MetricsAggregator.calculate_stats(durations)
            for op_type, durations in by_operation.items()
        }
    
    @staticmethod
    def calculate_success_rate(metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """
        Calculate success rate by operation type
        
        Args:
            metrics: List of performance metrics
            
        Returns:
            Dictionary mapping operation_type to success rate
        """
        by_operation: Dict[str, Dict[str, int]] = {}
        
        for metric in metrics:
            if metric.operation_type not in by_operation:
                by_operation[metric.operation_type] = {'total': 0, 'success': 0}
            
            by_operation[metric.operation_type]['total'] += 1
            if metric.success:
                by_operation[metric.operation_type]['success'] += 1
        
        return {
            op_type: counts['success'] / counts['total'] if counts['total'] > 0 else 0.0
            for op_type, counts in by_operation.items()
        }
    
    @staticmethod
    def calculate_throughput(
        metrics: List[PerformanceMetric],
        operation_type: str
    ) -> float:
        """
        Calculate throughput (operations per minute)
        
        Args:
            metrics: List of performance metrics
            operation_type: Type of operation to calculate throughput for
            
        Returns:
            Operations per minute
        """
        filtered = [m for m in metrics if m.operation_type == operation_type and m.success]
        
        if not filtered:
            return 0.0
        
        # Parse timestamps and find time span
        timestamps = [datetime.fromisoformat(m.timestamp) for m in filtered]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        time_span_minutes = (max_time - min_time).total_seconds() / 60.0
        
        if time_span_minutes == 0:
            return 0.0
        
        return len(filtered) / time_span_minutes


# Global metrics buffer instance
_global_buffer = MetricsBuffer()


def get_global_buffer() -> MetricsBuffer:
    """Get global metrics buffer instance"""
    return _global_buffer
