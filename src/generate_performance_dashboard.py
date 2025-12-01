#!/usr/bin/env python3
"""
Run Performance Tests and Generate Dashboard
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database_manager import DatabaseManager
from src.performance_monitor import get_monitor, set_monitor, PerformanceMonitor
from src.dashboard_generator import DashboardGenerator
from src.metrics_collector import (
    get_global_buffer, 
    PerformanceMetric, 
    ProcessingSession
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Generate performance dashboard from collected metrics
    
    This script:
    1. Connects to database to get dataset context
    2. Loads metrics from database
    3. Generates performance report with key metrics
    4. Creates HTML dashboard
    """
    try:
        logger.info("="*80)
        logger.info("PERFORMANCE DASHBOARD GENERATOR")
        logger.info("="*80)
        
        # Connect to database to get context
        logger.info("Connecting to database...")
        db_manager = DatabaseManager(config.database)
        
        # Get dataset context
        num_cvs = db_manager.get_candidate_count()
        num_jobs = db_manager.get_job_count()
        
        logger.info(f"Dataset: {num_cvs} CVs, {num_jobs} jobs")
        
        # Load metrics from database and populate buffer
        logger.info("\nLoading metrics from database...")
        buffer = get_global_buffer()
        
        # Load performance metrics
        perf_metrics_data = db_manager.load_performance_metrics()
        for metric_dict in perf_metrics_data:
            metric = PerformanceMetric(
                operation_type=metric_dict['operation_type'],
                entity_id=metric_dict['entity_id'],
                duration_seconds=metric_dict['duration_seconds'],
                success=metric_dict['success'],
                error_message=metric_dict['error_message'],
                metadata=metric_dict['metadata'],
                dataset_size_cvs=metric_dict['dataset_size_cvs'],
                dataset_size_jobs=metric_dict['dataset_size_jobs'],
                timestamp=metric_dict['timestamp']
            )
            buffer.add_performance_metric(metric)
        
        # Load processing sessions
        sessions_data = db_manager.load_processing_sessions()
        for session_dict in sessions_data:
            session = ProcessingSession(
                session_id=session_dict['session_id'],
                session_type=session_dict['session_type'],
                start_time=session_dict['start_time'],
                end_time=session_dict['end_time'],
                duration_seconds=session_dict['duration_seconds'],
                items_processed=session_dict['items_processed'],
                items_success=session_dict['items_success'],
                items_failed=session_dict['items_failed'],
                items_skipped=session_dict['items_skipped'],
                total_cvs_in_db=session_dict['total_cvs_in_db'],
                total_jobs_in_db=session_dict['total_jobs_in_db'],
                metadata=session_dict['metadata']
            )
            buffer.add_session(session)
        
        # Load query metrics from database
        query_metrics_data = db_manager.load_query_metrics()
        from src.metrics_collector import QueryMetric
        for metric_dict in query_metrics_data:
            metric = QueryMetric(
                query_type=metric_dict['query_type'],
                duration_ms=metric_dict['duration_ms'],
                rows_affected=metric_dict['rows_affected'],
                index_used=metric_dict['index_used'],
                timestamp=metric_dict['timestamp']
            )
            buffer.add_query_metric(metric)
        
        # Load system metrics from database
        system_metrics_data = db_manager.load_system_metrics()
        from src.metrics_collector import SystemMetric
        for metric_dict in system_metrics_data:
            metric = SystemMetric(
                cpu_percent=metric_dict['cpu_percent'],
                memory_mb=metric_dict['memory_mb'],
                disk_io_mb=metric_dict['disk_io_mb'],
                active_workers=metric_dict['active_workers'],
                throughput_per_min=metric_dict['throughput_per_min'],
                dataset_size_cvs=metric_dict['dataset_size_cvs'],
                dataset_size_jobs=metric_dict['dataset_size_jobs'],
                timestamp=metric_dict['timestamp']
            )
            buffer.add_system_metric(metric)
        
        perf_metrics = buffer.get_all_performance_metrics()
        query_metrics = buffer.get_all_query_metrics()
        system_metrics = buffer.get_all_system_metrics()
        sessions = buffer.get_all_sessions()
        
        logger.info(f"Collected metrics:")
        logger.info(f"  - Performance: {len(perf_metrics)}")
        logger.info(f"  - Query: {len(query_metrics)}")
        logger.info(f"  - System: {len(system_metrics)}")
        logger.info(f"  - Sessions: {len(sessions)}")
        
        if len(perf_metrics) == 0 and len(sessions) == 0:
            logger.warning("No performance metrics or sessions found in database!")
            logger.warning("Run CV processing or recommendations first:")
            logger.warning("  python src/process_cvs.py")
            logger.warning("  python src/generate_recommendations.py")
            return 1
        
        # Generate dashboard
        logger.info("\nGenerating performance dashboard...")
        output_dir = Path("data/performance_reports")
        
        dashboard_gen = DashboardGenerator(buffer)
        report = dashboard_gen.generate_report(output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("✅ DASHBOARD GENERATED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info("\nFiles created:")
        logger.info(f"  - HTML Dashboard: performance_dashboard_*.html")
        logger.info(f"  - JSON Report: performance_report_*.json")
        logger.info(f"  - Raw Metrics: raw_metrics_*.json")
        
        # Print key metrics for assignment
        logger.info("\n" + "="*80)
        logger.info("KEY PERFORMANCE METRICS (Assignment Requirements)")
        logger.info("="*80)
        
        perf_summary = report.get('performance_summary', {})
        throughput = report.get('throughput_analysis', {})
        query_perf = report.get('query_performance', {})
        
        # 1. Average CV parsing time
        if 'cv_parsing' in perf_summary:
            logger.info(f"\n✅ Average CV Parsing Time: {perf_summary['cv_parsing']['mean']:.3f}s")
            logger.info(f"   P95: {perf_summary['cv_parsing']['p95']:.3f}s")
            logger.info(f"   Total processed: {perf_summary['cv_parsing']['count']}")
        
        # 2. Average recommendation generation time
        if 'recommendation_generation' in perf_summary:
            logger.info(f"\n✅ Average Recommendation Generation Time: {perf_summary['recommendation_generation']['mean']:.3f}s")
            logger.info(f"   P95: {perf_summary['recommendation_generation']['p95']:.3f}s")
            logger.info(f"   Total generated: {perf_summary['recommendation_generation']['count']}")
        
        # 3. System throughput
        logger.info(f"\n✅ System Throughput:")
        if 'cv_parsing' in throughput:
            logger.info(f"   CV Processing: {throughput['cv_parsing']['operations_per_minute']:.2f} CVs/minute")
        if 'recommendation_generation' in throughput:
            logger.info(f"   Recommendations: {throughput['recommendation_generation']['operations_per_minute']:.2f} recommendations/minute")
        
        # 4. Database query performance
        if query_perf and 'overall' in query_perf:
            logger.info(f"\n✅ Database Query Performance:")
            logger.info(f"   Average query time: {query_perf['overall']['mean']*1000:.2f}ms")
            logger.info(f"   P95 query time: {query_perf['overall']['p95']*1000:.2f}ms")
            logger.info(f"   Total queries: {query_perf['overall']['count']}")
        
        # Print full summary
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        for op_type, stats in report.get('performance_summary', {}).items():
            logger.info(f"\n{op_type.replace('_', ' ').title()}:")
            logger.info(f"  Mean: {stats['mean']:.3f}s")
            logger.info(f"  P95: {stats['p95']:.3f}s")
            logger.info(f"  Count: {stats['count']}")
        
        # Print bottlenecks
        logger.info("\n" + "="*80)
        logger.info("BOTTLENECK ANALYSIS")
        logger.info("="*80)
        
        for op_type, data in report.get('bottleneck_analysis', {}).items():
            logger.info(f"{op_type}: {data['percentage']:.1f}% of total time")
        
        # Print throughput
        logger.info("\n" + "="*80)
        logger.info("THROUGHPUT ANALYSIS")
        logger.info("="*80)
        
        for op_type, data in throughput.items():
            if isinstance(data, dict) and 'operations_per_minute' in data:
                logger.info(f"{op_type}: {data['operations_per_minute']:.2f} operations/minute")
        
        db_manager.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
