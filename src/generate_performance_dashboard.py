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
from src.metrics_collector import get_global_buffer

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
    2. Retrieves collected metrics from global buffer
    3. Generates comprehensive performance report
    4. Creates HTML dashboard with extrapolations
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
        
        # Set dataset context in monitor
        monitor = get_monitor()
        monitor.set_dataset_context(num_cvs=num_cvs, num_jobs=num_jobs)
        
        # Get metrics buffer
        buffer = get_global_buffer()
        
        perf_metrics = buffer.get_all_performance_metrics()
        query_metrics = buffer.get_all_query_metrics()
        system_metrics = buffer.get_all_system_metrics()
        
        logger.info(f"Collected metrics:")
        logger.info(f"  - Performance: {len(perf_metrics)}")
        logger.info(f"  - Query: {len(query_metrics)}")
        logger.info(f"  - System: {len(system_metrics)}")
        
        if len(perf_metrics) == 0:
            logger.warning("No performance metrics collected yet!")
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
        
        # Print summary
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
        
        # Print extrapolation example
        logger.info("\n" + "="*80)
        logger.info("SCALE EXTRAPOLATION (Example: 1K CVs × 100 Jobs)")
        logger.info("="*80)
        
        projections = report.get('extrapolations', {}).get('projections', {})
        if '1K_cvs_100_jobs' in projections:
            scale_data = projections['1K_cvs_100_jobs']
            for op, proj in scale_data.get('operations', {}).items():
                total_min = proj['estimated_total_seconds'] / 60
                logger.info(f"{op}: ~{total_min:.1f} minutes total")
        
        db_manager.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
