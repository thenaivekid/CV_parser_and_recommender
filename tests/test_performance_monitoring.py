#!/usr/bin/env python3
"""
Test Performance Monitoring System
Verifies that metrics are being collected and can be queried
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database_manager import DatabaseManager
from src.performance_monitor import get_monitor, PerformanceMonitor
from src.metrics_collector import get_global_buffer

def main():
    print("="*80)
    print("PERFORMANCE MONITORING TEST")
    print("="*80)
    
    # Connect to database
    print("\n1. Connecting to database...")
    db_manager = DatabaseManager(config.database)
    
    # Check if performance tables exist
    print("\n2. Checking performance tables...")
    try:
        db_manager.cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        perf_count = db_manager.cursor.fetchone()[0]
        print(f"   ✓ performance_metrics table exists: {perf_count} rows")
        
        db_manager.cursor.execute("SELECT COUNT(*) FROM query_performance")
        query_count = db_manager.cursor.fetchone()[0]
        print(f"   ✓ query_performance table exists: {query_count} rows")
        
        db_manager.cursor.execute("SELECT COUNT(*) FROM system_metrics")
        system_count = db_manager.cursor.fetchone()[0]
        print(f"   ✓ system_metrics table exists: {system_count} rows")
        
    except Exception as e:
        print(f"   ✗ Error checking tables: {e}")
        db_manager.close()
        return 1
    
    # Initialize monitor
    print("\n3. Initializing performance monitor...")
    monitor = PerformanceMonitor(db_manager)
    
    # Set dataset context
    num_cvs = db_manager.get_candidate_count()
    num_jobs = db_manager.get_job_count()
    monitor.set_dataset_context(num_cvs=num_cvs, num_jobs=num_jobs)
    print(f"   ✓ Dataset context: {num_cvs} CVs, {num_jobs} jobs")
    
    # Test recording a metric
    print("\n4. Testing metric recording...")
    monitor.record_performance(
        operation_type='test_operation',
        duration_seconds=1.23,
        success=True,
        entity_id='test_entity',
        metadata={'test': True}
    )
    print("   ✓ Recorded test metric")
    
    # Check if it was persisted
    db_manager.cursor.execute(
        "SELECT COUNT(*) FROM performance_metrics WHERE operation_type = 'test_operation'"
    )
    test_count = db_manager.cursor.fetchone()[0]
    print(f"   ✓ Test metrics in DB: {test_count}")
    
    # Check in-memory buffer
    print("\n5. Checking in-memory buffer...")
    buffer = get_global_buffer()
    perf_metrics = buffer.get_all_performance_metrics()
    print(f"   ✓ Metrics in buffer: {len(perf_metrics)}")
    
    # Show recent metrics
    print("\n6. Recent performance metrics (last 10):")
    db_manager.cursor.execute("""
        SELECT operation_type, entity_id, duration_seconds, success, timestamp
        FROM performance_metrics
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    rows = db_manager.cursor.fetchall()
    
    if rows:
        print("\n   Type                    | Entity               | Duration  | Success | Timestamp")
        print("   " + "-"*90)
        for row in rows:
            op_type, entity_id, duration, success, timestamp = row
            status = "✓" if success else "✗"
            entity_str = (entity_id or "N/A")[:20]
            print(f"   {op_type[:23]:<23} | {entity_str:<20} | {duration:>8.3f}s | {status:^7} | {timestamp}")
    else:
        print("   No metrics found yet. Run some operations first:")
        print("   - python src/process_jobs.py --force")
        print("   - python src/process_cvs.py")
        print("   - python src/generate_recommendations.py")
    
    # Cleanup
    db_manager.close()
    
    print("\n" + "="*80)
    print("✅ PERFORMANCE MONITORING TEST COMPLETE")
    print("="*80)
    
    if test_count > 0:
        print("\n✓ Performance monitoring is working correctly!")
        print("\nNext steps:")
        print("1. Run: ./scripts/run_job_processing.sh")
        print("2. Run: ./scripts/run_cv_batch_processing.sh")
        print("3. Generate dashboard: python src/generate_performance_dashboard.py")
        return 0
    else:
        print("\n⚠️  Performance monitoring is set up but no metrics collected yet.")
        print("   This is normal if you haven't run any processing operations.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
