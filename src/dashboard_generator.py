"""
Dashboard Generator - Create performance reports and visualizations
Generates HTML dashboards with current metrics and extrapolated projections
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.metrics_collector import (
    MetricsBuffer,
    MetricsAggregator,
    PerformanceMetric,
    ProcessingSession,
    get_global_buffer
)

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generate performance dashboards and reports"""
    
    def __init__(self, buffer: Optional[MetricsBuffer] = None):
        """
        Initialize dashboard generator
        
        Args:
            buffer: Optional metrics buffer (uses global if not provided)
        """
        self.buffer = buffer or get_global_buffer()
        self.aggregator = MetricsAggregator()
    
    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Report data dictionary
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all metrics
        perf_metrics = self.buffer.get_all_performance_metrics()
        query_metrics = self.buffer.get_all_query_metrics()
        system_metrics = self.buffer.get_all_system_metrics()
        sessions = self.buffer.get_all_sessions()
        
        # Calculate statistics
        stats_by_operation = self.aggregator.aggregate_by_operation(perf_metrics)
        success_rates = self.aggregator.calculate_success_rate(perf_metrics)
        
        # Extract dataset context (prefer from sessions, fallback to metrics)
        dataset_info = self._extract_dataset_info_from_sessions(sessions, perf_metrics)
        
        # Build report
        report = {
            'generated_at': datetime.now().isoformat(),
            'performance_summary': stats_by_operation,
            'success_rates': success_rates,
            'total_metrics_collected': {
                'performance': len(perf_metrics),
                'query': len(query_metrics),
                'system': len(system_metrics)
            },
            'bottleneck_analysis': self._analyze_bottlenecks(perf_metrics),
            'throughput_analysis': self._analyze_throughput(perf_metrics),
            'query_performance': self._analyze_query_performance(query_metrics),
            'resource_efficiency': self._analyze_resources(system_metrics)
        }
        
        # Save JSON report
        json_path = output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")
        
        # Generate HTML dashboard
        html_path = output_dir / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        self._generate_html_dashboard(report, html_path)
        logger.info(f"Saved HTML dashboard to {html_path}")
        
        # Export raw metrics
        metrics_path = output_dir / f"raw_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.buffer.export_to_json(metrics_path)
        
        return report
    
    def _summarize_sessions(self, sessions: List[ProcessingSession]) -> List[Dict[str, Any]]:
        """Summarize processing sessions for report"""
        return [
            {
                'session_id': s.session_id,
                'type': s.session_type,
                'start_time': s.start_time,
                'end_time': s.end_time,
                'duration_seconds': s.duration_seconds,
                'items_processed': s.items_processed,
                'items_success': s.items_success,
                'items_failed': s.items_failed,
                'items_skipped': s.items_skipped,
                'throughput_per_minute': (s.items_success / (s.duration_seconds / 60)) if s.duration_seconds and s.items_success else 0,
                'success_rate': (s.items_success / s.items_processed * 100) if s.items_processed else 0,
                'metadata': s.metadata
            }
            for s in sessions
        ]
    
    def _extract_dataset_info_from_sessions(
        self, 
        sessions: List[ProcessingSession], 
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Extract dataset info from sessions (preferred) or fallback to metrics"""
        # Try to get from most recent session
        if sessions:
            latest_session = sessions[-1]
            return {
                'num_cvs_in_db': latest_session.total_cvs_in_db or 0,
                'num_jobs_in_db': latest_session.total_jobs_in_db or 0,
                'num_cvs_processed_this_run': latest_session.items_success if latest_session.session_type == 'cv_processing' else 0,
                'num_jobs_processed_this_run': latest_session.items_success if latest_session.session_type == 'job_processing' else 0,
                'source': 'session_data'
            }
        
        # Fallback to metrics
        return {
            **self._extract_dataset_info(metrics),
            'source': 'metric_data'
        }
    
    def _analyze_throughput_from_sessions(self, sessions: List[ProcessingSession]) -> Dict[str, Any]:
        """Analyze throughput from session data"""
        if not sessions:
            return {'message': 'No session data available'}
        
        analysis = {
            'by_session_type': {}
        }
        
        for session in sessions:
            if session.duration_seconds and session.items_success > 0:
                session_type = session.session_type
                throughput = session.items_success / (session.duration_seconds / 60)
                
                if session_type not in analysis['by_session_type']:
                    analysis['by_session_type'][session_type] = []
                
                analysis['by_session_type'][session_type].append({
                    'session_id': session.session_id,
                    'items_per_minute': throughput,
                    'duration_seconds': session.duration_seconds,
                    'items_processed': session.items_success
                })
        
        # Calculate averages
        summary = {}
        for session_type, throughputs in list(analysis['by_session_type'].items()):
            if throughputs and isinstance(throughputs, list):
                avg_throughput = sum(t['items_per_minute'] for t in throughputs) / len(throughputs)
                summary[session_type] = {
                    'avg_per_minute': avg_throughput,
                    'avg_per_hour': avg_throughput * 60,
                    'num_sessions': len(throughputs)
                }
        
        analysis['by_session_type'] = summary
        return analysis
    
    def _extract_dataset_info(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract dataset size information from metrics"""
        if not metrics:
            return {'num_cvs': 0, 'num_jobs': 0}
        
        # Get most recent dataset sizes
        for metric in reversed(metrics):
            if metric.dataset_size_cvs is not None or metric.dataset_size_jobs is not None:
                return {
                    'num_cvs': metric.dataset_size_cvs or 0,
                    'num_jobs': metric.dataset_size_jobs or 0
                }
        
        return {'num_cvs': 0, 'num_jobs': 0}
    
    def _analyze_bottlenecks(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze where time is being spent - focus on CV processing and recommendations"""
        if not metrics:
            return {}
        
        # Filter to only CV-related and recommendation operations (no job operations)
        relevant_operations = {
            'cv_parsing',
            'embedding_generation',
            'cv_db_insert',
            'cv_embedding_db_insert',
            'recommendation_generation'
        }
        
        total_time_by_operation = {}
        for metric in metrics:
            if metric.success and metric.operation_type in relevant_operations:
                if metric.operation_type not in total_time_by_operation:
                    total_time_by_operation[metric.operation_type] = 0.0
                total_time_by_operation[metric.operation_type] += metric.duration_seconds
        
        total_time = sum(total_time_by_operation.values())
        
        if total_time == 0:
            return {}
        
        # Calculate percentages
        bottlenecks = {
            op: {
                'total_time_seconds': time,
                'percentage': (time / total_time) * 100
            }
            for op, time in sorted(
                total_time_by_operation.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return bottlenecks
    
    def _analyze_query_performance(self, query_metrics: List) -> Dict[str, Any]:
        """Analyze database query performance"""
        if not query_metrics:
            return {'message': 'No query metrics available'}
        
        # Convert ms to seconds for consistency
        query_times = [m.duration_ms / 1000.0 for m in query_metrics]
        query_types = {}
        
        for metric in query_metrics:
            query_type = metric.query_type if hasattr(metric, 'query_type') else 'unknown'
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(metric.duration_ms / 1000.0)
        
        analysis = {
            'overall': self.aggregator.calculate_stats(query_times),
            'by_type': {}
        }
        
        for qtype, times in query_types.items():
            analysis['by_type'][qtype] = self.aggregator.calculate_stats(times)
        
        return analysis
    
    def _analyze_throughput(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze throughput metrics for CV-related operations only"""
        throughput = {}
        
        # Only track CV-related operations (no job operations)
        for op_type in ['cv_parsing', 'embedding_generation', 'cv_db_insert', 'cv_embedding_db_insert', 'recommendation_generation']:
            rate = self.aggregator.calculate_throughput(metrics, op_type)
            if rate > 0:
                throughput[op_type] = {
                    'operations_per_minute': rate,
                    'operations_per_hour': rate * 60,
                    'operations_per_day': rate * 60 * 24
                }
        
        return throughput
    
    def _analyze_resources(self, system_metrics: List) -> Dict[str, Any]:
        """Analyze resource efficiency"""
        if not system_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in system_metrics]
        memory_values = [m.memory_mb for m in system_metrics]
        
        return {
            'cpu': self.aggregator.calculate_stats(cpu_values),
            'memory': self.aggregator.calculate_stats(memory_values)
        }
    
    def _generate_html_dashboard(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML dashboard"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Dashboard - CV Parser & Recommender</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .card h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-box .label {{
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-box .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }}
        .metric-box .unit {{
            font-size: 0.9em;
            color: #888;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .bottleneck-bar {{
            background: #e9ecef;
            height: 30px;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        .bottleneck-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Performance Dashboard</h1>
        <div class="timestamp">Generated: {report['generated_at']}</div>
    </div>
    
    <div class="card">
        <h2>⚡ Performance Summary (CV Processing & Recommendations)</h2>
        <div class="metric-grid">
            {self._render_performance_metrics(report['performance_summary'])}
        </div>
    </div>
    
    <div class="card">
        <h2>🎯 Success Rates (CV Processing & Recommendations)</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Success Rate</th>
            </tr>
            {self._render_success_rates(report['success_rates'])}
        </table>
    </div>
    
    <div class="card">
        <h2>🔍 Bottleneck Analysis (CV Processing Only)</h2>
        <p>Time distribution across CV processing and recommendation operations:</p>
        {self._render_bottlenecks(report['bottleneck_analysis'])}
    </div>
    
    <div class="card">
        <h2>🚀 System Throughput</h2>
        <p>Processing rate for CVs and recommendations:</p>
        {self._render_throughput(report['throughput_analysis'])}
    </div>
    
    <div class="card">
        <h2>🗄️ Database Query Performance</h2>
        <p>Track database query execution times for retrieval optimization:</p>
        {self._render_query_performance(report.get('query_performance', {}))}
    </div>
    
    <div class="card">
        <h2>💻 Resource Efficiency</h2>
        {self._render_resources(report['resource_efficiency'])}
    </div>
    
    <div class="card">
        <h2>📝 Metrics Collection Summary</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="label">Performance Metrics</div>
                <div class="value">{report['total_metrics_collected']['performance']}</div>
            </div>
            <div class="metric-box">
                <div class="label">Query Metrics</div>
                <div class="value">{report['total_metrics_collected']['query']}</div>
            </div>
            <div class="metric-box">
                <div class="label">System Snapshots</div>
                <div class="value">{report['total_metrics_collected']['system']}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _render_sessions(self, sessions: List[Dict[str, Any]]) -> str:
        """Render processing sessions summary"""
        if not sessions:
            return "<p>No processing sessions recorded.</p>"
        
        html = "<table><tr>"
        html += "<th>Session ID</th><th>Type</th><th>Duration</th>"
        html += "<th>Processed</th><th>Success</th><th>Failed</th><th>Skipped</th>"
        html += "<th>Throughput</th><th>Success Rate</th></tr>"
        
        for session in sessions:
            html += f"""
            <tr>
                <td><code>{session['session_id']}</code></td>
                <td>{session['type'].replace('_', ' ').title()}</td>
                <td>{session['duration_seconds']:.1f}s</td>
                <td>{session['items_processed']}</td>
                <td>{session['items_success']}</td>
                <td>{session['items_failed']}</td>
                <td>{session['items_skipped']}</td>
                <td><strong>{session['throughput_per_minute']:.2f}/min</strong></td>
                <td>{session['success_rate']:.1f}%</td>
            </tr>
            """
        
        return html + "</table>"
    
    def _render_processing_info(self, dataset_info: Dict[str, Any]) -> str:
        """Render information about what was processed this run"""
        cvs_processed = dataset_info.get('num_cvs_processed_this_run', 0)
        jobs_processed = dataset_info.get('num_jobs_processed_this_run', 0)
        
        if cvs_processed == 0 and jobs_processed == 0:
            return ""
        
        html = "<div class='warning'>"
        html += "<strong>This Run:</strong> "
        if cvs_processed > 0:
            html += f"Processed <strong>{cvs_processed}</strong> CVs"
        if jobs_processed > 0:
            if cvs_processed > 0:
                html += " and "
            html += f"processed <strong>{jobs_processed}</strong> jobs"
        html += "</div>"
        return html
    
    def _render_performance_metrics(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Render performance metrics as HTML - only CV processing and recommendations"""
        html = ""
        
        # Filter to relevant operations (CV-related only, no job operations)
        relevant_ops = ['cv_parsing', 'embedding_generation', 'cv_db_insert', 'cv_embedding_db_insert', 'recommendation_generation']
        
        for op in relevant_ops:
            if op in stats:
                metrics = stats[op]
                html += f"""
                <div class="metric-box">
                    <div class="label">{op.replace('_', ' ').title()}</div>
                    <div class="value">{metrics['mean']:.3f}<span class="unit">s</span></div>
                    <div class="unit">p95: {metrics['p95']:.3f}s | count: {metrics['count']}</div>
                </div>
                """
        return html
    
    def _render_success_rates(self, rates: Dict[str, float]) -> str:
        """Render success rates as HTML table rows - only CV processing and recommendations"""
        html = ""
        
        # Filter to relevant operations (CV-related only, no job operations)
        relevant_ops = ['cv_parsing', 'embedding_generation', 'cv_db_insert', 'cv_embedding_db_insert', 'recommendation_generation']
        
        for op in relevant_ops:
            if op in rates:
                rate = rates[op]
                percentage = rate * 100
                color = "#28a745" if rate >= 0.95 else ("#ffc107" if rate >= 0.8 else "#dc3545")
                html += f"""
                <tr>
                    <td>{op.replace('_', ' ').title()}</td>
                    <td><strong style="color: {color}">{percentage:.1f}%</strong></td>
                </tr>
                """
        return html
    
    def _render_bottlenecks(self, bottlenecks: Dict[str, Any]) -> str:
        """Render bottleneck analysis"""
        html = ""
        for op, data in bottlenecks.items():
            percentage = data['percentage']
            html += f"""
            <div style="margin: 15px 0;">
                <div style="margin-bottom: 5px;">
                    <strong>{op.replace('_', ' ').title()}</strong>: {data['total_time_seconds']:.2f}s ({percentage:.1f}%)
                </div>
                <div class="bottleneck-bar">
                    <div class="bottleneck-fill" style="width: {percentage}%">
                        {percentage:.1f}%
                    </div>
                </div>
            </div>
            """
        return html
    
    def _render_query_performance(self, query_perf: Dict[str, Any]) -> str:
        """Render database query performance"""
        if not query_perf or not query_perf.get('overall') or query_perf['overall']['count'] == 0:
            return """
            <div class="warning">
                <strong>⚠️ No Query Metrics Collected</strong>
                <p>Database query performance tracking is not yet instrumented.</p>
                <p><strong>To enable:</strong> Add query tracking to <code>database_manager.py</code> methods</p>
                <p>Key methods to instrument:</p>
                <ul>
                    <li><code>search_similar_candidates()</code> - Vector similarity search (CRITICAL for Task 4)</li>
                    <li><code>get_candidate()</code>, <code>get_job()</code> - Data retrieval</li>
                    <li><code>get_candidate_embedding()</code> - Embedding retrieval</li>
                </ul>
            </div>
            """
        
        html = "<div class='metric-grid'>"
        
        if 'overall' in query_perf:
            stats = query_perf['overall']
            html += f"""
            <div class="metric-box">
                <div class="label">Average Query Time</div>
                <div class="value">{stats['mean']*1000:.1f}<span class="unit">ms</span></div>
                <div class="unit">P95: {stats['p95']*1000:.1f}ms</div>
            </div>
            """
        
        html += "</div>"
        
        if 'by_type' in query_perf and query_perf['by_type']:
            html += "<h3>By Query Type</h3><table><tr><th>Query Type</th><th>Avg Time</th><th>P95</th><th>Count</th></tr>"
            for qtype, stats in query_perf['by_type'].items():
                html += f"""
                <tr>
                    <td>{qtype.replace('_', ' ').title()}</td>
                    <td>{stats['mean']*1000:.1f}ms</td>
                    <td>{stats['p95']*1000:.1f}ms</td>
                    <td>{stats['count']}</td>
                </tr>
                """
            html += "</table>"
        
        return html
    
    def _render_throughput(self, throughput: Dict[str, Any]) -> str:
        """Render throughput analysis"""
        if not throughput:
            return "<p>No throughput data available.</p>"
        
        html = "<div class='metric-grid'>"
        
        for op_type, data in throughput.items():
            if isinstance(data, dict) and 'operations_per_minute' in data:
                html += f"""
                <div class="metric-box">
                    <div class="label">{op_type.replace('_', ' ').title()}</div>
                    <div class="value">{data['operations_per_minute']:.2f}<span class="unit">/min</span></div>
                    <div class="unit">{data['operations_per_hour']:.0f}/hour</div>
                </div>
                """
        
        html += "</div>"
        return html
    
    def _render_resources(self, resources: Dict[str, Any]) -> str:
        """Render resource efficiency"""
        if not resources or ('cpu' not in resources and 'memory' not in resources):
            return """
            <div class="warning">
                <strong>⚠️ No System Resource Metrics Collected</strong>
                <p>System resource monitoring (CPU/Memory) is not yet instrumented.</p>
                <p><strong>To enable:</strong> Add <code>monitor.record_system_snapshot()</code> calls to processing scripts</p>
                <p>Add to:</p>
                <ul>
                    <li><code>src/process_cvs.py</code> - Call every 10 CVs processed</li>
                    <li><code>src/process_jobs.py</code> - Call every 5 jobs processed</li>
                    <li><code>src/generate_recommendations.py</code> - Call during batch processing</li>
                </ul>
            </div>
            """
        
        html = "<div class='metric-grid'>"
        if 'cpu' in resources:
            html += f"""
            <div class="metric-box">
                <div class="label">CPU Usage</div>
                <div class="value">{resources['cpu']['mean']:.1f}<span class="unit">%</span></div>
                <div class="unit">peak: {resources['cpu']['max']:.1f}%</div>
            </div>
            """
        if 'memory' in resources:
            html += f"""
            <div class="metric-box">
                <div class="label">Memory Usage</div>
                <div class="value">{resources['memory']['mean']:.1f}<span class="unit">MB</span></div>
                <div class="unit">peak: {resources['memory']['max']:.1f}MB</div>
            </div>
            """
        html += "</div>"
        return html
