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
        
        # Calculate statistics
        stats_by_operation = self.aggregator.aggregate_by_operation(perf_metrics)
        success_rates = self.aggregator.calculate_success_rate(perf_metrics)
        
        # Extract dataset context
        dataset_info = self._extract_dataset_info(perf_metrics)
        
        # Build report
        report = {
            'generated_at': datetime.now().isoformat(),
            'dataset_info': dataset_info,
            'performance_summary': stats_by_operation,
            'success_rates': success_rates,
            'total_metrics_collected': {
                'performance': len(perf_metrics),
                'query': len(query_metrics),
                'system': len(system_metrics)
            },
            'bottleneck_analysis': self._analyze_bottlenecks(perf_metrics),
            'extrapolations': self._calculate_extrapolations(stats_by_operation, dataset_info),
            'throughput_analysis': self._analyze_throughput(perf_metrics),
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
        """Analyze where time is being spent"""
        if not metrics:
            return {}
        
        total_time_by_operation = {}
        for metric in metrics:
            if metric.success:
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
    
    def _calculate_extrapolations(
        self,
        stats: Dict[str, Dict[str, float]],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate extrapolated performance at different scales
        
        This is the key function for toy database -> production predictions
        """
        current_cvs = dataset_info.get('num_cvs', 25)
        current_jobs = dataset_info.get('num_jobs', 5)
        
        extrapolations = {
            'current_scale': {
                'num_cvs': current_cvs,
                'num_jobs': current_jobs
            },
            'projections': {}
        }
        
        # Define target scales
        scales = [
            {'name': '100_cvs_10_jobs', 'cvs': 100, 'jobs': 10},
            {'name': '1K_cvs_100_jobs', 'cvs': 1000, 'jobs': 100},
            {'name': '10K_cvs_1K_jobs', 'cvs': 10000, 'jobs': 1000},
            {'name': '100K_cvs_10K_jobs', 'cvs': 100000, 'jobs': 10000}
        ]
        
        for scale in scales:
            scale_factor_cvs = scale['cvs'] / max(current_cvs, 1)
            scale_factor_jobs = scale['jobs'] / max(current_jobs, 1)
            
            projections = {}
            
            # CV-dependent operations (linear scaling)
            cv_operations = ['cv_parsing', 'embedding_generation', 'cv_db_insert']
            for op in cv_operations:
                if op in stats:
                    mean_time = stats[op]['mean']
                    projections[op] = {
                        'estimated_mean_seconds': mean_time,  # Per-CV time stays same
                        'estimated_total_seconds': mean_time * scale['cvs'],
                        'scaling': 'linear_with_cvs'
                    }
            
            # Job-dependent operations (linear with jobs, per candidate)
            if 'recommendation_generation' in stats:
                mean_time = stats['recommendation_generation']['mean']
                # Time per recommendation scales with number of jobs
                projections['recommendation_generation'] = {
                    'estimated_mean_seconds': mean_time * scale_factor_jobs,
                    'estimated_total_seconds': mean_time * scale_factor_jobs * scale['cvs'],
                    'scaling': 'linear_with_jobs_times_cvs'
                }
            
            # Similarity search (sub-linear with pgvector IVFFLAT)
            if 'similarity_search' in stats:
                mean_time = stats['similarity_search']['mean']
                # IVFFLAT is roughly O(sqrt(n)) to O(n/k)
                # Conservative estimate: O(n^0.7)
                search_scale = scale_factor_jobs ** 0.7
                projections['similarity_search'] = {
                    'estimated_mean_seconds': mean_time * search_scale,
                    'estimated_total_seconds': mean_time * search_scale * scale['cvs'],
                    'scaling': 'sublinear_with_jobs',
                    'note': 'Assumes pgvector IVFFLAT index optimization'
                }
            
            extrapolations['projections'][scale['name']] = {
                'scale': scale,
                'operations': projections
            }
        
        return extrapolations
    
    def _analyze_throughput(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze throughput metrics"""
        throughput = {}
        
        for op_type in ['cv_parsing', 'recommendation_generation']:
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
        <h2>📦 Dataset Context</h2>
        <div class="info">
            <strong>Current Scale:</strong> Processing metrics with 
            <code>{report['dataset_info']['num_cvs']} CVs</code> and 
            <code>{report['dataset_info']['num_jobs']} jobs</code>
        </div>
        <p>These metrics are collected at toy scale. Extrapolations below estimate production performance.</p>
    </div>
    
    <div class="card">
        <h2>⚡ Performance Summary</h2>
        <div class="metric-grid">
            {self._render_performance_metrics(report['performance_summary'])}
        </div>
    </div>
    
    <div class="card">
        <h2>🎯 Success Rates</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Success Rate</th>
            </tr>
            {self._render_success_rates(report['success_rates'])}
        </table>
    </div>
    
    <div class="card">
        <h2>🔍 Bottleneck Analysis</h2>
        <p>Time distribution across operations:</p>
        {self._render_bottlenecks(report['bottleneck_analysis'])}
    </div>
    
    <div class="card">
        <h2>📈 Scale Extrapolations</h2>
        <div class="warning">
            <strong>⚠️ Projection Notice:</strong> These are mathematical extrapolations based on current measurements.
            Actual production performance may vary due to network latency, API rate limits, and database optimization.
        </div>
        {self._render_extrapolations(report['extrapolations'])}
    </div>
    
    <div class="card">
        <h2>🚀 Throughput Analysis</h2>
        {self._render_throughput(report['throughput_analysis'])}
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
    
    def _render_performance_metrics(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Render performance metrics as HTML"""
        html = ""
        for op, metrics in stats.items():
            html += f"""
            <div class="metric-box">
                <div class="label">{op.replace('_', ' ').title()}</div>
                <div class="value">{metrics['mean']:.3f}<span class="unit">s</span></div>
                <div class="unit">p95: {metrics['p95']:.3f}s | count: {metrics['count']}</div>
            </div>
            """
        return html
    
    def _render_success_rates(self, rates: Dict[str, float]) -> str:
        """Render success rates as HTML table rows"""
        html = ""
        for op, rate in rates.items():
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
    
    def _render_extrapolations(self, extrapolations: Dict[str, Any]) -> str:
        """Render scale extrapolations"""
        html = ""
        for scale_name, data in extrapolations.get('projections', {}).items():
            scale = data['scale']
            html += f"""
            <h3>Scale: {scale['cvs']} CVs × {scale['jobs']} Jobs</h3>
            <table>
                <tr>
                    <th>Operation</th>
                    <th>Est. Mean Time</th>
                    <th>Est. Total Time</th>
                    <th>Scaling Model</th>
                </tr>
            """
            for op, proj in data['operations'].items():
                html += f"""
                <tr>
                    <td>{op.replace('_', ' ').title()}</td>
                    <td>{proj['estimated_mean_seconds']:.3f}s</td>
                    <td>{proj['estimated_total_seconds'] / 60:.1f} min</td>
                    <td><code>{proj['scaling']}</code></td>
                </tr>
                """
            html += "</table>"
        return html
    
    def _render_throughput(self, throughput: Dict[str, Any]) -> str:
        """Render throughput analysis"""
        if not throughput:
            return "<p>No throughput data available.</p>"
        
        html = "<div class='metric-grid'>"
        for op, data in throughput.items():
            html += f"""
            <div class="metric-box">
                <div class="label">{op.replace('_', ' ').title()}</div>
                <div class="value">{data['operations_per_minute']:.2f}<span class="unit">/min</span></div>
                <div class="unit">{data['operations_per_hour']:.0f}/hour</div>
            </div>
            """
        html += "</div>"
        return html
    
    def _render_resources(self, resources: Dict[str, Any]) -> str:
        """Render resource efficiency"""
        if not resources:
            return "<p>No resource data available.</p>"
        
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
