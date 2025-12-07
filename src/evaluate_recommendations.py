"""
Recommendation Evaluation Runner with Scalable Batch Processing
Evaluates recommendation quality using ground truth data with performance tracking
"""
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np

from src.config import Config
from src.database_manager import DatabaseManager
from src.recommendation_engine import RecommendationEngine
from src.evaluation_metrics import EvaluationMetrics, AggregatedMetrics, format_metrics_report
from src.performance_monitor import get_monitor, set_monitor, PerformanceMonitor, track_performance
from src.metrics_collector import ProcessingSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Scalable evaluation pipeline for recommendation systems.
    
    Features:
    - Batch processing with memory-efficient streaming
    - Parallel recommendation generation for multiple candidates
    - Quality metrics (Precision, Recall, F1, NDCG, MRR, Hit Rate, Coverage)
    - Performance metrics (latency, throughput) via PerformanceMonitor
    - Baseline comparisons (Random, Popularity, Skills-Only, Embeddings-Only)
    - Comprehensive evaluation reports (JSON + human-readable)
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: Dict[str, Any],
        monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize evaluator
        
        Args:
            db_manager: Database manager instance
            config: Configuration dictionary
            monitor: Optional performance monitor (will create if None)
        """
        self.db = db_manager
        self.config = config
        self.monitor = monitor or get_monitor()
        
        # Initialize recommendation engine with config
        rec_config = config.get('recommendation', {})
        self.engine = RecommendationEngine(
            weights=rec_config.get('weights'),
            use_two_stage=rec_config.get('use_two_stage', True),
            stage1_top_k=rec_config.get('stage1_top_k', 50),
            stage1_threshold=rec_config.get('stage1_threshold', 0.3)
        )
        
        logger.info(f"Evaluator initialized with engine weights: {self.engine.weights}")
    
    @track_performance('load_ground_truth')
    def load_ground_truth_csv(
        self,
        csv_path: str,
        split: str = 'test'
    ) -> Tuple[Dict[str, List[Tuple[str, int]]], Set[str]]:
        """
        Load ground truth from CSV file with memory-efficient streaming
        
        Args:
            csv_path: Path to ground truth CSV file
            split: Which split to load ('train', 'val', or 'test')
            
        Returns:
            Tuple of:
            - Dictionary mapping candidate_id to list of (job_id, relevance) tuples
            - Set of all job IDs in the ground truth
        """
        logger.info(f"Loading ground truth from {csv_path} (split: {split})")
        
        ground_truth = defaultdict(list)
        all_jobs = set()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    candidate_id = row['candidate_id']
                    job_id = row['job_id']
                    relevance = int(row['relevance'])
                    
                    ground_truth[candidate_id].append((job_id, relevance))
                    all_jobs.add(job_id)
        
        logger.info(f"Loaded ground truth for {len(ground_truth)} candidates, {len(all_jobs)} unique jobs")
        return dict(ground_truth), all_jobs
    
    def _batch_fetch_recommendations(
        self,
        candidate_ids: List[str],
        batch_size: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Fetch existing recommendations from database in batches (memory-efficient)
        
        Args:
            candidate_ids: List of candidate IDs to fetch recommendations for
            batch_size: Number of candidates to process per batch
            
        Returns:
            Dictionary mapping candidate_id to list of (job_id, match_score) tuples
        """
        all_recommendations = {}
        
        for i in range(0, len(candidate_ids), batch_size):
            batch = candidate_ids[i:i+batch_size]
            logger.debug(f"Fetching recommendations batch {i//batch_size + 1}: {len(batch)} candidates")
            
            # Build query with IN clause for batch
            placeholders = ','.join(['%s'] * len(batch))
            query = f"""
                SELECT candidate_id, job_id, match_score
                FROM recommendations
                WHERE candidate_id IN ({placeholders})
                ORDER BY candidate_id, match_score DESC
            """
            
            self.db.cursor.execute(query, batch)
            rows = self.db.cursor.fetchall()
            
            # Group by candidate
            for candidate_id, job_id, match_score in rows:
                if candidate_id not in all_recommendations:
                    all_recommendations[candidate_id] = []
                all_recommendations[candidate_id].append((job_id, match_score))
        
        return all_recommendations
    
    @track_performance('generate_recommendations_for_evaluation')
    def generate_recommendations_batch(
        self,
        candidate_ids: List[str],
        top_k: int = 20,
        use_existing: bool = True,
        max_workers: int = 4
    ) -> Dict[str, List[str]]:
        """
        Generate recommendations for a batch of candidates (parallel processing)
        
        Args:
            candidate_ids: List of candidate IDs to generate recommendations for
            top_k: Number of top recommendations to generate per candidate
            use_existing: If True, use existing recommendations from DB (faster)
                         If False, regenerate on-the-fly (slower but fresh)
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping candidate_id to ordered list of job_ids
        """
        logger.info(f"Generating recommendations for {len(candidate_ids)} candidates (top_k={top_k}, use_existing={use_existing})")
        
        if use_existing:
            # FAST PATH: Fetch from database
            db_recommendations = self._batch_fetch_recommendations(candidate_ids)
            
            recommendations = {}
            for candidate_id in candidate_ids:
                if candidate_id in db_recommendations:
                    # Take top-K, extract job_ids only
                    recommendations[candidate_id] = [
                        job_id for job_id, score in db_recommendations[candidate_id][:top_k]
                    ]
                else:
                    logger.warning(f"No recommendations found for candidate {candidate_id}")
                    recommendations[candidate_id] = []
            
            return recommendations
        
        else:
            # SLOW PATH: Generate on-the-fly (for testing different algorithms)
            recommendations = {}
            
            def process_candidate(candidate_id: str) -> Tuple[str, List[str]]:
                """Worker function for parallel processing"""
                try:
                    candidate = self.db.get_candidate(candidate_id)
                    if not candidate:
                        return candidate_id, []
                    
                    # Get all jobs with similarity
                    if self.engine.use_two_stage:
                        jobs = self.db.get_top_k_jobs_by_similarity(
                            candidate_id,
                            top_k=self.engine.stage1_top_k,
                            similarity_threshold=self.engine.stage1_threshold
                        )
                    else:
                        jobs = self.db.get_all_jobs_with_similarity(candidate_id)
                    
                    # Rank jobs
                    ranked_recommendations = self.engine.rank_jobs_for_candidate(
                        candidate, jobs, top_k=top_k
                    )
                    
                    job_ids = [rec['job_id'] for rec in ranked_recommendations]
                    return candidate_id, job_ids
                    
                except Exception as e:
                    logger.error(f"Error generating recommendations for {candidate_id}: {e}")
                    return candidate_id, []
            
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_candidate, cid) for cid in candidate_ids]
                
                for future in as_completed(futures):
                    candidate_id, job_ids = future.result()
                    recommendations[candidate_id] = job_ids
            
            return recommendations
    
    @track_performance('evaluate_quality')
    def evaluate_quality(
        self,
        predictions: Dict[str, List[str]],
        ground_truth: Dict[str, List[Tuple[str, int]]],
        all_jobs: Set[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation quality using multiple metrics
        
        Args:
            predictions: Dict mapping candidate_id to ordered list of predicted job_ids
            ground_truth: Dict mapping candidate_id to list of (job_id, relevance) tuples
            all_jobs: Set of all possible job IDs (for coverage calculation)
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary containing evaluation metrics and per-candidate details
        """
        logger.info(f"Evaluating quality for {len(predictions)} candidates at K={k_values}")
        
        # Convert ground truth to formats needed by different metrics
        all_predictions = []
        all_ground_truth_binary = []  # For Precision/Recall (binary: relevant or not)
        all_relevance_scores = []     # For NDCG (graded relevance)
        per_candidate_metrics = []
        
        for candidate_id in predictions.keys():
            pred_list = predictions.get(candidate_id, [])
            gt_list = ground_truth.get(candidate_id, [])
            
            if not gt_list:
                logger.warning(f"No ground truth for candidate {candidate_id}, skipping")
                continue
            
            # Binary ground truth (relevance > 0 means relevant)
            gt_binary = {job_id for job_id, rel in gt_list if rel > 0}
            
            # Relevance scores dict (all jobs, including irrelevant ones)
            relevance_dict = {job_id: rel for job_id, rel in gt_list}
            
            all_predictions.append(pred_list)
            all_ground_truth_binary.append(gt_binary)
            all_relevance_scores.append(relevance_dict)
            
            # Compute per-candidate metrics (for debugging/analysis)
            candidate_metrics = {
                'candidate_id': candidate_id,
                'num_predictions': len(pred_list),
                'num_relevant_items': len(gt_binary)
            }
            
            for k in k_values:
                candidate_metrics[f'precision@{k}'] = EvaluationMetrics.precision_at_k(pred_list, gt_binary, k)
                candidate_metrics[f'recall@{k}'] = EvaluationMetrics.recall_at_k(pred_list, gt_binary, k)
                candidate_metrics[f'ndcg@{k}'] = EvaluationMetrics.ndcg_at_k(pred_list, relevance_dict, k)
            
            candidate_metrics['mrr'] = EvaluationMetrics.mean_reciprocal_rank(pred_list, gt_binary)
            per_candidate_metrics.append(candidate_metrics)
        
        # Aggregate metrics across all candidates
        aggregate_metrics = AggregatedMetrics.compute_aggregate_metrics(
            all_predictions=all_predictions,
            all_ground_truths=all_ground_truth_binary,
            all_relevance_scores=all_relevance_scores,
            k_values=k_values,
            all_items=all_jobs
        )
        
        return {
            'aggregate': aggregate_metrics,
            'per_candidate': per_candidate_metrics,
            'summary': {
                'num_candidates_evaluated': len(all_predictions),
                'num_unique_jobs': len(all_jobs),
                'avg_predictions_per_candidate': np.mean([len(p) for p in all_predictions]),
                'avg_relevant_per_candidate': np.mean([len(gt) for gt in all_ground_truth_binary])
            }
        }
    
    def evaluate_baseline(
        self,
        baseline_name: str,
        candidate_ids: List[str],
        ground_truth: Dict[str, List[Tuple[str, int]]],
        all_jobs: Set[str],
        top_k: int = 20,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate a baseline recommendation strategy
        
        Args:
            baseline_name: Name of baseline ('random', 'popularity', 'skills_only', 'embeddings_only')
            candidate_ids: List of candidate IDs to evaluate
            ground_truth: Ground truth data
            all_jobs: Set of all job IDs
            top_k: Number of recommendations to generate
            k_values: K values for evaluation
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating baseline: {baseline_name}")
        
        # Generate baseline predictions
        if baseline_name == 'random':
            predictions = self._generate_random_baseline(candidate_ids, list(all_jobs), top_k)
        elif baseline_name == 'popularity':
            predictions = self._generate_popularity_baseline(candidate_ids, top_k)
        elif baseline_name == 'skills_only':
            predictions = self._generate_skills_only_baseline(candidate_ids, top_k)
        elif baseline_name == 'embeddings_only':
            predictions = self._generate_embeddings_only_baseline(candidate_ids, top_k)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        
        # Evaluate
        results = self.evaluate_quality(predictions, ground_truth, all_jobs, k_values)
        results['baseline_name'] = baseline_name
        
        return results
    
    def _generate_random_baseline(
        self,
        candidate_ids: List[str],
        all_job_ids: List[str],
        top_k: int
    ) -> Dict[str, List[str]]:
        """Generate random recommendations (baseline)"""
        import random
        predictions = {}
        for candidate_id in candidate_ids:
            predictions[candidate_id] = random.sample(all_job_ids, min(top_k, len(all_job_ids)))
        return predictions
    
    def _generate_popularity_baseline(
        self,
        candidate_ids: List[str],
        top_k: int
    ) -> Dict[str, List[str]]:
        """Generate popularity-based recommendations (most recommended jobs overall)"""
        # Get most popular jobs from database
        self.db.cursor.execute("""
            SELECT job_id, COUNT(*) as rec_count
            FROM recommendations
            GROUP BY job_id
            ORDER BY rec_count DESC
            LIMIT %s
        """, (top_k,))
        
        popular_jobs = [row[0] for row in self.db.cursor.fetchall()]
        
        # Same popular jobs for everyone
        predictions = {candidate_id: popular_jobs for candidate_id in candidate_ids}
        return predictions
    
    def _generate_skills_only_baseline(
        self,
        candidate_ids: List[str],
        top_k: int
    ) -> Dict[str, List[str]]:
        """Generate skills-only recommendations (ignore experience, education, embeddings)"""
        predictions = {}
        
        for candidate_id in candidate_ids:
            candidate = self.db.get_candidate(candidate_id)
            if not candidate:
                predictions[candidate_id] = []
                continue
            
            jobs = self.db.get_all_jobs()
            
            # Score jobs based on skills only
            job_scores = []
            for job in jobs:
                skills_score, _, _ = self.engine.calculate_skills_match(
                    candidate.get('skills_technical', []),
                    candidate.get('skills_soft', []),
                    job.get('skills_technical', []),
                    job.get('skills_soft', [])
                )
                job_scores.append((job['job_id'], skills_score))
            
            # Sort by skills score
            job_scores.sort(key=lambda x: x[1], reverse=True)
            predictions[candidate_id] = [job_id for job_id, _ in job_scores[:top_k]]
        
        return predictions
    
    def _generate_embeddings_only_baseline(
        self,
        candidate_ids: List[str],
        top_k: int
    ) -> Dict[str, List[str]]:
        """Generate embeddings-only recommendations (pure vector similarity)"""
        predictions = {}
        
        for candidate_id in candidate_ids:
            # Get top-K by similarity only
            jobs = self.db.get_top_k_jobs_by_similarity(
                candidate_id,
                top_k=top_k,
                similarity_threshold=0.0  # No threshold
            )
            predictions[candidate_id] = [job['job_id'] for job in jobs]
        
        return predictions
    
    def run_evaluation(
        self,
        ground_truth_csv: str,
        split: str = 'test',
        top_k: int = 20,
        k_values: List[int] = [5, 10, 20],
        use_existing_recommendations: bool = True,
        evaluate_baselines: bool = True,
        output_dir: str = 'data/performance_reports'
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            ground_truth_csv: Path to ground truth CSV file
            split: Which split to evaluate ('test', 'val', 'train')
            top_k: Number of recommendations to generate/evaluate
            k_values: List of K values for metrics
            use_existing_recommendations: Use existing recommendations from DB
            evaluate_baselines: Also evaluate baseline methods
            output_dir: Directory to save evaluation reports
            
        Returns:
            Complete evaluation results dictionary
        """
        logger.info("=" * 70)
        logger.info(f"Starting evaluation pipeline (split={split}, top_k={top_k})")
        logger.info("=" * 70)
        
        # Start performance monitoring session
        session_id = f"evaluation_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitor.start_session(session_id, 'evaluation')
        
        start_time = time.time()
        
        # Load ground truth
        ground_truth, all_jobs = self.load_ground_truth_csv(ground_truth_csv, split)
        candidate_ids = list(ground_truth.keys())
        
        logger.info(f"Loaded {len(candidate_ids)} candidates for evaluation")
        
        # Generate/fetch recommendations
        predictions = self.generate_recommendations_batch(
            candidate_ids,
            top_k=top_k,
            use_existing=use_existing_recommendations
        )
        
        # Evaluate main system
        logger.info("Evaluating main recommendation system...")
        main_results = self.evaluate_quality(predictions, ground_truth, all_jobs, k_values)
        
        # Evaluate baselines
        baseline_results = {}
        if evaluate_baselines:
            for baseline_name in ['random', 'popularity', 'skills_only', 'embeddings_only']:
                try:
                    baseline_results[baseline_name] = self.evaluate_baseline(
                        baseline_name, candidate_ids, ground_truth, all_jobs, top_k, k_values
                    )
                except Exception as e:
                    logger.error(f"Failed to evaluate baseline {baseline_name}: {e}")
        
        # Compile final results
        total_time = time.time() - start_time
        
        results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'split': split,
                'top_k': top_k,
                'k_values': k_values,
                'num_candidates': len(candidate_ids),
                'num_jobs': len(all_jobs),
                'use_existing_recommendations': use_existing_recommendations,
                'total_time_seconds': total_time,
                'session_id': session_id
            },
            'main_system': main_results,
            'baselines': baseline_results
        }
        
        # End monitoring session
        self.monitor.end_session(
            items_processed=len(candidate_ids),
            items_success=len(predictions)
        )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_file = output_path / f"evaluation_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {json_file}")
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print human-readable evaluation summary"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY".center(70))
        print("=" * 70)
        
        metadata = results['metadata']
        print(f"\nDataset: {metadata['split'].upper()} split")
        print(f"Candidates: {metadata['num_candidates']}")
        print(f"Jobs: {metadata['num_jobs']}")
        print(f"Total Time: {metadata['total_time_seconds']:.2f} seconds")
        
        # Main system metrics
        print(format_metrics_report(results['main_system']['aggregate'], "Main Recommendation System"))
        
        # Baseline comparisons
        if results['baselines']:
            print("\n" + "=" * 70)
            print("BASELINE COMPARISONS".center(70))
            print("=" * 70)
            
            # Comparison table
            print(f"\n{'Metric':<20} {'Main':>12}", end='')
            for baseline_name in results['baselines'].keys():
                print(f" {baseline_name:>12}", end='')
            print()
            print("-" * (20 + 12 * (1 + len(results['baselines']))))
            
            main_metrics = results['main_system']['aggregate']
            for metric_name in sorted(main_metrics.keys()):
                print(f"{metric_name:<20} {main_metrics[metric_name]:>12.4f}", end='')
                for baseline_name, baseline_data in results['baselines'].items():
                    baseline_value = baseline_data['aggregate'].get(metric_name, 0.0)
                    print(f" {baseline_value:>12.4f}", end='')
                print()
        
        print("\n" + "=" * 70)


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate recommendation quality')
    parser.add_argument('--config', default='configurations/config.yaml', help='Config file')
    parser.add_argument('--ground-truth', required=True, help='Ground truth CSV file')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Data split')
    parser.add_argument('--top-k', type=int, default=20, help='Number of recommendations')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20], help='K values for metrics')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate recommendations (don\'t use existing)')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline evaluations')
    parser.add_argument('--output-dir', default='data/performance_reports', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config_obj = Config(args.config)
    
    # Initialize components
    db_manager = DatabaseManager(config_obj.database)
    monitor = PerformanceMonitor(db_manager)
    set_monitor(monitor)
    
    # Convert config to dict for evaluator
    config_dict = {
        'recommendation': {
            'weights': None,  # Will use defaults
            'use_two_stage': True,
            'stage1_top_k': 50,
            'stage1_threshold': 0.3
        }
    }
    
    # Run evaluation
    evaluator = RecommendationEvaluator(db_manager, config_dict, monitor)
    results = evaluator.run_evaluation(
        ground_truth_csv=args.ground_truth,
        split=args.split,
        top_k=args.top_k,
        k_values=args.k_values,
        use_existing_recommendations=not args.regenerate,
        evaluate_baselines=not args.no_baselines,
        output_dir=args.output_dir
    )
    
    # Close database
    db_manager.close()
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
