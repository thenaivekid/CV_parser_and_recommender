"""
Evaluation Metrics for Recommendation Quality Assessment
Implements standard information retrieval metrics: Precision, Recall, F1, NDCG, MRR, Hit Rate, Coverage
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for recommendation systems.
    
    Supports both binary relevance (0/1) and graded relevance (0/1/2).
    """
    
    @staticmethod
    def precision_at_k(
        predictions: List[str],
        ground_truth: Set[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Precision@K: proportion of recommended items that are relevant
        
        Formula: Precision@K = (# relevant items in top-K) / K
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs (binary: either relevant or not)
            k: Number of top predictions to consider (if None, use all predictions)
            
        Returns:
            Precision score in range [0, 1]
            
        Example:
            predictions = ['job1', 'job2', 'job3', 'job4', 'job5']
            ground_truth = {'job2', 'job4', 'job7'}
            precision_at_k(predictions, ground_truth, k=5) = 2/5 = 0.4
        """
        if not predictions:
            return 0.0
            
        k = k or len(predictions)
        top_k = predictions[:k]
        
        relevant_in_top_k = len([item for item in top_k if item in ground_truth])
        return relevant_in_top_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(
        predictions: List[str],
        ground_truth: Set[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Recall@K: proportion of relevant items that are recommended
        
        Formula: Recall@K = (# relevant items in top-K) / (total # relevant items)
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs
            k: Number of top predictions to consider (if None, use all predictions)
            
        Returns:
            Recall score in range [0, 1]
            
        Example:
            predictions = ['job1', 'job2', 'job3', 'job4', 'job5']
            ground_truth = {'job2', 'job4', 'job7'}
            recall_at_k(predictions, ground_truth, k=5) = 2/3 = 0.667
        """
        if not ground_truth:
            return 0.0
        if not predictions:
            return 0.0
            
        k = k or len(predictions)
        top_k = predictions[:k]
        
        relevant_in_top_k = len([item for item in top_k if item in ground_truth])
        return relevant_in_top_k / len(ground_truth)
    
    @staticmethod
    def f1_at_k(
        predictions: List[str],
        ground_truth: Set[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate F1@K: harmonic mean of Precision@K and Recall@K
        
        Formula: F1@K = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs
            k: Number of top predictions to consider
            
        Returns:
            F1 score in range [0, 1]
        """
        precision = EvaluationMetrics.precision_at_k(predictions, ground_truth, k)
        recall = EvaluationMetrics.recall_at_k(predictions, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(
        predictions: List[str],
        relevance_scores: Dict[str, float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Measures ranking quality with position discount. Higher-ranked relevant items
        contribute more to the score.
        
        Formula:
            DCG@K = Σ(rel_i / log2(i+1)) for i=1 to K
            NDCG@K = DCG@K / IDCG@K
        where IDCG is DCG of the ideal ranking
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            relevance_scores: Dictionary mapping item IDs to relevance scores
                            (e.g., 0=irrelevant, 1=somewhat relevant, 2=highly relevant)
            k: Number of top predictions to consider
            
        Returns:
            NDCG score in range [0, 1]
            
        Example:
            predictions = ['job1', 'job2', 'job3']
            relevance_scores = {'job1': 0, 'job2': 2, 'job3': 1}
            # DCG = 0/log2(2) + 2/log2(3) + 1/log2(4) = 0 + 1.26 + 0.5 = 1.76
            # IDCG (ideal: job2, job3, job1) = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2.63
            # NDCG = 1.76 / 2.63 = 0.67
        """
        if not predictions or not relevance_scores:
            return 0.0
        
        k = k or len(predictions)
        top_k = predictions[:k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item_id in enumerate(top_k, start=1):
            relevance = relevance_scores.get(item_id, 0)
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (Ideal DCG) - sort relevance scores in descending order
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances, start=1):
            idcg += relevance / np.log2(i + 1)
        
        # Return normalized score
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: List[str],
        ground_truth: Set[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank)
        
        Measures the rank position of the FIRST relevant item.
        Useful for scenarios where users care about finding at least one good result quickly.
        
        Formula: MRR = 1 / rank_of_first_relevant_item
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs
            
        Returns:
            MRR score in range [0, 1]
            
        Example:
            predictions = ['job1', 'job2', 'job3', 'job4']
            ground_truth = {'job3', 'job5'}
            # First relevant item (job3) is at rank 3
            # MRR = 1/3 = 0.333
        """
        if not predictions or not ground_truth:
            return 0.0
        
        for rank, item_id in enumerate(predictions, start=1):
            if item_id in ground_truth:
                return 1.0 / rank
        
        return 0.0  # No relevant item found
    
    @staticmethod
    def hit_rate_at_k(
        predictions: List[str],
        ground_truth: Set[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Hit Rate@K (also called Recall@K in some contexts)
        
        Binary metric: 1 if at least one relevant item is in top-K, 0 otherwise.
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs
            k: Number of top predictions to consider
            
        Returns:
            1.0 if hit, 0.0 if miss
            
        Example:
            predictions = ['job1', 'job2', 'job3']
            ground_truth = {'job2', 'job5'}
            hit_rate_at_k(predictions, ground_truth, k=3) = 1.0 (job2 is in top-3)
        """
        if not ground_truth or not predictions:
            return 0.0
        
        k = k or len(predictions)
        top_k = predictions[:k]
        
        # Check if any relevant item is in top-K
        has_hit = any(item in ground_truth for item in top_k)
        return 1.0 if has_hit else 0.0
    
    @staticmethod
    def coverage(
        all_predictions: List[List[str]],
        all_items: Set[str]
    ) -> float:
        """
        Calculate Catalog Coverage: percentage of items that appear in recommendations
        
        Measures diversity - do recommendations cover a wide range of items or 
        are they concentrated on a few popular ones?
        
        Formula: Coverage = (# unique items recommended) / (# total items)
        
        Args:
            all_predictions: List of prediction lists (one per user/candidate)
            all_items: Set of all possible items (entire catalog)
            
        Returns:
            Coverage score in range [0, 1]
            
        Example:
            all_predictions = [['job1', 'job2'], ['job2', 'job3'], ['job1', 'job3']]
            all_items = {'job1', 'job2', 'job3', 'job4', 'job5'}
            # Recommended: {job1, job2, job3} = 3 items
            # Coverage = 3/5 = 0.6
        """
        if not all_items:
            return 0.0
        
        # Collect all unique recommended items
        recommended_items = set()
        for predictions in all_predictions:
            recommended_items.update(predictions)
        
        return len(recommended_items) / len(all_items)
    
    @staticmethod
    def average_precision(
        predictions: List[str],
        ground_truth: Set[str]
    ) -> float:
        """
        Calculate Average Precision (AP) for a single query
        
        AP considers both precision and ranking position of relevant items.
        
        Formula: AP = (Σ Precision@k * rel(k)) / # relevant items
        where rel(k) = 1 if item at rank k is relevant, 0 otherwise
        
        Args:
            predictions: Ordered list of predicted/recommended item IDs
            ground_truth: Set of relevant item IDs
            
        Returns:
            AP score in range [0, 1]
        """
        if not ground_truth or not predictions:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for k, item_id in enumerate(predictions, start=1):
            if item_id in ground_truth:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_sum += precision_at_k
        
        return precision_sum / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    @staticmethod
    def mean_average_precision(
        all_predictions: List[List[str]],
        all_ground_truths: List[Set[str]]
    ) -> float:
        """
        Calculate MAP (Mean Average Precision) across multiple queries
        
        Args:
            all_predictions: List of prediction lists (one per query)
            all_ground_truths: List of ground truth sets (one per query)
            
        Returns:
            MAP score in range [0, 1]
        """
        if not all_predictions or len(all_predictions) != len(all_ground_truths):
            return 0.0
        
        ap_scores = []
        for predictions, ground_truth in zip(all_predictions, all_ground_truths):
            ap = EvaluationMetrics.average_precision(predictions, ground_truth)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0


class AggregatedMetrics:
    """Aggregate metrics across multiple candidates/queries"""
    
    @staticmethod
    def compute_aggregate_metrics(
        all_predictions: List[List[str]],
        all_ground_truths: List[Set[str]],
        all_relevance_scores: Optional[List[Dict[str, float]]] = None,
        k_values: List[int] = [5, 10, 20],
        all_items: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Compute aggregated metrics across all candidates
        
        Args:
            all_predictions: List of prediction lists (one per candidate)
            all_ground_truths: List of ground truth sets (one per candidate)
            all_relevance_scores: Optional list of relevance score dicts for NDCG
            k_values: List of K values to evaluate (default: [5, 10, 20])
            all_items: Optional set of all items for coverage calculation
            
        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}
        
        n_queries = len(all_predictions)
        
        for k in k_values:
            # Precision@K
            precisions = [
                EvaluationMetrics.precision_at_k(preds, gt, k)
                for preds, gt in zip(all_predictions, all_ground_truths)
            ]
            metrics[f'precision@{k}'] = np.mean(precisions)
            
            # Recall@K
            recalls = [
                EvaluationMetrics.recall_at_k(preds, gt, k)
                for preds, gt in zip(all_predictions, all_ground_truths)
            ]
            metrics[f'recall@{k}'] = np.mean(recalls)
            
            # F1@K
            f1_scores = [
                EvaluationMetrics.f1_at_k(preds, gt, k)
                for preds, gt in zip(all_predictions, all_ground_truths)
            ]
            metrics[f'f1@{k}'] = np.mean(f1_scores)
            
            # Hit Rate@K
            hit_rates = [
                EvaluationMetrics.hit_rate_at_k(preds, gt, k)
                for preds, gt in zip(all_predictions, all_ground_truths)
            ]
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates)
            
            # NDCG@K (if relevance scores provided)
            if all_relevance_scores:
                ndcg_scores = [
                    EvaluationMetrics.ndcg_at_k(preds, rel_scores, k)
                    for preds, rel_scores in zip(all_predictions, all_relevance_scores)
                ]
                metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        # MRR (rank-based, not K-dependent)
        mrr_scores = [
            EvaluationMetrics.mean_reciprocal_rank(preds, gt)
            for preds, gt in zip(all_predictions, all_ground_truths)
        ]
        metrics['mrr'] = np.mean(mrr_scores)
        
        # MAP
        metrics['map'] = EvaluationMetrics.mean_average_precision(
            all_predictions, all_ground_truths
        )
        
        # Coverage (if all items provided)
        if all_items:
            metrics['coverage'] = EvaluationMetrics.coverage(all_predictions, all_items)
        
        return metrics


def format_metrics_report(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> str:
    """
    Format metrics dictionary as human-readable report
    
    Args:
        metrics: Dictionary of metric names to scores
        title: Report title
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        f"{title:^70}",
        "=" * 70,
        ""
    ]
    
    # Group metrics by category
    categories = {
        'Precision': [k for k in metrics if k.startswith('precision')],
        'Recall': [k for k in metrics if k.startswith('recall')],
        'F1 Score': [k for k in metrics if k.startswith('f1')],
        'NDCG': [k for k in metrics if k.startswith('ndcg')],
        'Hit Rate': [k for k in metrics if k.startswith('hit_rate')],
        'Ranking': ['mrr', 'map'],
        'Diversity': ['coverage']
    }
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            lines.append(f"\n{category}:")
            lines.append("-" * 40)
            for name, value in sorted(category_metrics.items()):
                lines.append(f"  {name:.<30} {value:.4f}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Example usage and unit tests
    print("Running evaluation metrics examples...\n")
    
    # Example 1: Binary relevance
    predictions = ['job1', 'job2', 'job3', 'job4', 'job5']
    ground_truth = {'job2', 'job4', 'job7'}
    
    print("Example 1: Binary Relevance")
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Precision@5: {EvaluationMetrics.precision_at_k(predictions, ground_truth, 5):.3f}")
    print(f"Recall@5: {EvaluationMetrics.recall_at_k(predictions, ground_truth, 5):.3f}")
    print(f"F1@5: {EvaluationMetrics.f1_at_k(predictions, ground_truth, 5):.3f}")
    print(f"MRR: {EvaluationMetrics.mean_reciprocal_rank(predictions, ground_truth):.3f}")
    print(f"Hit Rate@5: {EvaluationMetrics.hit_rate_at_k(predictions, ground_truth, 5):.3f}\n")
    
    # Example 2: Graded relevance (NDCG)
    relevance_scores = {'job1': 0, 'job2': 2, 'job3': 1, 'job4': 2, 'job5': 0, 'job7': 2}
    print("Example 2: Graded Relevance (NDCG)")
    print(f"Predictions: {predictions}")
    print(f"Relevance Scores: {relevance_scores}")
    print(f"NDCG@5: {EvaluationMetrics.ndcg_at_k(predictions, relevance_scores, 5):.3f}\n")
    
    # Example 3: Aggregated metrics
    all_predictions = [
        ['job1', 'job2', 'job3'],
        ['job4', 'job5', 'job6'],
        ['job2', 'job7', 'job8']
    ]
    all_ground_truths = [
        {'job2', 'job9'},
        {'job4', 'job10'},
        {'job7', 'job11'}
    ]
    all_items = {f'job{i}' for i in range(1, 12)}
    
    print("Example 3: Aggregated Metrics")
    metrics = AggregatedMetrics.compute_aggregate_metrics(
        all_predictions, all_ground_truths, k_values=[3, 5], all_items=all_items
    )
    print(format_metrics_report(metrics, "Test Evaluation Results"))
