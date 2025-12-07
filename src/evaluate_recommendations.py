"""
Evaluate Recommendations: Precision, Recall, F1
Compare DB recommendations against ground truth JSON
Supports batch processing and parallel execution for production scale
"""
import logging
import json
import numpy as np
from typing import Dict, List, Set, Iterator
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.config import Config
from src.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_iterator(items: List, batch_size: int) -> Iterator[List]:
    """Yield successive batches from items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_ground_truth(json_path: str) -> Dict[str, Set[str]]:
    """Load ground truth: candidate_id → set of job_ids"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt = {}
    for candidate in data:
        candidate_id = candidate['candidate_id']
        relevant_jobs = {rec['job_id'] for rec in candidate['recommendations']}
        gt[candidate_id] = relevant_jobs
    
    logger.info(f"Loaded ground truth for {len(gt)} candidates")
    return gt


def fetch_recommendations(db_manager: DatabaseManager, candidate_ids: List[str], batch_size: int = 100) -> Dict[str, List[str]]:
    """
    Fetch recommendations from DB for each candidate using batched queries
    Uses IN clause to reduce DB round trips
    """
    logger.info(f"Fetching recommendations for {len(candidate_ids)} candidates in batches of {batch_size}...")
    
    all_recs = {}
    batches = list(batch_iterator(candidate_ids, batch_size))
    
    for batch_num, batch in enumerate(batches, 1):
        placeholders = ','.join(['%s'] * len(batch))
        
        # Fetch all recommendations for this batch in one query
        db_manager.cursor.execute(f"""
            SELECT r.candidate_id, r.job_id, r.match_score
            FROM recommendations r
            WHERE r.candidate_id IN ({placeholders})
            ORDER BY r.candidate_id, r.match_score DESC
        """, batch)
        
        # Group by candidate and keep top-20 per candidate
        current_cid = None
        current_jobs = []
        
        for cid, job_id, score in db_manager.cursor.fetchall():
            if cid != current_cid:
                if current_cid:
                    all_recs[current_cid] = current_jobs
                current_cid = cid
                current_jobs = [job_id]
            else:
                current_jobs.append(job_id)
        
        # Don't forget the last candidate
        if current_cid:
            all_recs[current_cid] = current_jobs
        
        logger.info(f"Batch {batch_num}/{len(batches)} complete ({len(batch)} candidates)")
    
    logger.info(f"Fetched recommendations for {len(all_recs)} candidates")
    return all_recs


def evaluate_batch(batch_ids: List[str], predictions: Dict[str, List[str]], 
                   ground_truth: Dict[str, Set[str]], top_k_values: List[int]) -> Dict:
    """
    Evaluate a single batch of candidates (designed for parallel execution)
    Returns per-batch results that can be aggregated
    """
    batch_results = {k: {'precision': [], 'recall': [], 'f1': []} for k in top_k_values}
    
    for candidate_id in batch_ids:
        if candidate_id not in predictions or candidate_id not in ground_truth:
            continue
        
        preds = predictions[candidate_id]
        gt = ground_truth[candidate_id]
        
        if not gt:
            continue
        
        for k in top_k_values:
            top_k_preds = preds[:k]
            relevant_in_topk = sum(1 for job in top_k_preds if job in gt)
            
            precision = relevant_in_topk / k if k > 0 else 0
            recall = relevant_in_topk / len(gt) if len(gt) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            batch_results[k]['precision'].append(precision)
            batch_results[k]['recall'].append(recall)
            batch_results[k]['f1'].append(f1)
    
    return batch_results


def evaluate(predictions: Dict[str, List[str]], ground_truth: Dict[str, Set[str]], 
             top_k_values: List[int], batch_size: int = 100, max_workers: int = 4):
    """
    Calculate Precision@topK, Recall@topK, F1@topK using parallel batch processing
    
    Args:
        predictions: candidate_id → list of recommended job_ids
        ground_truth: candidate_id → set of relevant job_ids
        top_k_values: List of K values to evaluate
        batch_size: Number of candidates per batch
        max_workers: Number of parallel processes
    
    Returns:
        Dictionary of averaged metrics
    """
    candidate_ids = list(ground_truth.keys())
    batches = list(batch_iterator(candidate_ids, batch_size))
    
    logger.info(f"Evaluating {len(candidate_ids)} candidates in {len(batches)} batches using {max_workers} workers")
    
    # Aggregate results across all batches
    all_results = {k: {'precision': [], 'recall': [], 'f1': []} for k in top_k_values}
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_batch, batch, predictions, ground_truth, top_k_values): i
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                batch_results = future.result()
                
                # Merge batch results into global results
                for k in top_k_values:
                    all_results[k]['precision'].extend(batch_results[k]['precision'])
                    all_results[k]['recall'].extend(batch_results[k]['recall'])
                    all_results[k]['f1'].extend(batch_results[k]['f1'])
                
                logger.info(f"Batch {batch_num + 1}/{len(batches)} evaluated")
            except Exception as e:
                logger.error(f"Batch {batch_num + 1} failed: {e}")
    
    # Calculate averages
    avg_results = {}
    for k in top_k_values:
        avg_results[f'precision@top{k}'] = float(np.mean(all_results[k]['precision'])) if all_results[k]['precision'] else 0.0
        avg_results[f'recall@top{k}'] = float(np.mean(all_results[k]['recall'])) if all_results[k]['recall'] else 0.0
        avg_results[f'f1@top{k}'] = float(np.mean(all_results[k]['f1'])) if all_results[k]['f1'] else 0.0
    
    return avg_results


def print_results(metrics: Dict):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("RECOMMENDATION QUALITY EVALUATION".center(60))
    print("=" * 60 + "\n")
    
    for metric_name in sorted(metrics.keys()):
        value = metrics[metric_name]
        print(f"  {metric_name:.<35} {value:.4f}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate recommendations with parallel batch processing')
    parser.add_argument('--config', default='configurations/config.yaml', help='Config file')
    parser.add_argument('--ground-truth', default='data/evaluation/ground_truth.json', help='Ground truth JSON')
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 5, 10, 20], help='Top-K values')
    parser.add_argument('--output', default='data/performance_reports/evaluation.json', help='Output JSON')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Load config and DB
    config = Config(args.config)
    db_manager = DatabaseManager(config.database)
    
    # Load ground truth
    logger.info("Loading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    
    # Fetch recommendations from DB (batched queries)
    recs = fetch_recommendations(db_manager, list(gt.keys()), batch_size=args.batch_size)
    
    db_manager.close()
    
    # Evaluate (parallel batch processing)
    logger.info("Starting evaluation...")
    metrics = evaluate(recs, gt, args.top_k, batch_size=args.batch_size, max_workers=args.workers)
    
    # Print results
    print_results(metrics)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        'evaluation_date': datetime.now().isoformat(),
        'num_candidates': len(gt),
        'top_k_values': args.top_k,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
