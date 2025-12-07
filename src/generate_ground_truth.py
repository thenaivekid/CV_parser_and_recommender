"""
Ground Truth Generator for Recommendation Evaluation
Creates labeled data using profession-based heuristic matching
"""
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import random

from src.config import Config
from src.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """
    Generate ground truth labels for candidate-job pairs using profession-based heuristics.
    
    Relevance Scoring:
    - 2 (Highly Relevant): Exact profession match (ACCOUNTANT candidate → ACCOUNTANT job)
    - 1 (Somewhat Relevant): Related profession (ACCOUNTANT → FINANCE, HR → hr_recruiter)
    - 0 (Irrelevant): Unrelated profession (ACCOUNTANT → CHEF)
    """
    
    # Define profession mappings for relevance scoring
    PROFESSION_MAPPINGS = {
        'ACCOUNTANT': {
            'highly_relevant': ['ACCOUNTANT', 'senior_accountant', 'junior_accountant'],
            'somewhat_relevant': ['FINANCE', 'financial_analyst', 'finance_manager']
        },
        'FINANCE': {
            'highly_relevant': ['FINANCE', 'financial_analyst', 'finance_manager'],
            'somewhat_relevant': ['ACCOUNTANT', 'senior_accountant', 'junior_accountant']
        },
        'HR': {
            'highly_relevant': ['HR', 'hr_recruiter', 'hr_business_partner'],
            'somewhat_relevant': []
        },
        'INFORMATION-TECHNOLOGY': {
            'highly_relevant': ['it_officer_bank', 'software_engineer_backend', 'data_scientist_nlp'],
            'somewhat_relevant': []
        },
        'DESIGNER': {
            'highly_relevant': ['DESIGNER', 'graphic_designer', 'ux_ui_designer'],
            'somewhat_relevant': []
        },
        'ADVOCATE': {
            'highly_relevant': ['ADVOCATE', 'corporate_lawyer', 'litigation_lawyer'],
            'somewhat_relevant': []
        },
        'TEACHER': {
            'highly_relevant': ['english_teacher'],
            'somewhat_relevant': []
        },
        'CHEF': {
            'highly_relevant': ['chef_for_kitchen', 'restaurant_manager'],
            'somewhat_relevant': []
        },
        'FITNESS': {
            'highly_relevant': ['fitness_coach'],
            'somewhat_relevant': []
        },
        'ARTS': {
            'highly_relevant': ['art_collection_helper'],
            'somewhat_relevant': ['DESIGNER', 'graphic_designer', 'ux_ui_designer']
        }
    }
    
    def __init__(self, db_manager: DatabaseManager, output_dir: str = "data/evaluation"):
        """
        Initialize ground truth generator
        
        Args:
            db_manager: Database manager instance
            output_dir: Directory to save ground truth files
        """
        self.db = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_relevance_score(self, candidate_profession: str, job_id: str) -> int:
        """
        Calculate relevance score for candidate-job pair
        
        Args:
            candidate_profession: Candidate's profession (e.g., 'ACCOUNTANT')
            job_id: Job identifier (e.g., 'senior_accountant')
            
        Returns:
            0 (irrelevant), 1 (somewhat relevant), or 2 (highly relevant)
        """
        if not candidate_profession or candidate_profession not in self.PROFESSION_MAPPINGS:
            return 0
            
        mapping = self.PROFESSION_MAPPINGS[candidate_profession]
        
        # Normalize job_id for comparison
        job_id_normalized = job_id.strip().lower()
        
        # Check highly relevant
        for relevant_job in mapping['highly_relevant']:
            if relevant_job.lower() in job_id_normalized or job_id_normalized in relevant_job.lower():
                return 2
                
        # Check somewhat relevant
        for relevant_job in mapping['somewhat_relevant']:
            if relevant_job.lower() in job_id_normalized or job_id_normalized in relevant_job.lower():
                return 1
                
        return 0
    
    def fetch_all_candidates(self) -> List[Tuple[str, str]]:
        """
        Fetch all candidates with their professions
        
        Returns:
            List of (candidate_id, profession) tuples
        """
        try:
            self.db.cursor.execute("""
                SELECT candidate_id, profession 
                FROM candidates 
                WHERE profession IS NOT NULL
                ORDER BY candidate_id
            """)
            return self.db.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching candidates: {e}")
            return []
    
    def fetch_all_jobs(self) -> List[str]:
        """
        Fetch all job IDs
        
        Returns:
            List of job_id strings
        """
        try:
            self.db.cursor.execute("SELECT job_id FROM jobs ORDER BY job_id")
            return [row[0] for row in self.db.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching jobs: {e}")
            return []
    
    def generate_ground_truth(
        self, 
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        include_all_pairs: bool = False,
        min_relevant_per_candidate: int = 1
    ) -> Dict[str, List[Dict]]:
        """
        Generate ground truth dataset with train/val/test splits
        
        Args:
            train_ratio: Proportion for training set (default: 0.6)
            val_ratio: Proportion for validation set (default: 0.2)
            test_ratio: Proportion for test set (default: 0.2)
            include_all_pairs: If True, include ALL candidate-job pairs (can be huge!)
                              If False, include only relevant pairs + random sample of irrelevant
            min_relevant_per_candidate: Minimum relevant jobs to include per candidate
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        logger.info("Generating ground truth dataset...")
        
        candidates = self.fetch_all_candidates()
        jobs = self.fetch_all_jobs()
        
        logger.info(f"Found {len(candidates)} candidates and {len(jobs)} jobs")
        
        # Generate all labeled pairs
        ground_truth_data = []
        relevant_count = 0
        somewhat_relevant_count = 0
        irrelevant_count = 0
        
        for candidate_id, profession in candidates:
            candidate_relevant_jobs = []
            
            for job_id in jobs:
                relevance = self.get_relevance_score(profession, job_id)
                
                # Always include relevant pairs
                if relevance > 0:
                    ground_truth_data.append({
                        'candidate_id': candidate_id,
                        'job_id': job_id,
                        'relevance': relevance,
                        'candidate_profession': profession,
                        'reason': f"Profession-based match: {profession} -> {job_id}"
                    })
                    candidate_relevant_jobs.append(job_id)
                    
                    if relevance == 2:
                        relevant_count += 1
                    else:
                        somewhat_relevant_count += 1
                        
                # Optionally include irrelevant pairs
                elif include_all_pairs:
                    ground_truth_data.append({
                        'candidate_id': candidate_id,
                        'job_id': job_id,
                        'relevance': 0,
                        'candidate_profession': profession,
                        'reason': f"No profession match: {profession} -> {job_id}"
                    })
                    irrelevant_count += 1
            
            # If not including all pairs, sample some irrelevant pairs for balance
            if not include_all_pairs:
                irrelevant_jobs = [j for j in jobs if j not in candidate_relevant_jobs]
                # Sample 2x the number of relevant jobs as irrelevant (for class balance)
                sample_size = min(len(irrelevant_jobs), max(2 * len(candidate_relevant_jobs), 3))
                sampled_irrelevant = random.sample(irrelevant_jobs, sample_size)
                
                for job_id in sampled_irrelevant:
                    ground_truth_data.append({
                        'candidate_id': candidate_id,
                        'job_id': job_id,
                        'relevance': 0,
                        'candidate_profession': profession,
                        'reason': f"Sampled irrelevant: {profession} -> {job_id}"
                    })
                    irrelevant_count += 1
        
        logger.info(f"Generated {len(ground_truth_data)} labeled pairs:")
        logger.info(f"  - Highly relevant (2): {relevant_count}")
        logger.info(f"  - Somewhat relevant (1): {somewhat_relevant_count}")
        logger.info(f"  - Irrelevant (0): {irrelevant_count}")
        
        # Shuffle and split by candidates (not by pairs) to avoid data leakage
        random.shuffle(candidates)
        
        n_candidates = len(candidates)
        train_end = int(n_candidates * train_ratio)
        val_end = train_end + int(n_candidates * val_ratio)
        
        train_candidates = {c[0] for c in candidates[:train_end]}
        val_candidates = {c[0] for c in candidates[train_end:val_end]}
        test_candidates = {c[0] for c in candidates[val_end:]}
        
        # Split data
        splits = {
            'train': [d for d in ground_truth_data if d['candidate_id'] in train_candidates],
            'val': [d for d in ground_truth_data if d['candidate_id'] in val_candidates],
            'test': [d for d in ground_truth_data if d['candidate_id'] in test_candidates]
        }
        
        logger.info(f"Split sizes:")
        logger.info(f"  - Train: {len(splits['train'])} pairs ({len(train_candidates)} candidates)")
        logger.info(f"  - Val: {len(splits['val'])} pairs ({len(val_candidates)} candidates)")
        logger.info(f"  - Test: {len(splits['test'])} pairs ({len(test_candidates)} candidates)")
        
        return splits
    
    def save_ground_truth(self, splits: Dict[str, List[Dict]], format: str = 'both') -> Tuple[Path, Path]:
        """
        Save ground truth to CSV and JSON files
        
        Args:
            splits: Dictionary with train/val/test splits
            format: 'csv', 'json', or 'both'
            
        Returns:
            Tuple of (csv_path, json_path)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        csv_path = None
        json_path = None
        
        # Save as CSV
        if format in ['csv', 'both']:
            csv_path = self.output_dir / f'ground_truth_{timestamp}.csv'
            
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['split', 'candidate_id', 'job_id', 'relevance', 'candidate_profession', 'reason']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for split_name, data in splits.items():
                    for row in data:
                        writer.writerow({'split': split_name, **row})
            
            logger.info(f"Saved ground truth CSV to: {csv_path}")
        
        # Save as JSON
        if format in ['json', 'both']:
            json_path = self.output_dir / f'ground_truth_{timestamp}.json'
            
            output_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_pairs': sum(len(v) for v in splits.values()),
                    'splits': {k: len(v) for k, v in splits.items()},
                    'profession_mappings': self.PROFESSION_MAPPINGS
                },
                'data': splits
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved ground truth JSON to: {json_path}")
        
        # Also save latest symlink
        if csv_path:
            latest_csv = self.output_dir / 'ground_truth_latest.csv'
            if latest_csv.exists():
                latest_csv.unlink()
            latest_csv.symlink_to(csv_path.name)
        
        if json_path:
            latest_json = self.output_dir / 'ground_truth_latest.json'
            if latest_json.exists():
                latest_json.unlink()
            latest_json.symlink_to(json_path.name)
        
        return csv_path, json_path


def main():
    """Generate ground truth dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ground truth dataset for evaluation')
    parser.add_argument('--config', default='configurations/config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', default='data/evaluation', help='Output directory')
    parser.add_argument('--all-pairs', action='store_true', help='Include ALL candidate-job pairs (WARNING: can be huge!)')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='both', help='Output format')
    parser.add_argument('--train-ratio', type=float, default=0.6, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        logger.error("Train/val/test ratios must sum to 1.0")
        return
    
    # Load config
    config = Config(args.config)
    
    # Initialize database
    db_manager = DatabaseManager(config.database)
    
    # Generate ground truth
    generator = GroundTruthGenerator(db_manager, args.output_dir)
    splits = generator.generate_ground_truth(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        include_all_pairs=args.all_pairs
    )
    
    # Save to files
    csv_path, json_path = generator.save_ground_truth(splits, format=args.format)
    
    logger.info("=" * 60)
    logger.info("Ground truth generation complete!")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {json_path}")
    logger.info("=" * 60)
    
    # Close database connection
    db_manager.close()


if __name__ == '__main__':
    main()
