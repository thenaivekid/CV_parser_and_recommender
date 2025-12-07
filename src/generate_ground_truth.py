"""
Generate Ground Truth JSON for Evaluation
Simple format: Each candidate has list of relevant job_ids (ordered by relevance)
"""
import logging
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from src.config import Config
from src.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple mapping: profession → relevant job_ids
RELEVANT_JOBS = {
    'ACCOUNTANT': ['ACCOUNTANT', 'senior_accountant', 'junior_accountant', 'financial_analyst', 'finance_manager'],
    'FINANCE': ['FINANCE', 'financial_analyst', 'finance_manager', 'senior_accountant'],
    'HR': ['HR', 'hr_recruiter', 'hr_business_partner'],
    'INFORMATION-TECHNOLOGY': ['it_officer_bank', 'software_engineer_backend', 'data_scientist_nlp'],
    'DESIGNER': ['DESIGNER', 'graphic_designer', 'ux_ui_designer'],
    'ADVOCATE': ['ADVOCATE', 'corporate_lawyer', 'litigation_lawyer'],
    'TEACHER': ['english_teacher'],
    'CHEF': ['chef_for_kitchen', 'restaurant_manager'],
    'FITNESS': ['fitness_coach'],
    'ARTS': ['art_collection_helper', 'graphic_designer', 'ux_ui_designer']
}


def generate_ground_truth(db_manager: DatabaseManager, output_path: str = "data/evaluation/ground_truth.json"):
    """
    Generate ground truth JSON with ALL jobs for each candidate
    Format: Relevant jobs first, then irrelevant jobs
    """
    logger.info("Generating ground truth JSON...")
    
    # Get all candidates
    db_manager.cursor.execute("""
        SELECT candidate_id, name, profession 
        FROM candidates 
        WHERE profession IS NOT NULL
        ORDER BY candidate_id
    """)
    candidates = db_manager.cursor.fetchall()
    
    # Get all jobs
    db_manager.cursor.execute("SELECT job_id, job_title FROM jobs ORDER BY job_id")
    all_jobs = db_manager.cursor.fetchall()
    
    logger.info(f"Found {len(candidates)} candidates and {len(all_jobs)} jobs")
    
    ground_truth = []
    
    for candidate_id, candidate_name, profession in candidates:
        # Get relevant jobs for this profession
        relevant = RELEVANT_JOBS.get(profession, [])
        
        # Build recommendations list: relevant first, then irrelevant
        recommendations = []
        
        # Add relevant jobs first (in order)
        for job_id, job_title in all_jobs:
            if job_id in relevant:
                recommendations.append({
                    "job_id": job_id,
                    "job_title": job_title,
                })
        
        
        ground_truth.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "profession": profession,
            "recommendations": recommendations
        })
        
    
    # Save JSON
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    logger.info(f"✓ Saved ground truth to: {output_path}")
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ground truth JSON')
    parser.add_argument('--config', default='configurations/config.yaml', help='Config file')
    parser.add_argument('--output', default='data/evaluation/ground_truth.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    config = Config(args.config)
    db_manager = DatabaseManager(config.database)
    
    generate_ground_truth(db_manager, args.output)
    
    db_manager.close()
    logger.info("=" * 60)
    logger.info("Ground truth generation complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
