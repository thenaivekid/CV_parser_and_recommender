#!/usr/bin/env python3
"""
Job Processing Script
Loads job descriptions from JSON files, generates embeddings, and stores in database
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_generator import EmbeddingGenerator
from src.database_manager import DatabaseManager
from src.config import config
from src.performance_monitor import get_monitor, set_monitor, PerformanceMonitor, track_time
from src.dashboard_generator import DashboardGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('job_processing.log')
    ]
)
logger = logging.getLogger(__name__)


class JobProcessor:
    """Process job descriptions and store in database"""
    
    def __init__(self, embedding_gen: EmbeddingGenerator, db_manager: DatabaseManager):
        """
        Initialize job processor
        
        Args:
            embedding_gen: EmbeddingGenerator instance
            db_manager: DatabaseManager instance
        """
        self.embedding_gen = embedding_gen
        self.db_manager = db_manager
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def _create_job_text(self, job_json: Dict[str, Any]) -> str:
        """
        Create text representation of job for embedding generation
        
        Args:
            job_json: Parsed job data
            
        Returns:
            Concatenated text representation
        """
        parts = []
        
        # Job title and company
        if job_json.get('job_title'):
            parts.append(f"Title: {job_json['job_title']}")
        if job_json.get('company'):
            parts.append(f"Company: {job_json['company']}")
        
        # Description and responsibilities
        if job_json.get('description'):
            parts.append(f"Description: {job_json['description']}")
        if job_json.get('responsibilities'):
            parts.append(f"Responsibilities: {job_json['responsibilities']}")
        
        # Skills
        tech_skills = job_json.get('skills_technical', [])
        if tech_skills:
            parts.append(f"Technical Skills: {', '.join(tech_skills)}")
        
        soft_skills = job_json.get('skills_soft', [])
        if soft_skills:
            parts.append(f"Soft Skills: {', '.join(soft_skills)}")
        
        # Experience and education
        if job_json.get('seniority_level'):
            parts.append(f"Seniority: {job_json['seniority_level']}")
        if job_json.get('education_required'):
            parts.append(f"Education: {job_json['education_required']}")
        if job_json.get('education_field'):
            parts.append(f"Field: {job_json['education_field']}")
        
        # Certifications
        certifications = job_json.get('certifications', [])
        if certifications:
            parts.append(f"Certifications: {', '.join(certifications)}")
        
        return " | ".join(parts) if parts else "No information available"
    
    def process_job(self, job_file: Path, skip_existing: bool = True) -> bool:
        """
        Process a single job file
        
        Args:
            job_file: Path to job JSON file
            skip_existing: Skip if job already exists in database
            
        Returns:
            True if successful, False otherwise
        """
        job_id = job_file.stem  # Use filename without extension as job_id
        
        try:
            # Check if already exists
            if skip_existing and self.db_manager.job_exists(job_id):
                logger.info(f"⏭️  Skipping existing job: {job_id}")
                self.stats['skipped'] += 1
                return True
            
            # Load job JSON
            logger.info(f"📄 Processing job: {job_id}")
            with open(job_file, 'r', encoding='utf-8') as f:
                job_json = json.load(f)
            
            # Create job text for embedding
            job_text = self._create_job_text(job_json)
            
            # Generate embedding with timing
            logger.debug(f"  → Generating embedding for {job_id}, text length: {len(job_text)}")
            with track_time('job_embedding_generation', entity_id=job_id):
                embedding_result = self.embedding_gen.generate_embedding(job_text)
                embedding = embedding_result
            
            # Store in database with timing
            logger.debug(f"  → Storing job {job_id} in database")
            with track_time('job_db_insert', entity_id=job_id):
                if not self.db_manager.insert_job(job_id, job_json):
                    logger.error(f"  ❌ Failed to insert job: {job_id}")
                    self.stats['failed'] += 1
                    return False
            
            logger.debug(f"  → Storing embedding for {job_id}")
            with track_time('job_embedding_db_insert', entity_id=job_id):
                if not self.db_manager.insert_job_embedding(
                    job_id, 
                    embedding, 
                    self.embedding_gen.model_name
                ):
                    logger.error(f"  ❌ Failed to insert embedding: {job_id}")
                    self.stats['failed'] += 1
                    return False
            
            logger.info(f"  ✓ Successfully processed: {job_id}")
            self.stats['success'] += 1
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error processing job {job_id}: {e}")
            logger.debug("Exception details:", exc_info=True)
            self.stats['failed'] += 1
            return False
    
    def process_all_jobs(self, jobs_path: Path, skip_existing: bool = True):
        """
        Process all job JSON files in directory
        
        Args:
            jobs_path: Path to directory containing job JSON files
            skip_existing: Skip existing jobs
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing jobs from: {jobs_path}")
        logger.info(f"Skip existing: {skip_existing}")
        logger.info(f"{'='*60}\n")
        
        # Find all JSON files
        job_files = list(jobs_path.glob("*.json"))
        
        if not job_files:
            logger.warning(f"No JSON files found in {jobs_path}")
            return
        
        self.stats['total'] = len(job_files)
        logger.info(f"Found {len(job_files)} job files\n")
        
        # Process each job
        for i, job_file in enumerate(job_files, 1):
            logger.info(f"[{i}/{len(job_files)}] Processing: {job_file.name}")
            self.process_job(job_file, skip_existing)
            logger.info("")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary"""
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Jobs:      {self.stats['total']}")
        logger.info(f"✓ Success:       {self.stats['success']}")
        logger.info(f"⏭️  Skipped:       {self.stats['skipped']}")
        logger.info(f"❌ Failed:        {self.stats['failed']}")
        logger.info(f"{'='*60}\n")


def main():
    """Main entry point for job processing"""
    parser = argparse.ArgumentParser(
        description="Process job descriptions: Generate Embeddings and Store in Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process all jobs
  %(prog)s --force                  # Reprocess existing jobs
  %(prog)s --verbose                # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess existing jobs (default: skip existing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    parser.add_argument(
        '--jobs-path',
        type=Path,
        default=None,
        help=f'Path to jobs directory (default: {config.job_base_path})'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("="*60)
        logger.info("JOB PROCESSING - EMBEDDING GENERATION")
        logger.info("="*60)
        logger.info(f"Embedding Model: {config.embd_model}")
        logger.info(f"Database: {config.database['database']}")
        logger.info(f"Skip Existing: {not args.force}")
        logger.info("="*60)
        
        # Initialize components
        logger.info("\n📦 Initializing components...")
        
        logger.info("  → Loading embedding model...")
        embedding_gen = EmbeddingGenerator(model_name=config.embd_model)
        
        logger.info("  → Connecting to database...")
        db_manager = DatabaseManager(config.database)
        
        # Initialize performance monitor with database connection
        monitor = PerformanceMonitor(db_manager)
        set_monitor(monitor)
        
        # Determine jobs path
        jobs_path = args.jobs_path or config.job_base_path
        
        if not jobs_path.exists():
            logger.error(f"Jobs path does not exist: {jobs_path}")
            return 1
        
        # Start processing session to track THIS run's metrics
        session_metadata = {
            'force': args.force,
            'jobs_path': str(jobs_path)
        }
        monitor.start_session('job_processing', metadata=session_metadata)
        
        logger.info("✓ All components initialized\n")
        
        # Create processor
        processor = JobProcessor(embedding_gen, db_manager)
        
        # Process jobs
        skip_existing = not args.force
        processor.process_all_jobs(jobs_path, skip_existing)
        
        # End processing session with actual stats from this run
        monitor.end_session(
            items_processed=processor.stats['total'],
            items_success=processor.stats['success'],
            items_failed=processor.stats['failed'],
            items_skipped=processor.stats['skipped']
        )
        
        # Cleanup
        db_manager.close()
        
        logger.info("✅ Processing completed successfully!")
        
        # Generate performance dashboard
        logger.info("\n📊 Generating performance dashboard...")
        try:
            output_dir = Path("data/performance_reports")
            dashboard_gen = DashboardGenerator()
            report = dashboard_gen.generate_report(output_dir)
            logger.info(f"✓ Performance dashboard saved to {output_dir}")
            logger.info(f"  View: {output_dir}/performance_dashboard_*.html")
        except Exception as e:
            logger.warning(f"Could not generate dashboard: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Processing interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}")
        logger.debug("Exception details:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
