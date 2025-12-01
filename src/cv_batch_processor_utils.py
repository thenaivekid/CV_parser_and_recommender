"""
Batch Processor for CV Parser and Recommender System
Orchestrates parsing, embedding generation, and database storage with multiprocessing
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

from cv_parser import CVParser
from src.embedding_generator import EmbeddingGenerator
from src.database_manager import DatabaseManager
from src.config import config
from src.performance_monitor import get_monitor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch process resumes with multiprocessing support"""
    
    def __init__(
        self,
        parser: CVParser,
        embedding_gen: EmbeddingGenerator,
        db_manager: DatabaseManager
    ):
        """
        Initialize batch processor
        
        Args:
            parser: Resume parser instance
            embedding_gen: Embedding generator instance
            db_manager: Database manager instance
        """
        self.parser = parser
        self.embedding_gen = embedding_gen
        self.db_manager = db_manager
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def get_resume_files(self, profession: str, count: int) -> List[Path]:
        """
        Get list of PDF files for a profession
        
        Args:
            profession: Profession category
            count: Number of files to retrieve
            
        Returns:
            List of Path objects for PDF files
        """
        profession_path = config.resume_base_path / profession
        
        if not profession_path.exists():
            logger.error(f"Profession directory not found: {profession_path}")
            return []
        
        pdf_files = sorted(profession_path.glob('*.pdf'))[:count]
        logger.info(f"Found {len(pdf_files)} PDF files in {profession}")
        return pdf_files
    
    def process_single_resume(
        self, 
        pdf_path: Path, 
        profession: str,
        skip_existing: bool = True
    ) -> Tuple[bool, str]:
        """
        Process a single resume: parse, embed, and store
        
        Args:
            pdf_path: Path to PDF file
            profession: Profession category
            skip_existing: Skip if already in database
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        candidate_id = f"{profession}_{pdf_path.stem}"
        
        try:
            # Check if already exists
            if skip_existing and self.db_manager.candidate_exists(candidate_id):
                logger.info(f"Skipping {candidate_id} - already exists")
                return (False, f"Skipped (exists): {candidate_id}")
            
            # Step 1: Parse PDF
            logger.info(f"Parsing: {pdf_path.name}")
            resume_json = self.parser.parse_cv(str(pdf_path))
            
            if not resume_json:
                logger.warning(f"Failed to parse or empty resume: {pdf_path.name}")
                return (False, f"Failed (empty): {candidate_id}")
            
            if not resume_json.get('basics', {}).get('name'):
                logger.warning(f"Parsed resume missing candidate name: {pdf_path.name}")
            # Step 2: Generate embedding
            logger.info(f"Generating embedding for: {candidate_id}")
            embedding_text, embedding_vector = self.embedding_gen.process_resume(resume_json)
            
            # Step 3: Store in database
            logger.info(f"Storing candidate: {candidate_id}")
            candidate_success = self.db_manager.insert_candidate(
                candidate_id, 
                resume_json, 
                profession
            )
            
            if not candidate_success:
                return (False, f"Failed (DB candidate): {candidate_id}")
            
            embedding_success = self.db_manager.insert_embedding(
                candidate_id,
                embedding_vector,
                config.embd_model
            )
            
            if not embedding_success:
                return (False, f"Failed (DB embedding): {candidate_id}")
            
            logger.info(f"✓ Successfully processed: {candidate_id}")
            return (True, f"Success: {candidate_id}")
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return (False, f"Failed (exception): {candidate_id}")
    
    def process_profession(
        self, 
        profession: str, 
        count: int,
        skip_existing: bool = True
    ) -> dict:
        """
        Process all resumes for a profession
        
        Args:
            profession: Profession category
            count: Number of resumes to process
            skip_existing: Skip if already in database
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing profession: {profession}")
        logger.info(f"{'='*60}")
        
        # Get PDF files
        pdf_files = self.get_resume_files(profession, count)
        
        if not pdf_files:
            logger.warning(f"No PDF files found for {profession}")
            return {'success': 0, 'failed': 0, 'skipped': 0, 'total': 0}
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total': len(pdf_files)}
        
        # Get performance monitor for system snapshots
        monitor = get_monitor()
        
        # Process each file with progress bar
        for idx, pdf_path in enumerate(tqdm(pdf_files, desc=f"{profession}", unit="resume"), 1):
            success, message = self.process_single_resume(pdf_path, profession, skip_existing)
            
            if success:
                stats['success'] += 1
            elif 'Skipped' in message:
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
            
            # Record system snapshot every 10 CVs
            if idx % 2 == 0:
                monitor.record_system_snapshot(
                    active_workers=1,
                    throughput_per_min=(stats['success'] / idx) * 60 if idx > 0 else 0.0
                )
        
        logger.info(f"\n{profession} Summary:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        
        return stats
    
    def process_all_professions(self, skip_existing: bool = True):
        """
        Process all configured professions sequentially
        
        Args:
            skip_existing: Skip if already in database
        """
        logger.info("\n" + "="*60)
        logger.info("Starting batch processing of all professions")
        logger.info("="*60)
        
        overall_stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total': 0}
        
        for profession in config.professions:
            stats = self.process_profession(
                profession, 
                config.num_cv_per_profession,
                skip_existing
            )
            
            overall_stats['success'] += stats['success']
            overall_stats['failed'] += stats['failed']
            overall_stats['skipped'] += stats['skipped']
            overall_stats['total'] += stats['total']
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("OVERALL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Resumes: {overall_stats['total']}")
        logger.info(f"Successfully Processed: {overall_stats['success']}")
        logger.info(f"Failed: {overall_stats['failed']}")
        logger.info(f"Skipped (existing): {overall_stats['skipped']}")
        
        # Database statistics
        total_in_db = self.db_manager.get_candidate_count()
        profession_counts = self.db_manager.get_profession_counts()
        
        logger.info(f"\nDatabase Statistics:")
        logger.info(f"Total Candidates: {total_in_db}")
        logger.info(f"By Profession:")
        for prof, count in profession_counts.items():
            logger.info(f"  {prof}: {count}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Batch processing complete!")
        logger.info("="*60)


def _process_resume_worker(args: Tuple[Path, str, dict, str, str]) -> Tuple[bool, str]:
    """
    Worker function for multiprocessing
    
    Args:
        args: Tuple of (pdf_path, profession, db_config, llm_parser, embd_model)
        
    Returns:
        Tuple of (success, message)
    """
    pdf_path, profession, db_config, llm_parser, embd_model = args
    
    try:
        # Initialize components in worker process
        parser = CVParser(provider=llm_parser)
        embedding_gen = EmbeddingGenerator(model_name=embd_model)
        db_manager = DatabaseManager(db_config)
        
        # Create temporary processor
        processor = BatchProcessor(parser, embedding_gen, db_manager)
        result = processor.process_single_resume(pdf_path, profession, skip_existing=True)
        
        # Cleanup
        db_manager.close()
        
        return result
        
    except Exception as e:
        return (False, f"Worker error: {str(e)}")


class ParallelBatchProcessor(BatchProcessor):
    """Batch processor with multiprocessing support"""
    
    def __init__(
        self,
        parser: CVParser,
        embedding_gen: EmbeddingGenerator,
        db_manager: DatabaseManager,
        num_workers: Optional[int] = None
    ):
        """
        Initialize parallel batch processor
        
        Args:
            parser: Resume parser instance
            embedding_gen: Embedding generator instance
            db_manager: Database manager instance
            num_workers: Number of worker processes (default: CPU count)
        """
        super().__init__(parser, embedding_gen, db_manager)
        self.num_workers = num_workers or min(config.num_workers, cpu_count())
        logger.info(f"Initialized with {self.num_workers} worker processes")
    
    def process_profession_parallel(
        self,
        profession: str,
        count: int,
        skip_existing: bool = True
    ) -> dict:
        """
        Process profession resumes in parallel using multiprocessing
        
        Args:
            profession: Profession category
            count: Number of resumes to process
            skip_existing: Skip if already in database
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing profession (parallel): {profession}")
        logger.info(f"{'='*60}")
        
        # Get PDF files
        pdf_files = self.get_resume_files(profession, count)
        
        if not pdf_files:
            logger.warning(f"No PDF files found for {profession}")
            return {'success': 0, 'failed': 0, 'skipped': 0, 'total': 0}
        
        # Prepare arguments for workers
        worker_args = [
            (
                pdf_path,
                profession,
                config.database,
                config.llm_parser,
                config.embd_model
            )
            for pdf_path in pdf_files
        ]
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total': len(pdf_files)}
        
        # Process with multiprocessing pool
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_resume_worker, worker_args),
                total=len(worker_args),
                desc=f"{profession}",
                unit="resume"
            ))
        
        # Aggregate results
        for success, message in results:
            if success:
                stats['success'] += 1
            elif 'Skipped' in message:
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"\n{profession} Summary:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        
        return stats
    
    def process_all_professions_parallel(self, skip_existing: bool = True):
        """
        Process all professions with parallel processing
        
        Args:
            skip_existing: Skip if already in database
        """
        logger.info("\n" + "="*60)
        logger.info(f"Starting parallel batch processing ({self.num_workers} workers)")
        logger.info("="*60)
        
        overall_stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total': 0}
        
        for profession in config.professions:
            stats = self.process_profession_parallel(
                profession,
                config.num_cv_per_profession,
                skip_existing
            )
            
            overall_stats['success'] += stats['success']
            overall_stats['failed'] += stats['failed']
            overall_stats['skipped'] += stats['skipped']
            overall_stats['total'] += stats['total']
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("OVERALL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Resumes: {overall_stats['total']}")
        logger.info(f"Successfully Processed: {overall_stats['success']}")
        logger.info(f"Failed: {overall_stats['failed']}")
        logger.info(f"Skipped (existing): {overall_stats['skipped']}")
        
        # Database statistics
        total_in_db = self.db_manager.get_candidate_count()
        profession_counts = self.db_manager.get_profession_counts()
        
        logger.info(f"\nDatabase Statistics:")
        logger.info(f"Total Candidates: {total_in_db}")
        logger.info(f"By Profession:")
        for prof, count in profession_counts.items():
            logger.info(f"  {prof}: {count}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Parallel batch processing complete!")
        logger.info("="*60)
