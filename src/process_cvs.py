#!/usr/bin/env python3
"""
Resume Processing Script
Entry point for batch processing resumes with parsing, embedding, and database storage
"""
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser import ResumeParser
from src.embedding_generator import EmbeddingGenerator
from src.database_manager import DatabaseManager
from src.cv_batch_processor_utils import BatchProcessor, ParallelBatchProcessor
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('resume_processing.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for resume processing"""
    parser = argparse.ArgumentParser(
        description="Process resumes: Parse, Generate Embeddings, and Store in Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process all resumes with parallel processing
  %(prog)s --no-parallel            # Process sequentially
  %(prog)s --workers 8              # Use 8 worker processes
  %(prog)s --force                  # Reprocess existing resumes
  %(prog)s --verbose                # Enable debug logging
  %(prog)s --profession ARTS        # Process only ARTS profession
        """
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (sequential mode)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: {config.num_workers})'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess existing resumes (default: skip existing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    parser.add_argument(
        '--profession',
        type=str,
        choices=config.professions,
        help='Process only specific profession'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help=f'Number of resumes per profession (default: {config.num_cv_per_profession})'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("="*60)
        logger.info("CV PARSER AND RECOMMENDER - BATCH PROCESSING")
        logger.info("="*60)
        logger.info(f"LLM Provider: {config.llm_parser}")
        logger.info(f"Embedding Model: {config.embd_model}")
        logger.info(f"Database: {config.database['database']}")
        logger.info(f"Parallel Processing: {not args.no_parallel}")
        if not args.no_parallel:
            workers = args.workers or config.num_workers
            logger.info(f"Worker Processes: {workers}")
        logger.info(f"Skip Existing: {not args.force}")
        logger.info("="*60)
        
        # Initialize components
        logger.info("\n📦 Initializing components...")
        
        logger.info("  → Initializing resume parser...")
        resume_parser = ResumeParser(provider=config.llm_parser)
        
        logger.info("  → Loading embedding model...")
        embedding_gen = EmbeddingGenerator(model_name=config.embd_model)
        
        logger.info("  → Connecting to database...")
        db_manager = DatabaseManager(config.database)
        
        logger.info("✓ All components initialized\n")
        
        # Create batch processor
        if args.no_parallel:
            logger.info("Using sequential batch processor")
            processor = BatchProcessor(resume_parser, embedding_gen, db_manager)
        else:
            logger.info(f"Using parallel batch processor with {args.workers or config.num_workers} workers")
            processor = ParallelBatchProcessor(
                resume_parser, 
                embedding_gen, 
                db_manager,
                num_workers=args.workers
            )
        
        # Process resumes
        skip_existing = not args.force
        
        if args.profession:
            # Process single profession
            count = args.count or config.num_cv_per_profession
            if args.no_parallel:
                processor.process_profession(args.profession, count, skip_existing)
            else:
                processor.process_profession_parallel(args.profession, count, skip_existing)
        else:
            # Process all professions
            if args.no_parallel:
                processor.process_all_professions(skip_existing)
            else:
                processor.process_all_professions_parallel(skip_existing)
        
        # Cleanup
        db_manager.close()
        
        logger.info("\n✅ Processing completed successfully!")
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
