#!/usr/bin/env python3
"""
Keyword Research CLI - Main entry point

Usage:
    python run.py --config config.yaml [--dry-run] [--debug]
"""

import argparse
import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger
from src.utils.cache import CacheManager
from src.core.orchestrator import PipelineOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Keyword Research CLI - Generate keyword inventory for Google Ads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --config config.yaml
    python run.py --config config.yaml --dry-run
    python run.py --config config.yaml --debug
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and show what would be done without making external calls'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    try:
        config = Config.from_yaml(args.config)
        config.validate()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config.logging.level
    run_id = setup_logging(log_level, config.output_dir)
    logger = get_logger('Main')
    
    logger.info(f"Starting keyword research pipeline - Run ID: {run_id}")
    logger.info(f"Target website: {config.website}")
    logger.info(f"Max keywords: {config.max_keywords}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No external calls will be made")
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager(config.cache_dir)
        
        # Initialize and run pipeline
        orchestrator = PipelineOrchestrator(config, cache_manager, dry_run=args.dry_run)
        result = orchestrator.run()
        
        if result['success']:
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Generated {result.get('total_keywords', 0)} keywords")
            logger.info(f"Outputs saved to: {result.get('output_path', 'unknown')}")
            
            # Print summary with Unicode fallback for Windows
            try:
                print(f"\n✅ Keyword research completed!")
                print(f"📊 Total keywords: {result.get('total_keywords', 0)}")
                print(f"📁 Output directory: {result.get('output_path', 'unknown')}")
                print(f"🆔 Run ID: {run_id}")
            except UnicodeEncodeError:
                # Fallback for Windows systems with encoding issues
                print(f"\n[SUCCESS] Keyword research completed!")
                print(f"Total keywords: {result.get('total_keywords', 0)}")
                print(f"Output directory: {result.get('output_path', 'unknown')}")
                print(f"Run ID: {run_id}")
            
        else:
            logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
            try:
                print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            except UnicodeEncodeError:
                print(f"\n[ERROR] Pipeline failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        try:
            print("\n⏹️  Pipeline interrupted by user")
        except UnicodeEncodeError:
            print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        try:
            print(f"\n❌ Unexpected error: {e}")
        except UnicodeEncodeError:
            print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()