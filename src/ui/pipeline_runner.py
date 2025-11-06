"""
Pipeline runner with progress tracking for Streamlit UI.
Wraps PipelineOrchestrator to provide real-time progress updates.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Generator, Optional, Callable
from datetime import datetime

# Add src to path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.config import Config
from src.utils.cache import CacheManager
from src.core.orchestrator import PipelineOrchestrator

# Import visualize module from root
try:
    from visualize import generate_all_charts
except ImportError:
    # Fallback if visualize not available
    def generate_all_charts(json_path: str, output_dir: str):
        pass


class PipelineRunner:
    """Runs pipeline with progress tracking."""
    
    PHASES = [
        "Initializing",
        "Fetching website content",
        "Extracting content",
        "Generating seed keywords",
        "Expanding keywords",
        "Normalizing keywords",
        "Generating embeddings",
        "Clustering keywords",
        "Classifying intent",
        "Suggesting match types",
        "Scoring and ranking",
        "Creating ad groups",
        "Analyzing competitors",
        "Optimizing budget",
        "Generating charts",
        "Complete"
    ]
    
    def __init__(self, config: Config, cache_manager: CacheManager, 
                 dry_run: bool = False, progress_callback: Optional[Callable] = None):
        """Initialize pipeline runner.
        
        Args:
            config: Configuration object
            cache_manager: Cache manager instance
            dry_run: If True, skip external API calls
            progress_callback: Optional callback function(phase, progress, message)
        """
        self.config = config
        self.cache = cache_manager
        self.dry_run = dry_run
        self.progress_callback = progress_callback
        self.orchestrator = None
    
    def run_with_progress(self) -> Generator[Dict[str, Any], None, None]:
        """Run pipeline with progress updates.
        
        Yields:
            Dict with phase, progress (0-100), message, and optional data
        """
        total_phases = len(self.PHASES) - 1  # Exclude "Complete"
        
        try:
            # Phase 0: Initializing
            yield {
                'phase': 0,
                'phase_name': self.PHASES[0],
                'progress': 0,
                'message': 'Initializing pipeline...',
                'status': 'running'
            }
            
            # Initialize orchestrator
            self.orchestrator = PipelineOrchestrator(
                self.config, 
                self.cache, 
                dry_run=self.dry_run
            )
            
            # Run pipeline steps manually to track progress
            # Phase 1: Fetch and extract content
            yield {
                'phase': 1,
                'phase_name': self.PHASES[1],
                'progress': int((1 / total_phases) * 100),
                'message': 'Fetching website content...',
                'status': 'running'
            }
            site_contents = self.orchestrator._fetch_and_extract_content()
            
            # Phase 2: Generate seed keywords
            yield {
                'phase': 2,
                'phase_name': self.PHASES[2],
                'progress': int((2 / total_phases) * 100),
                'message': 'Extracting and processing content...',
                'status': 'running'
            }
            
            yield {
                'phase': 3,
                'phase_name': self.PHASES[3],
                'progress': int((3 / total_phases) * 100),
                'message': 'Generating seed keywords with NLP...',
                'status': 'running'
            }
            seed_keywords = self.orchestrator._generate_seed_keywords(site_contents)
            
            # Phase 3: Expand keywords
            yield {
                'phase': 4,
                'phase_name': self.PHASES[4],
                'progress': int((4 / total_phases) * 100),
                'message': f'Expanding {len(seed_keywords)} seed keywords...',
                'status': 'running'
            }
            expanded_keywords = self.orchestrator._expand_keywords(seed_keywords)
            
            # Phase 4: Normalize
            yield {
                'phase': 5,
                'phase_name': self.PHASES[5],
                'progress': int((5 / total_phases) * 100),
                'message': 'Normalizing and deduplicating keywords...',
                'status': 'running'
            }
            normalized_keywords = self.orchestrator._normalize_keywords(expanded_keywords)
            
            # Phase 5: Generate embeddings
            yield {
                'phase': 6,
                'phase_name': self.PHASES[6],
                'progress': int((6 / total_phases) * 100),
                'message': f'Generating embeddings for {len(normalized_keywords)} keywords...',
                'status': 'running'
            }
            embeddings_data = self.orchestrator._generate_embeddings(normalized_keywords, site_contents)
            
            # Phase 6: Cluster
            yield {
                'phase': 7,
                'phase_name': self.PHASES[7],
                'progress': int((7 / total_phases) * 100),
                'message': 'Clustering keywords semantically...',
                'status': 'running'
            }
            clusters = self.orchestrator._cluster_keywords(embeddings_data)
            
            # Phase 7: Classify intent
            yield {
                'phase': 8,
                'phase_name': self.PHASES[8],
                'progress': int((8 / total_phases) * 100),
                'message': 'Classifying keyword intent...',
                'status': 'running'
            }
            classified_keywords = self.orchestrator._classify_intent(normalized_keywords)
            
            # Phase 8: Suggest match types
            yield {
                'phase': 9,
                'phase_name': self.PHASES[9],
                'progress': int((9 / total_phases) * 100),
                'message': 'Suggesting match types...',
                'status': 'running'
            }
            final_keywords = self.orchestrator._suggest_match_types(classified_keywords)
            
            # Phase 9: Score and rank
            yield {
                'phase': 10,
                'phase_name': self.PHASES[10],
                'progress': int((10 / total_phases) * 100),
                'message': 'Scoring and ranking keywords...',
                'status': 'running'
            }
            ranked_keywords = self.orchestrator._score_and_rank_keywords(final_keywords, embeddings_data)
            
            # Phase 10: Create ad groups
            yield {
                'phase': 11,
                'phase_name': self.PHASES[11],
                'progress': int((11 / total_phases) * 100),
                'message': 'Creating semantic ad groups...',
                'status': 'running'
            }
            ad_groups = self.orchestrator._create_semantic_ad_groups(clusters, ranked_keywords)
            
            # Phase 11: Analyze competitors
            yield {
                'phase': 12,
                'phase_name': self.PHASES[12],
                'progress': int((12 / total_phases) * 100),
                'message': 'Analyzing competitors...',
                'status': 'running'
            }
            competitor_analysis = self.orchestrator._analyze_competitors()
            
            # Phase 12: Optimize budget
            yield {
                'phase': 13,
                'phase_name': self.PHASES[13],
                'progress': int((13 / total_phases) * 100),
                'message': 'Optimizing budget allocation...',
                'status': 'running'
            }
            budget_optimization = self.orchestrator._optimize_budget_allocation(ranked_keywords)
            
            # Phase 13: Export results
            self.orchestrator._export_results(ad_groups, competitor_analysis, budget_optimization)
            
            # Phase 14: Generate PMax themes
            self.orchestrator._generate_pmax_themes(ad_groups)
            
            # Phase 15: Generate charts
            yield {
                'phase': 14,
                'phase_name': self.PHASES[14],
                'progress': int((14 / total_phases) * 100),
                'message': 'Generating visualization charts...',
                'status': 'running'
            }
            
            # Generate charts (only top 4 most relevant)
            json_path = self.orchestrator.output_path / "keyword_data.json"
            if json_path.exists():
                run_id = self.orchestrator.run_id
                chart_dir = f"charts/{run_id}"
                try:
                    # Generate only the 4 most relevant charts
                    generate_all_charts(str(json_path), chart_dir)
                except Exception as e:
                    # Chart generation failed, but continue
                    self.logger.warning(f"Chart generation failed: {e}")
            
            # Complete
            result = {
                'success': True,
                'total_keywords': len(ranked_keywords) if ranked_keywords else 0,
                'total_ad_groups': len(ad_groups) if ad_groups else 0,
                'output_path': str(self.orchestrator.output_path),
                'run_id': self.orchestrator.run_id,
                'ad_groups': ad_groups,
                'competitor_analysis': competitor_analysis,
                'budget_optimization': budget_optimization
            }
            
            yield {
                'phase': 15,
                'phase_name': self.PHASES[15],
                'progress': 100,
                'message': 'Pipeline completed successfully!',
                'status': 'complete',
                'result': result
            }
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            yield {
                'phase': -1,
                'phase_name': 'Error',
                'progress': 0,
                'message': f'Pipeline failed: {error_msg}',
                'status': 'error',
                'error': error_msg
            }

