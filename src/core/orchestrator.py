"""Pipeline orchestrator that coordinates all components."""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from ..utils.config import Config
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from .site_fetcher import SiteFetcher
from .content_extractor import ContentExtractor
from .seed_generator import SeedGenerator
from ..scraper.wordstream_client import WordStreamClient
from ..nlp.normalizer import KeywordNormalizer
from ..nlp.embedding_store import EmbeddingStore
from ..nlp.clustering import KeywordClusterer
from ..llm.intent_classifier import IntentClassifier
from ..llm.match_type_suggester import MatchTypeSuggester
from ..llm.ad_group_namer import AdGroupNamer
from ..llm.gemini_client import GeminiClient


class PipelineOrchestrator:
    """Orchestrates the entire keyword research pipeline."""
    
    def __init__(self, config: Config, cache_manager: CacheManager, dry_run: bool = False):
        """Initialize orchestrator.
        
        Args:
            config: Configuration object
            cache_manager: Cache manager instance
            dry_run: If True, skip external API calls
        """
        self.config = config
        self.cache = cache_manager
        self.dry_run = dry_run
        self.logger = get_logger('Orchestrator')
        
        # Initialize components
        self.site_fetcher = SiteFetcher(cache_manager)
        self.content_extractor = ContentExtractor()
        self.seed_generator = SeedGenerator(config.llm.__dict__)
        self.wordstream_client = WordStreamClient(cache_manager)
        self.normalizer = KeywordNormalizer()
        self.embedding_store = EmbeddingStore(cache_dir=config.cache_dir)
        self.clusterer = KeywordClusterer()
        self.intent_classifier = IntentClassifier(config.llm.__dict__)
        self.match_type_suggester = MatchTypeSuggester(config.llm.__dict__)
        self.ad_group_namer = AdGroupNamer(config.llm.__dict__)
        
        # Initialize Gemini client for advanced features
        try:
            if hasattr(config, 'gemini') and hasattr(config.gemini, 'api_key') and config.gemini.api_key:
                self.gemini_client = GeminiClient(config.gemini.api_key, config.gemini.model)
            else:
                self.gemini_client = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini client: {e}")
            self.gemini_client = None
        
        # Create run-specific output directory
        self.run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.output_path = Path(config.output_dir) / self.run_id
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Returns:
            Dict with success status and results
        """
        try:
            self.logger.info("Starting keyword research pipeline")
            
            # Step 1: Fetch and extract site content
            site_contents = self._fetch_and_extract_content()
            
            # Step 2: Generate seed keywords with advanced NLP
            seed_keywords = self._generate_seed_keywords(site_contents)
            
            # Step 3: Expand keywords using WordStream and SERP sources
            expanded_keywords = self._expand_keywords(seed_keywords)
            
            # Step 4: Normalize and deduplicate keywords
            normalized_keywords = self._normalize_keywords(expanded_keywords)
            
            # Step 5: Generate embeddings and cluster keywords
            embeddings_data = self._generate_embeddings(normalized_keywords, site_contents)
            clusters = self._cluster_keywords(embeddings_data)
            
            # Step 6: Classify intent and suggest match types
            classified_keywords = self._classify_intent(normalized_keywords)
            final_keywords = self._suggest_match_types(classified_keywords)
            
            # Step 7: Score and rank keywords with advanced metrics
            ranked_keywords = self._score_and_rank_keywords(final_keywords, embeddings_data)
            
            # Step 8: Create semantic ad groups
            ad_groups = self._create_semantic_ad_groups(clusters, ranked_keywords)
            
            # Step 9: Analyze competitors (if Gemini available)
            competitor_analysis = self._analyze_competitors()
            
            # Step 10: Optimize budget allocation (if Gemini available)
            budget_optimization = self._optimize_budget_allocation(ranked_keywords)
            
            # Step 11: Export results with enhanced data
            self._export_results(ad_groups, competitor_analysis, budget_optimization)
            
            # Step 12: Generate PMax themes
            self._generate_pmax_themes(ad_groups)
            
            result = {
                'success': True,
                'total_keywords': len(ranked_keywords),
                'total_ad_groups': len(ad_groups),
                'output_path': str(self.output_path),
                'run_id': self.run_id
            }
            
            self.logger.info("Pipeline completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fetch_and_extract_content(self) -> List[Dict]:
        """Fetch and extract content from target website and competitors."""
        self.logger.info("Fetching and extracting website content")
        
        all_contents = []
        
        # Target website
        if not self.dry_run:
            target_urls = self.site_fetcher.discover_site_pages(self.config.website)
            self.logger.info(f"Discovered {len(target_urls)} pages for target website")
            
            target_pages = self.site_fetcher.fetch_multiple_pages(target_urls, max_pages=10)
            
            for page_data in target_pages:
                content = self.content_extractor.extract_content(page_data)
                if content:
                    content['source'] = 'target'
                    all_contents.append(content)
        
        # Competitors (simplified - just homepage for MVP)
        for competitor_url in self.config.competitors[:3]:  # Limit to 3 competitors
            if not self.dry_run:
                self.logger.info(f"Fetching competitor: {competitor_url}")
                page_data = self.site_fetcher.fetch_page(competitor_url)
                
                if page_data:
                    content = self.content_extractor.extract_content(page_data)
                    if content:
                        content['source'] = 'competitor'
                        all_contents.append(content)
        
        # In dry run mode, create mock content
        if self.dry_run:
            all_contents = self._create_mock_content()
        
        self.logger.info(f"Extracted content from {len(all_contents)} pages")
        return all_contents
    
    def _create_mock_content(self) -> List[Dict]:
        """Create mock content for dry run mode."""
        return [{
            'url': self.config.website,
            'title': 'AI Workflow Automation Platform',
            'meta_description': 'Automate your business workflows with AI-powered tools',
            'headings': {
                'h1': ['AI Workflow Automation'],
                'h2': ['Task Management', 'Process Automation', 'AI Integration'],
                'h3': []
            },
            'body_text': 'Streamline your business processes with our AI workflow automation platform. '
                        'Automate repetitive tasks, manage workflows, and integrate AI into your business processes.',
            'source': 'target',
            'word_count': 50
        }]
    
    def _generate_seed_keywords(self, site_contents: List[Dict]) -> List[str]:
        """Generate seed keywords from site content using advanced NLP."""
        self.logger.info("Generating seed keywords with advanced NLP")
        
        seeds = self.seed_generator.generate_seeds(site_contents, self.config.seeds)
        expanded_seeds = self.seed_generator.expand_seeds_with_variations(seeds)
        
        self.logger.info(f"Generated {len(expanded_seeds)} seed keywords")
        return expanded_seeds[:50]  # Limit seeds for expansion
    
    def _expand_keywords(self, seed_keywords: List[str]) -> List[Dict]:
        """Expand seed keywords using WordStream and SERP sources."""
        if self.dry_run:
            self.logger.info("DRY RUN: Skipping keyword expansion")
            return self._create_keyword_data(seed_keywords)
        
        self.logger.info("Expanding keywords using external sources")
        
        # Use WordStream client to fetch keyword suggestions
        expanded_keywords = self.wordstream_client.fetch_keywords(seed_keywords)
        
        # Add enriched trends data
        expanded_keywords = self.wordstream_client.enrich_keywords_with_trends(expanded_keywords)
        
        self.logger.info(f"Expanded to {len(expanded_keywords)} keywords from external sources")
        
        # If no external keywords, fallback to seed-based data
        if not expanded_keywords:
            self.logger.warning("No external keywords found, using seed-based fallback")
            expanded_keywords = self._create_keyword_data(seed_keywords)
        
        return expanded_keywords
    
    def _normalize_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Normalize and deduplicate keywords."""
        self.logger.info("Normalizing and deduplicating keywords")
        
        # Normalize with advanced NLP
        normalized = self.normalizer.normalize_keywords(keywords)
        
        # Apply filters from config
        filtered = self.normalizer.filter_keywords(normalized, self.config.filters.__dict__)
        
        # Limit to max_keywords
        final_keywords = filtered[:self.config.max_keywords]
        
        self.logger.info(f"Normalized to {len(final_keywords)} keywords")
        return final_keywords
    
    def _generate_embeddings(self, keywords: List[Dict], site_contents: List[Dict]) -> Dict:
        """Generate embeddings for keywords and site content."""
        self.logger.info("Generating semantic embeddings")
        
        embeddings_data = self.embedding_store.generate_embeddings(keywords, site_contents)
        
        # Calculate similarities
        similarities = self.embedding_store.calculate_similarities(
            embeddings_data['keyword_embeddings'],
            embeddings_data['content_embeddings']
        )
        
        embeddings_data.update(similarities)
        return embeddings_data
    
    def _cluster_keywords(self, embeddings_data: Dict) -> Dict:
        """Cluster keywords using semantic similarity."""
        self.logger.info("Clustering keywords semantically")
        
        clustering_result = self.clusterer.cluster_keywords(
            embeddings_data['keyword_embeddings'],
            method="auto",
            max_clusters=30
        )
        
        return clustering_result
    
    def _classify_intent(self, keywords: List[Dict]) -> List[Dict]:
        """Classify search intent for keywords."""
        self.logger.info("Classifying keyword intent")
        
        classified_keywords = self.intent_classifier.classify_keywords(keywords)
        
        self.logger.info(f"Classified intent for {len(classified_keywords)} keywords")
        return classified_keywords
    
    def _suggest_match_types(self, keywords: List[Dict]) -> List[Dict]:
        """Suggest match types for keywords."""
        self.logger.info("Suggesting match types")
        
        # Determine budget level for strategy
        budget_level = "medium"  # Could be derived from config in future
        
        keywords_with_match_types = self.match_type_suggester.suggest_match_types(
            keywords, budget_level
        )
        
        self.logger.info(f"Suggested match types for {len(keywords_with_match_types)} keywords")
        return keywords_with_match_types
    
    def _create_keyword_data(self, keywords: List[str]) -> List[Dict]:
        """Create keyword data structures with mock metrics for MVP."""
        keyword_data = []
        
        for i, keyword in enumerate(keywords):
            # Mock data for MVP - in production would come from APIs
            data = {
                'keyword': keyword,
                'volume': max(100, 1000 - i * 10),  # Decreasing mock volume
                'cpc_low': 0.5 + (i % 10) * 0.1,
                'cpc_high': 2.0 + (i % 15) * 0.2,
                'competition': min(0.9, 0.3 + (i % 20) * 0.03),
                'source': 'generated',
                'confidence': 0.8 if i < 50 else 0.6  # Higher confidence for top keywords
            }
            keyword_data.append(data)
        
        return keyword_data
    
    def _score_and_rank_keywords(self, keywords: List[Dict], embeddings_data: Dict = None) -> List[Dict]:
        """Score and rank keywords using advanced metrics."""
        self.logger.info("Scoring and ranking keywords with advanced metrics")
        
        for keyword_data in keywords:
            # Get basic metrics
            volume = keyword_data.get('volume', 0)
            cpc_high = keyword_data.get('cpc_high', 0)
            competition = keyword_data.get('competition', 0)
            intent = keyword_data.get('intent', 'commercial')
            
            # Normalize volume
            max_volume = max([kw.get('volume', 0) for kw in keywords]) or 1
            normalized_volume = volume / max_volume
            
            # Normalize CPC
            max_cpc = max([kw.get('cpc_high', 0) for kw in keywords]) or 1
            normalized_cpc = cpc_high / max_cpc
            
            # Intent weights (as per spec)
            intent_weights = {
                'transactional': 1.0,
                'commercial': 0.8,
                'informational': 0.4,
                'navigational': 0.2
            }
            intent_weight = intent_weights.get(intent, 0.6)
            
            # Competition penalty
            competition_penalty = 1 - competition
            
            # Context relevance from embeddings
            context_score = 0.5  # Default
            if embeddings_data and 'keyword_relevance' in embeddings_data:
                keyword_relevance = embeddings_data['keyword_relevance'].get(keyword_data['keyword'], {})
                context_score = keyword_relevance.get('relevance_score', 0.5)
            
            # Calculate final score (as per spec formula)
            if volume > 0:
                score = (0.55 * normalized_volume + 
                        0.25 * normalized_cpc + 
                        0.2 * intent_weight * competition_penalty)
            else:
                # Fallback formula when volume missing
                score = (0.6 * context_score + 
                        0.4 * normalized_cpc * intent_weight)
            
            keyword_data['score'] = round(score, 3)
            keyword_data['context_score'] = round(context_score, 3)
        
        # Sort by score descending
        ranked_keywords = sorted(keywords, key=lambda x: x['score'], reverse=True)
        
        self.logger.info(f"Ranked {len(ranked_keywords)} keywords")
        return ranked_keywords
    
    def _create_semantic_ad_groups(self, clusters: Dict, keywords: List[Dict]) -> List[Dict]:
        """Create semantic ad groups using clustering results."""
        self.logger.info("Creating semantic ad groups")
        
        # Map keywords to clusters
        keyword_to_cluster = {}
        for cluster in clusters.get('clusters', []):
            for kw_data in cluster['keywords']:
                keyword_to_cluster[kw_data['keyword']] = cluster['cluster_id']
        
        # Group keywords by cluster
        cluster_groups = {}
        unclustered_keywords = []
        
        for keyword_data in keywords:
            cluster_id = keyword_to_cluster.get(keyword_data['keyword'])
            
            if cluster_id:
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(keyword_data)
            else:
                unclustered_keywords.append(keyword_data)
        
        # Create ad groups from clusters
        ad_groups = []
        
        # Process clustered keywords
        cluster_data_list = []
        for cluster_id, cluster_keywords in cluster_groups.items():
            cluster_info = {
                'cluster_id': cluster_id,
                'keywords': cluster_keywords,
                'keyword_count': len(cluster_keywords)
            }
            cluster_data_list.append(cluster_info)
        
        # Generate ad group names using LLM
        business_context = f"Website: {self.config.website}"
        named_clusters = self.ad_group_namer.generate_ad_group_names(cluster_data_list, business_context)
        
        for cluster in named_clusters:
            ad_group = {
                'ad_group_name': cluster['ad_group_name'],
                'keywords': cluster['keywords'][:25],  # Limit per ad group
                'cluster_id': cluster['cluster_id'],
                'total_volume': sum(kw.get('volume', 0) for kw in cluster['keywords']),
                'keyword_count': len(cluster['keywords'][:25])
            }
            ad_groups.append(ad_group)
        
        # Handle unclustered keywords
        if unclustered_keywords:
            # Group unclustered by simple rules
            misc_groups = self._group_unclustered_keywords(unclustered_keywords)
            ad_groups.extend(misc_groups)
        
        self.logger.info(f"Created {len(ad_groups)} semantic ad groups")
        return ad_groups
    
    def _group_unclustered_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Group unclustered keywords using simple rules."""
        # Group by intent
        intent_groups = {}
        
        for kw_data in keywords:
            intent = kw_data.get('intent', 'commercial')
            
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(kw_data)
        
        ad_groups = []
        for intent, intent_keywords in intent_groups.items():
            if len(intent_keywords) >= 2:  # Only create groups with multiple keywords
                ad_group = {
                    'ad_group_name': f"{intent.title()} Keywords",
                    'keywords': intent_keywords[:15],  # Limit size
                    'cluster_id': f"unclustered_{intent}",
                    'total_volume': sum(kw.get('volume', 0) for kw in intent_keywords),
                    'keyword_count': len(intent_keywords[:15])
                }
                ad_groups.append(ad_group)
        
        return ad_groups
    
    def _analyze_competitors(self) -> Dict:
        """Analyze competitors using Gemini AI."""
        if not self.gemini_client:
            self.logger.info("Gemini not available - skipping competitor analysis")
            return {'analysis_available': False}
        
        self.logger.info("Analyzing competitors with Gemini AI")
        
        competitor_analysis = {}
        business_context = getattr(self.config, 'business', {})
        business_description = business_context.get('description', f"Business website: {self.config.website}")
        
        # Analyze each competitor
        for i, competitor_url in enumerate(self.config.competitors):
            try:
                self.logger.info(f"Analyzing competitor: {competitor_url}")
                analysis = self.gemini_client.analyze_competitor_website(
                    competitor_url, 
                    business_description
                )
                competitor_analysis[f"competitor_{i+1}"] = {
                    'url': competitor_url,
                    'analysis': analysis
                }
            except Exception as e:
                self.logger.warning(f"Competitor analysis failed for {competitor_url}: {e}")
                competitor_analysis[f"competitor_{i+1}"] = {
                    'url': competitor_url,
                    'analysis': {'error': str(e)}
                }
        
        competitor_analysis['analysis_available'] = True
        competitor_analysis['total_competitors'] = len(self.config.competitors)
        
        return competitor_analysis
    
    def _optimize_budget_allocation(self, keywords: List[Dict]) -> Dict:
        """Optimize budget allocation using Gemini AI."""
        if not self.gemini_client:
            self.logger.info("Gemini not available - skipping budget optimization")
            return {'optimization_available': False}
        
        self.logger.info("Optimizing budget allocation with Gemini AI")
        
        try:
            # Get budget settings from config
            budget_config = getattr(self.config, 'budget', {})
            total_budget = budget_config.get('total_monthly', 10000)
            conversion_rate = budget_config.get('conversion_rate', 0.02)
            
            # Campaign types for this business
            campaign_types = ['Search', 'Shopping', 'Performance Max']
            
            # Get optimization recommendations
            optimization = self.gemini_client.optimize_budget_allocation(
                keywords, 
                total_budget, 
                campaign_types, 
                conversion_rate
            )
            
            optimization['optimization_available'] = True
            optimization['input_budget'] = total_budget
            optimization['input_conversion_rate'] = conversion_rate
            
            return optimization
            
        except Exception as e:
            self.logger.warning(f"Budget optimization failed: {e}")
            return {
                'optimization_available': False,
                'error': str(e)
            }
    
    def _suggest_match_type(self, keyword_data: Dict) -> str:
        """Suggest match type for keyword."""
        keyword = keyword_data['keyword']
        score = keyword_data.get('score', 0)
        
        # High-value keywords get exact match
        if score > 0.8:
            return 'exact'
        
        # Branded or specific terms get phrase match
        if len(keyword.split()) >= 3 or any(term in keyword.lower() for term in ['software', 'tool', 'platform']):
            return 'phrase'
        
        # Default to broad match modifier
        return 'broad'
    
    def _export_results(self, ad_groups: List[Dict], competitor_analysis: Dict = None, 
                       budget_optimization: Dict = None) -> None:
        """Export results to CSV and JSON with enhanced analytics."""
        self.logger.info("Exporting results with enhanced analytics")
        
        # Export CSV (original format)
        if self.config.export.csv:
            self._export_csv(ad_groups)
        
        # Export JSON with enhanced data
        if self.config.export.json:
            self._export_json(ad_groups, competitor_analysis, budget_optimization)
        
        # Export competitor analysis report
        if competitor_analysis and competitor_analysis.get('analysis_available'):
            self._export_competitor_analysis(competitor_analysis)
        
        # Export budget optimization report
        if budget_optimization and budget_optimization.get('optimization_available'):
            self._export_budget_optimization(budget_optimization)
    
    def _export_csv(self, ad_groups: List[Dict]) -> None:
        """Export to CSV format."""
        csv_path = self.output_path / 'search_adgroups.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'ad_group', 'keyword', 'match_type', 'volume', 'cpc_low', 'cpc_high',
                'competition', 'score', 'source', 'confidence', 'notes'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for group in ad_groups:
                for keyword_data in group['keywords']:
                    writer.writerow({
                        'ad_group': group['ad_group_name'],
                        'keyword': keyword_data['keyword'],
                        'match_type': keyword_data['match_type'],
                        'volume': keyword_data.get('volume', ''),
                        'cpc_low': keyword_data.get('cpc_low', ''),
                        'cpc_high': keyword_data.get('cpc_high', ''),
                        'competition': keyword_data.get('competition', ''),
                        'score': keyword_data.get('score', ''),
                        'source': keyword_data.get('source', ''),
                        'confidence': keyword_data.get('confidence', ''),
                        'notes': 'MVP generated'
                    })
        
        self.logger.info(f"Exported CSV to {csv_path}")
    
    def _export_json(self, ad_groups: List[Dict], competitor_analysis: Dict = None, 
                    budget_optimization: Dict = None) -> None:
        """Export to JSON format with enhanced data."""
        json_path = self.output_path / 'keyword_data.json'
        
        export_data = {
            'metadata': {
                'run_id': self.run_id,
                'generated_at': datetime.now().isoformat(),
                'config': {
                    'website': self.config.website,
                    'max_keywords': self.config.max_keywords,
                    'language': self.config.language
                }
            },
            'ad_groups': ad_groups
        }
        
        # Add enhanced analytics if available
        if competitor_analysis:
            export_data['competitor_analysis'] = competitor_analysis
        
        if budget_optimization:
            export_data['budget_optimization'] = budget_optimization
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Exported JSON to {json_path}")
    
    def _generate_pmax_themes(self, ad_groups: List[Dict]) -> None:
        """Generate Performance Max theme suggestions using LLM."""
        self.logger.info("Generating Performance Max themes")
        
        # Sort ad groups by total volume and take top performers
        top_ad_groups = sorted(ad_groups, key=lambda x: x.get('total_volume', 0), reverse=True)[:8]
        
        # Generate themes using LLM
        business_context = f"Website: {self.config.website}, Industry: digital marketing tools"
        pmax_themes = self.ad_group_namer.generate_pmax_themes(top_ad_groups, business_context)
        
        # Write detailed PMax themes file
        themes_path = self.output_path / 'pmax_themes.md'
        
        with open(themes_path, 'w', encoding='utf-8') as f:
            f.write("# Performance Max Asset Themes\n\n")
            f.write(f"Generated for: {self.config.website}\n")
            f.write(f"Total themes: {len(pmax_themes)}\n")
            f.write(f"Generation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, theme in enumerate(pmax_themes, 1):
                f.write(f"## Theme {i}: {theme['theme_name']}\n\n")
                
                # Theme statistics
                f.write(f"**Keywords in theme:** {theme['keyword_count']}\n")
                f.write(f"**Total search volume:** {theme['total_volume']:,}\n\n")
                
                # Top keywords
                f.write(f"**Top Keywords:** {', '.join(theme['keywords'][:8])}\n\n")
                
                # Headlines
                f.write("**Headlines** (max 30 characters):\n")
                for headline in theme['headlines']:
                    f.write(f"- {headline} ({len(headline)} chars)\n")
                f.write("\n")
                
                # Long Headlines
                f.write("**Long Headlines** (max 90 characters):\n")
                for long_headline in theme['long_headlines']:
                    f.write(f"- {long_headline} ({len(long_headline)} chars)\n")
                f.write("\n")
                
                # Descriptions
                f.write("**Descriptions** (max 90 characters):\n")
                for description in theme['descriptions']:
                    f.write(f"- {description} ({len(description)} chars)\n")
                f.write("\n")
                
                # Callouts
                f.write("**Callouts** (max 25 characters):\n")
                for callout in theme['callouts']:
                    f.write(f"- {callout} ({len(callout)} chars)\n")
                f.write("\n")
                
                f.write("---\n\n")
            
            # Add negative keyword suggestions
            negative_keywords = self.match_type_suggester.suggest_negative_keywords(
                [kw for group in ad_groups for kw in group['keywords']], 
                business_context
            )
            
            if negative_keywords:
                f.write("## Suggested Negative Keywords\n\n")
                f.write("Add these negative keywords to prevent irrelevant traffic:\n\n")
                for neg_kw in negative_keywords:
                    f.write(f"- {neg_kw}\n")
                f.write("\n")
        
        self.logger.info(f"Generated {len(pmax_themes)} PMax themes at {themes_path}")
    
    def _export_competitor_analysis(self, competitor_analysis: Dict) -> None:
        """Export competitor analysis to markdown report."""
        report_path = self.output_path / 'competitor_analysis.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🕵️ Competitor Intelligence Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for competitor_id, data in competitor_analysis.items():
                if competitor_id.startswith('competitor_'):
                    f.write(f"## Competitor: {data['url']}\n\n")
                    
                    analysis = data.get('analysis', {})
                    if 'error' in analysis:
                        f.write(f"⚠️ Analysis failed: {analysis['error']}\n\n")
                        continue
                    
                    # Value propositions
                    if 'value_propositions' in analysis:
                        f.write("### 🎯 Value Propositions:\n")
                        for vp in analysis['value_propositions']:
                            f.write(f"- {vp}\n")
                        f.write("\n")
                    
                    # Keyword opportunities
                    if 'keyword_opportunities' in analysis:
                        f.write("### 🔍 Keyword Opportunities:\n")
                        for keyword in analysis['keyword_opportunities']:
                            f.write(f"- {keyword}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
        
        self.logger.info(f"Exported competitor analysis to {report_path}")
    
    def _export_budget_optimization(self, budget_optimization: Dict) -> None:
        """Export budget optimization to markdown report."""
        report_path = self.output_path / 'budget_optimization.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 💰 Budget Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Budget allocation
            if 'budget_allocation' in budget_optimization:
                f.write("## 🎯 Recommended Budget Allocation\n\n")
                allocation = budget_optimization['budget_allocation']
                
                for campaign_type, details in allocation.items():
                    campaign_name = campaign_type.replace('_', ' ').title()
                    f.write(f"### {campaign_name}\n")
                    f.write(f"- **Amount**: ${details.get('amount', 0):,.0f}\n")
                    f.write(f"- **Percentage**: {details.get('percentage', 0)}%\n")
                    f.write(f"- **Reasoning**: {details.get('reasoning', 'N/A')}\n\n")
            
            # Performance forecast
            if 'performance_forecast' in budget_optimization:
                f.write("## 📈 Performance Forecast\n\n")
                forecast = budget_optimization['performance_forecast']
                
                f.write(f"- **Estimated Clicks**: {forecast.get('estimated_clicks', 0):,}\n")
                f.write(f"- **Estimated Conversions**: {forecast.get('estimated_conversions', 0):,}\n")
                f.write(f"- **Estimated ROAS**: {forecast.get('estimated_roas', 0):.1f}x\n\n")
        
        self.logger.info(f"Exported budget optimization to {report_path}")