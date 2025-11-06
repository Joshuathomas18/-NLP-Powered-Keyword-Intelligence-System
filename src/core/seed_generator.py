"""Seed keyword generation from site content using NLP and LLM."""

import re
from typing import List, Dict, Set, Optional
from collections import Counter
import os

try:
    import spacy
    from keybert import KeyBERT
    import yake
    from openai import OpenAI
except ImportError:
    spacy = None
    KeyBERT = None
    yake = None
    OpenAI = None

from ..utils.logging import get_logger


class SeedGenerator:
    """Generates seed keywords from website content using advanced NLP and LLM."""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        self.logger = get_logger('SeedGenerator')
        self.llm_config = llm_config or {}
        
        # Initialize NLP tools
        self.nlp = None
        self.keybert = None
        self.yake_extractor = None
        self.openai_client = None
        
        self._init_nlp_tools()
        self._init_llm_client()
    
    def _init_nlp_tools(self):
        """Initialize NLP tools if available."""
        # Initialize spaCy
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model")
            except OSError:
                self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize KeyBERT
        if KeyBERT:
            try:
                self.keybert = KeyBERT()
                self.logger.info("Initialized KeyBERT")
            except Exception as e:
                self.logger.warning(f"Failed to initialize KeyBERT: {e}")
        
        # Initialize YAKE
        if yake:
            try:
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,  # n-gram size
                    dedupLim=0.7,  # deduplication threshold
                    top=20  # top keywords
                )
                self.logger.info("Initialized YAKE extractor")
            except Exception as e:
                self.logger.warning(f"Failed to initialize YAKE: {e}")
    
    def _init_llm_client(self):
        """Initialize LLM client if API key available."""
        gemini_key = os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # Try Gemini first (free, better for this use case)
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                from ..llm.gemini_client import GeminiClient
                self.gemini_client = GeminiClient(gemini_key, "gemini-2.0-flash-exp")
                self.client_type = "gemini"
                self.logger.info("Initialized Gemini client for seed expansion")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
                self.client_type = None
        # Fallback to OpenAI if available
        elif OpenAI and openai_key:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.client_type = "openai"
                self.logger.info("Initialized OpenAI client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
                self.client_type = None
        else:
            self.openai_client = None
            self.gemini_client = None
            self.client_type = None
    
    def generate_seeds(self, site_contents: List[Dict], manual_seeds: List[str] = None) -> List[str]:
        """Generate seed keywords from site content and manual seeds using advanced NLP.
        
        Args:
            site_contents: List of extracted content dictionaries
            manual_seeds: Manual seed keywords from config
            
        Returns:
            List of seed keywords
        """
        self.logger.info(f"Generating seeds from {len(site_contents)} pages using advanced NLP")
        
        seeds = set()
        
        # Add manual seeds
        if manual_seeds:
            seeds.update(manual_seeds)
            self.logger.info(f"Added {len(manual_seeds)} manual seeds")
        
        # Combine all content for analysis
        combined_content = self._combine_site_content(site_contents)
        
        # Extract seeds using multiple methods
        if self.nlp:
            spacy_seeds = self._extract_with_spacy(combined_content)
            seeds.update(spacy_seeds)
            self.logger.info(f"Added {len(spacy_seeds)} spaCy NER seeds")
        
        if self.keybert:
            keybert_seeds = self._extract_with_keybert(combined_content)
            seeds.update(keybert_seeds)
            self.logger.info(f"Added {len(keybert_seeds)} KeyBERT seeds")
        
        if self.yake_extractor:
            yake_seeds = self._extract_with_yake(combined_content)
            seeds.update(yake_seeds)
            self.logger.info(f"Added {len(yake_seeds)} YAKE seeds")
        
        # Fallback to basic extraction if no advanced tools
        if not seeds:
            for content in site_contents:
                content_seeds = self._extract_seeds_from_content(content)
                seeds.update(content_seeds)
        
        # Expand seeds with LLM if available
        if (self.gemini_client or self.openai_client) and len(seeds) > 5:
            llm_seeds = self._expand_with_llm(list(seeds)[:20], combined_content)
            seeds.update(llm_seeds)
            self.logger.info(f"Added {len(llm_seeds)} LLM-expanded seeds")
        
        # Score and filter seeds
        scored_seeds = self._score_seeds(list(seeds), site_contents)
        
        # Sort by score and return top seeds
        top_seeds = sorted(scored_seeds.items(), key=lambda x: x[1], reverse=True)
        final_seeds = [seed for seed, score in top_seeds[:100]]  # Limit to top 100
        
        self.logger.info(f"Generated {len(final_seeds)} final seed keywords")
        return final_seeds
    
    def _combine_site_content(self, site_contents: List[Dict]) -> str:
        """Combine site content into a single text for analysis."""
        content_parts = []
        
        for content in site_contents:
            parts = [
                content.get('title', ''),
                content.get('meta_description', ''),
                ' '.join(content.get('headings', {}).get('h1', [])),
                ' '.join(content.get('headings', {}).get('h2', [])),
                content.get('body_text', '')[:1000]  # Limit body text
            ]
            combined = ' '.join(filter(None, parts))
            content_parts.append(combined)
        
        return ' '.join(content_parts)
    
    def _extract_with_spacy(self, content: str, max_seeds: int = 30) -> Set[str]:
        """Extract seeds using spaCy NER and noun chunks."""
        seeds = set()
        
        try:
            doc = self.nlp(content)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'GPE', 'EVENT']:
                    entity_text = ent.text.lower().strip()
                    if self._is_valid_seed(entity_text):
                        seeds.add(entity_text)
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if self._is_valid_seed(chunk_text) and len(chunk_text.split()) <= 4:
                    seeds.add(chunk_text)
            
        except Exception as e:
            self.logger.warning(f"spaCy extraction failed: {e}")
        
        return set(list(seeds)[:max_seeds])
    
    def _extract_with_keybert(self, content: str, max_seeds: int = 25) -> Set[str]:
        """Extract seeds using KeyBERT for contextualized keywords."""
        seeds = set()
        
        try:
            # Extract keywords with KeyBERT
            keywords = self.keybert.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=max_seeds,
                diversity=0.7  # Diverse keywords
            )
            
            for keyword, score in keywords:
                if score > 0.3 and self._is_valid_seed(keyword):  # Minimum relevance score
                    seeds.add(keyword.lower())
                    
        except Exception as e:
            self.logger.warning(f"KeyBERT extraction failed: {e}")
        
        return seeds
    
    def _extract_with_yake(self, content: str, max_seeds: int = 20) -> Set[str]:
        """Extract seeds using YAKE for unsupervised keyword extraction."""
        seeds = set()
        
        try:
            keywords = self.yake_extractor.extract_keywords(content)
            
            for item in keywords[:max_seeds]:
                # Handle both tuple and list formats from YAKE
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    keyword = str(item[1]).strip() if len(item) > 1 else str(item[0]).strip()
                    score = item[0] if len(item) > 0 else 1.0
                    
                    # Ensure score is numeric for comparison
                    try:
                        score = float(score) if score is not None else 1.0
                    except (ValueError, TypeError):
                        score = 1.0
                    
                    if score < 0.5 and self._is_valid_seed(keyword):  # YAKE uses lower scores for better keywords
                        seeds.add(keyword.lower().strip())
                elif isinstance(item, str):
                    # Handle case where YAKE returns just keywords
                    if self._is_valid_seed(item):
                        seeds.add(item.lower().strip())
                    
        except Exception as e:
            self.logger.warning(f"YAKE extraction failed: {e}")
        
        return seeds
    
    def _expand_with_llm(self, seed_keywords: List[str], content: str) -> Set[str]:
        """Expand seeds using LLM for semantic variations."""
        expanded_seeds = set()
        
        try:
            # Prepare prompt for LLM expansion
            seeds_text = ', '.join(seed_keywords[:15])  # Limit for prompt size
            content_sample = content[:500]  # Sample of content
            
            prompt = f"""Based on this website content and initial keywords, generate 15-20 additional relevant seed keywords for Google Ads campaigns.

Website content sample: {content_sample}

Initial keywords: {seeds_text}

Generate semantically related keywords that potential customers might search for. Include:
- Transactional keywords (buy, order, purchase)
- Informational keywords (how to, guide, tips)
- Commercial keywords (best, top, compare)
- Location-based variations if relevant

Return as a simple comma-separated list of keywords (2-4 words each)."""

            # Use Gemini or OpenAI
            if self.client_type == "gemini" and self.gemini_client:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                response_text = self.gemini_client.generate_content(
                    full_prompt,
                    max_tokens=800,
                    temperature=0.3
                )
            elif self.client_type == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_config.get('model', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": "You are a keyword research expert helping generate relevant search terms for Google Ads campaigns."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.llm_config.get('max_tokens', 400),
                    temperature=0.3
                )
                response_text = response.choices[0].message.content.strip()
            else:
                return seeds  # Return original seeds if no LLM available
            
            # Parse LLM response
            llm_keywords = [kw.strip().lower() for kw in response_text.split(',')]
            
            for keyword in llm_keywords:
                if self._is_valid_seed(keyword):
                    expanded_seeds.add(keyword)
                    
        except Exception as e:
            self.logger.warning(f"LLM expansion failed: {e}")
        
        return expanded_seeds
    
    def _extract_seeds_from_content(self, content: Dict) -> Set[str]:
        """Extract potential seed keywords from single content item."""
        seeds = set()
        
        # High-value sources
        title = content.get('title', '')
        meta_desc = content.get('meta_description', '')
        
        # Extract from title (highest weight)
        if title:
            seeds.update(self._extract_noun_phrases(title))
        
        # Extract from meta description
        if meta_desc:
            seeds.update(self._extract_noun_phrases(meta_desc))
        
        # Extract from headings
        headings = content.get('headings', {})
        for level in ['h1', 'h2', 'h3']:
            for heading in headings.get(level, []):
                if heading:
                    seeds.update(self._extract_noun_phrases(heading))
        
        # Extract from body text (limited sample)
        body_text = content.get('body_text', '')
        if body_text:
            # Sample first 500 words to avoid processing too much
            words = body_text.split()[:500]
            sample_text = ' '.join(words)
            seeds.update(self._extract_noun_phrases(sample_text))
        
        return seeds
    
    def _extract_noun_phrases(self, text: str) -> Set[str]:
        """Extract noun phrases using simple patterns."""
        if not text:
            return set()
        
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        phrases = set()
        words = text.split()
        
        # Extract 1-4 word phrases
        for length in range(1, 5):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                
                if self._is_valid_seed(phrase):
                    phrases.add(phrase)
        
        return phrases
    
    def _is_valid_seed(self, phrase: str) -> bool:
        """Check if phrase is a valid seed keyword."""
        # Skip very short or very long phrases
        if len(phrase) < 3 or len(phrase) > 50:
            return False
        
        # Skip phrases with only stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now'
        }
        
        words = phrase.split()
        
        # Require at least one non-stop word
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        if len(meaningful_words) == 0:
            return False
        
        # Skip phrases that are all numbers
        if all(w.isdigit() for w in words):
            return False
        
        # Skip common web elements
        web_words = {'home', 'page', 'website', 'site', 'click', 'here', 'read', 'more', 'learn'}
        if any(w in web_words for w in words):
            return False
        
        return True
    
    def _score_seeds(self, seeds: List[str], site_contents: List[Dict]) -> Dict[str, float]:
        """Score seeds based on frequency and context."""
        scores = {}
        
        # Count frequency across all content
        all_text = []
        for content in site_contents:
            text_parts = [
                content.get('title', ''),
                content.get('meta_description', ''),
                ' '.join(content.get('headings', {}).get('h1', [])),
                ' '.join(content.get('headings', {}).get('h2', [])),
                content.get('body_text', '')[:1000]  # Sample body text
            ]
            all_text.append(' '.join(text_parts).lower())
        
        combined_text = ' '.join(all_text)
        
        for seed in seeds:
            score = 0.0
            
            # Base frequency score
            frequency = combined_text.count(seed.lower())
            score += frequency * 1.0
            
            # Bonus for appearing in titles
            title_appearances = sum(1 for content in site_contents 
                                  if seed.lower() in content.get('title', '').lower())
            score += title_appearances * 5.0
            
            # Bonus for appearing in meta descriptions
            meta_appearances = sum(1 for content in site_contents 
                                 if seed.lower() in content.get('meta_description', '').lower())
            score += meta_appearances * 3.0
            
            # Bonus for appearing in headings
            heading_appearances = 0
            for content in site_contents:
                headings = content.get('headings', {})
                for level_headings in headings.values():
                    heading_appearances += sum(1 for h in level_headings 
                                             if seed.lower() in h.lower())
            score += heading_appearances * 2.0
            
            # Penalty for very common words
            if frequency > 50:
                score *= 0.5
            
            # Bonus for multi-word phrases
            if len(seed.split()) > 1:
                score *= 1.5
            
            scores[seed] = score
        
        return scores
    
    def expand_seeds_with_variations(self, seeds: List[str]) -> List[str]:
        """Expand seeds with common variations and related terms."""
        expanded_seeds = set(seeds)
        
        for seed in seeds:
            # Add plural/singular variations
            variations = self._generate_variations(seed)
            expanded_seeds.update(variations)
        
        return list(expanded_seeds)
    
    def _generate_variations(self, seed: str) -> List[str]:
        """Generate simple variations of a seed keyword."""
        variations = []
        
        # Simple pluralization
        if not seed.endswith('s'):
            variations.append(seed + 's')
        elif seed.endswith('s') and len(seed) > 3:
            variations.append(seed[:-1])  # Remove 's'
        
        # Add common modifiers
        modifiers = ['best', 'top', 'free', 'online', 'software', 'tool', 'service', 'solution']
        for modifier in modifiers:
            if modifier not in seed:
                variations.append(f"{modifier} {seed}")
                variations.append(f"{seed} {modifier}")
        
        return variations[:5]  # Limit variations per seed