"""Keyword normalization and deduplication."""

import re
import spacy
from typing import List, Dict, Set
from fuzzywuzzy import fuzz
from collections import defaultdict

from ..utils.logging import get_logger


class KeywordNormalizer:
    """Normalizes and deduplicates keywords using NLP techniques."""
    
    def __init__(self):
        self.logger = get_logger('KeywordNormalizer')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Stop words to preserve (commercial intent)
        self.preserve_words = {
            'buy', 'purchase', 'order', 'shop', 'store', 'price', 'cost', 'cheap', 'discount',
            'best', 'top', 'compare', 'review', 'near me', 'local', 'online', 'free',
            'service', 'services', 'tool', 'tools', 'software', 'platform', 'solution'
        }
    
    def normalize_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Normalize and deduplicate keyword list.
        
        Args:
            keywords: List of keyword dictionaries
            
        Returns:
            List of normalized and deduplicated keywords
        """
        self.logger.info(f"Normalizing {len(keywords)} keywords")
        
        # Step 1: Basic text normalization
        normalized_keywords = []
        for kw in keywords:
            normalized_kw = self._normalize_keyword_text(kw)
            if normalized_kw:
                normalized_keywords.append(normalized_kw)
        
        # Step 2: Lemmatization (if spaCy available)
        if self.nlp:
            normalized_keywords = self._lemmatize_keywords(normalized_keywords)
        
        # Step 3: Fuzzy deduplication
        deduplicated_keywords = self._fuzzy_deduplicate(normalized_keywords)
        
        # Step 4: Merge duplicate data
        final_keywords = self._merge_duplicates(deduplicated_keywords)
        
        self.logger.info(f"Normalized to {len(final_keywords)} unique keywords")
        return final_keywords
    
    def _normalize_keyword_text(self, keyword_dict: Dict) -> Dict:
        """Normalize individual keyword text."""
        original_keyword = keyword_dict.get('keyword', '').strip()
        
        if not original_keyword:
            return None
        
        # Convert to lowercase
        normalized = original_keyword.lower()
        
        # Remove extra punctuation but preserve commercial terms
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove very short or very long keywords
        if len(normalized) < 2 or len(normalized) > 100:
            return None
        
        # Skip if only numbers
        if normalized.replace(' ', '').isdigit():
            return None
        
        # Create normalized copy
        normalized_dict = keyword_dict.copy()
        normalized_dict['keyword'] = normalized
        normalized_dict['original_keyword'] = original_keyword
        
        return normalized_dict
    
    def _lemmatize_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Lemmatize keywords using spaCy."""
        lemmatized_keywords = []
        
        for kw_dict in keywords:
            keyword = kw_dict['keyword']
            
            try:
                # Process with spaCy
                doc = self.nlp(keyword)
                
                # Lemmatize while preserving important commercial terms
                lemmatized_tokens = []
                for token in doc:
                    if token.text.lower() in self.preserve_words:
                        # Preserve commercial terms as-is
                        lemmatized_tokens.append(token.text.lower())
                    elif not token.is_stop or token.text.lower() in self.preserve_words:
                        # Lemmatize non-stop words
                        lemmatized_tokens.append(token.lemma_.lower())
                
                lemmatized_keyword = ' '.join(lemmatized_tokens).strip()
                
                if lemmatized_keyword:
                    kw_dict_copy = kw_dict.copy()
                    kw_dict_copy['keyword'] = lemmatized_keyword
                    kw_dict_copy['lemmatized'] = True
                    lemmatized_keywords.append(kw_dict_copy)
                
            except Exception as e:
                self.logger.warning(f"Lemmatization failed for '{keyword}': {e}")
                # Keep original if lemmatization fails
                lemmatized_keywords.append(kw_dict)
        
        return lemmatized_keywords
    
    def _fuzzy_deduplicate(self, keywords: List[Dict], similarity_threshold: int = 85) -> List[Dict]:
        """Remove fuzzy duplicates using token set ratio."""
        if len(keywords) <= 1:
            return keywords
        
        # Group similar keywords
        groups = []
        processed = set()
        
        for i, kw1 in enumerate(keywords):
            if i in processed:
                continue
            
            group = [kw1]
            processed.add(i)
            
            for j, kw2 in enumerate(keywords[i+1:], i+1):
                if j in processed:
                    continue
                
                # Calculate similarity
                similarity = fuzz.token_set_ratio(kw1['keyword'], kw2['keyword'])
                
                if similarity >= similarity_threshold:
                    group.append(kw2)
                    processed.add(j)
            
            groups.append(group)
        
        # Select best representative from each group
        deduplicated = []
        for group in groups:
            best_keyword = self._select_best_from_group(group)
            deduplicated.append(best_keyword)
        
        return deduplicated
    
    def _select_best_from_group(self, group: List[Dict]) -> Dict:
        """Select the best keyword from a group of similar keywords."""
        if len(group) == 1:
            return group[0]
        
        # Scoring criteria for best keyword
        def score_keyword(kw):
            score = 0
            
            # Prefer higher volume
            score += kw.get('volume', 0) * 0.4
            
            # Prefer higher confidence
            score += kw.get('confidence', 0) * 100 * 0.3
            
            # Prefer certain sources
            source_weights = {
                'wordstream': 1.0,
                'google_autocomplete': 0.8,
                'people_also_ask': 0.6,
                'generated': 0.4
            }
            score += source_weights.get(kw.get('source', ''), 0.2) * 100 * 0.2
            
            # Prefer longer, more descriptive keywords
            score += len(kw['keyword'].split()) * 10 * 0.1
            
            return score
        
        # Return highest scoring keyword
        best_keyword = max(group, key=score_keyword)
        
        # Merge volume data from all keywords in group
        total_volume = sum(kw.get('volume', 0) for kw in group)
        best_keyword['volume'] = max(best_keyword.get('volume', 0), total_volume // len(group))
        best_keyword['duplicate_count'] = len(group)
        
        return best_keyword
    
    def _merge_duplicates(self, keywords: List[Dict]) -> List[Dict]:
        """Final pass to merge any remaining exact duplicates."""
        keyword_map = {}
        
        for kw in keywords:
            key = kw['keyword']
            
            if key in keyword_map:
                # Merge data from duplicate
                existing = keyword_map[key]
                
                # Take higher volume
                existing['volume'] = max(
                    existing.get('volume', 0),
                    kw.get('volume', 0)
                )
                
                # Take better CPC data
                if kw.get('cpc_high', 0) > existing.get('cpc_high', 0):
                    existing['cpc_low'] = kw.get('cpc_low', existing.get('cpc_low', 0))
                    existing['cpc_high'] = kw.get('cpc_high', existing.get('cpc_high', 0))
                
                # Take higher confidence
                existing['confidence'] = max(
                    existing.get('confidence', 0),
                    kw.get('confidence', 0)
                )
                
                # Combine sources
                existing_source = existing.get('source', '')
                new_source = kw.get('source', '')
                if new_source and new_source != existing_source:
                    existing['source'] = f"{existing_source},{new_source}"
                
            else:
                keyword_map[key] = kw
        
        return list(keyword_map.values())
    
    def filter_keywords(self, keywords: List[Dict], filters: Dict) -> List[Dict]:
        """Filter keywords based on criteria."""
        filtered = []
        
        min_volume = filters.get('min_search_volume', 0)
        max_cpc = filters.get('max_cpc', float('inf'))
        
        for kw in keywords:
            volume = kw.get('volume', 0)
            cpc_high = kw.get('cpc_high', 0)
            
            # Volume filter
            if volume < min_volume:
                continue
            
            # CPC filter
            if cpc_high > max_cpc:
                continue
            
            # Quality filters
            keyword_text = kw['keyword']
            
            # Skip very short keywords
            if len(keyword_text) < 3:
                continue
            
            # Skip keywords with too many repetitions
            words = keyword_text.split()
            if len(set(words)) < len(words) * 0.7:  # Too many repeated words
                continue
            
            filtered.append(kw)
        
        self.logger.info(f"Filtered to {len(filtered)} keywords after applying criteria")
        return filtered