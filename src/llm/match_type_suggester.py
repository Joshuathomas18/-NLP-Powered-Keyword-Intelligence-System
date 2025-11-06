"""LLM-based match type suggestions for keywords."""

import json
from typing import List, Dict, Optional
import os

from ..utils.logging import get_logger
from ..llm.gemini_client import GeminiClient


class MatchTypeSuggester:
    """Suggests Google Ads match types for keywords using LLM and rules."""
    
    def __init__(self, llm_config: Dict):
        """Initialize match type suggester.
        
        Args:
            llm_config: LLM configuration from config
        """
        self.logger = get_logger('MatchTypeSuggester')
        self.config = llm_config
        
        # Initialize Gemini client (preferred) or OpenAI
        gemini_key = os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # Try Gemini first (free, better for this use case)
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                self.client = GeminiClient(gemini_key, "gemini-2.0-flash-exp")
                self.client_type = "gemini"
                self.logger.info("Using Gemini for match type suggestions")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini: {e}")
                self.client = None
                self.client_type = None
        # Fallback to OpenAI if available
        elif openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
                self.client_type = "openai"
                self.logger.info("Using OpenAI for match type suggestions")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                self.client = None
                self.client_type = None
        else:
            self.logger.warning("No LLM API key found. Using rule-based match type suggestions.")
            self.client = None
            self.client_type = None
    
    def suggest_match_types(self, keywords: List[Dict], budget_level: str = "medium") -> List[Dict]:
        """Suggest match types for a list of keywords.
        
        Args:
            keywords: List of keyword dictionaries with intent classifications
            budget_level: Budget level affecting match type strategy ("low", "medium", "high")
            
        Returns:
            List of keywords with match type suggestions
        """
        self.logger.info(f"Suggesting match types for {len(keywords)} keywords")
        
        if self.client and len(keywords) > 50:
            # Use LLM for large batches
            return self._llm_batch_suggestions(keywords, budget_level)
        else:
            # Use rule-based approach
            return self._rule_based_suggestions(keywords, budget_level)
    
    def _llm_batch_suggestions(self, keywords: List[Dict], budget_level: str) -> List[Dict]:
        """Get match type suggestions from LLM in batches."""
        suggested_keywords = []
        batch_size = 15  # Smaller batches for match type analysis
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            try:
                batch_suggestions = self._suggest_batch_llm(batch, budget_level)
                suggested_keywords.extend(batch_suggestions)
            except Exception as e:
                self.logger.warning(f"LLM batch failed, using fallback: {e}")
                # Fallback to rule-based for this batch
                fallback_batch = self._rule_based_suggestions(batch, budget_level)
                suggested_keywords.extend(fallback_batch)
        
        return suggested_keywords
    
    def _suggest_batch_llm(self, keyword_batch: List[Dict], budget_level: str) -> List[Dict]:
        """Get LLM suggestions for a batch of keywords."""
        # Prepare keyword data for LLM
        keyword_data = []
        for kw in keyword_batch:
            keyword_data.append({
                'keyword': kw['keyword'],
                'intent': kw.get('intent', 'commercial'),
                'volume': kw.get('volume', 0),
                'cpc_high': kw.get('cpc_high', 0),
                'competition': kw.get('competition', 0.5)
            })
        
        prompt = self._build_match_type_prompt(keyword_data, budget_level)
        
        # Use Gemini or OpenAI
        if self.client_type == "gemini":
            full_prompt = f"{self._get_match_type_system_prompt()}\n\n{prompt}"
            response_text = self.client.generate_content(
                full_prompt,
                max_tokens=self.config.get('max_tokens', 800),
                temperature=0.1
            )
        elif self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": self._get_match_type_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.get('max_tokens', 800),
                temperature=0.1
            )
            response_text = response.choices[0].message.content.strip()
        else:
            return self._rule_based_suggestions(keyword_batch, budget_level)
        
        return self._parse_match_type_response(response_text, keyword_batch)
    
    def _get_match_type_system_prompt(self) -> str:
        """System prompt for match type suggestions."""
        return """You are a Google Ads expert specializing in keyword match type strategy. Your task is to recommend the optimal match type for each keyword based on:

1. SEARCH INTENT:
   - Transactional: More restrictive match types (Exact, Phrase)
   - Informational: Broader match types (Broad, Phrase) 
   - Navigational: Exact match for brand terms
   - Commercial: Mix of Phrase and Broad

2. MATCH TYPES:
   - EXACT: [keyword] - Most restrictive, highest relevance
   - PHRASE: "keyword" - Medium control, good performance
   - BROAD: keyword - Widest reach, requires careful negatives

3. STRATEGY FACTORS:
   - High-value/branded terms → Exact
   - Long-tail specific terms → Phrase  
   - Discovery/awareness → Broad
   - High competition → More restrictive
   - Budget constraints → More restrictive

Respond with JSON array containing keyword and recommended match_type."""
    
    def _build_match_type_prompt(self, keyword_data: List[Dict], budget_level: str) -> str:
        """Build prompt for match type suggestions."""
        keywords_info = "\n".join([
            f"- {kw['keyword']} (intent: {kw['intent']}, volume: {kw['volume']}, cpc: ${kw['cpc_high']:.2f}, competition: {kw['competition']:.2f})"
            for kw in keyword_data
        ])
        
        budget_context = {
            'low': 'Limited budget - prefer more restrictive match types to control costs',
            'medium': 'Moderate budget - balanced approach between reach and control',
            'high': 'Generous budget - can afford broader match types for discovery'
        }
        
        return f"""Recommend the optimal match type (exact, phrase, or broad) for each keyword:

{keywords_info}

Budget context: {budget_context.get(budget_level, 'Moderate budget')}

Return JSON array with this format:
[
  {{"keyword": "example keyword", "match_type": "phrase", "reasoning": "brief explanation"}},
  {{"keyword": "another keyword", "match_type": "exact", "reasoning": "brief explanation"}}
]"""
    
    def _parse_match_type_response(self, response_text: str, original_keywords: List[Dict]) -> List[Dict]:
        """Parse LLM response and merge with original keyword data."""
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                suggestions = json.loads(json_text)
                
                # Create lookup
                match_type_lookup = {
                    item['keyword']: {
                        'match_type': item['match_type'],
                        'reasoning': item.get('reasoning', '')
                    }
                    for item in suggestions
                }
                
                # Merge with original data
                suggested_keywords = []
                for kw_data in original_keywords:
                    keyword = kw_data['keyword']
                    suggestion = match_type_lookup.get(keyword, {'match_type': 'phrase', 'reasoning': 'fallback'})
                    
                    kw_data_copy = kw_data.copy()
                    kw_data_copy['match_type'] = suggestion['match_type']
                    kw_data_copy['match_type_reasoning'] = suggestion['reasoning']
                    kw_data_copy['match_type_confidence'] = 0.9
                    suggested_keywords.append(kw_data_copy)
                
                return suggested_keywords
        
        except Exception as e:
            self.logger.warning(f"Failed to parse match type response: {e}")
        
        # Fallback to rule-based
        return self._rule_based_suggestions(original_keywords, 'medium')
    
    def _rule_based_suggestions(self, keywords: List[Dict], budget_level: str) -> List[Dict]:
        """Rule-based match type suggestions."""
        suggested_keywords = []
        
        for kw_data in keywords:
            keyword = kw_data['keyword']
            intent = kw_data.get('intent', 'commercial')
            volume = kw_data.get('volume', 0)
            competition = kw_data.get('competition', 0.5)
            cpc_high = kw_data.get('cpc_high', 0)
            
            match_type = self._determine_match_type_rules(
                keyword, intent, volume, competition, cpc_high, budget_level
            )
            
            kw_data_copy = kw_data.copy()
            kw_data_copy['match_type'] = match_type
            kw_data_copy['match_type_confidence'] = 0.7
            kw_data_copy['match_type_reasoning'] = self._get_rule_reasoning(
                keyword, intent, match_type, budget_level
            )
            suggested_keywords.append(kw_data_copy)
        
        return suggested_keywords
    
    def _determine_match_type_rules(self, keyword: str, intent: str, volume: int, 
                                   competition: float, cpc_high: float, budget_level: str) -> str:
        """Determine match type using rule-based logic."""
        keyword_lower = keyword.lower()
        word_count = len(keyword.split())
        
        # Brand terms and exact product names -> Exact
        if self._is_brand_term(keyword_lower) or self._is_product_specific(keyword_lower):
            return 'exact'
        
        # High-value transactional terms
        if intent == 'transactional':
            if volume > 1000 and competition > 0.7:
                return 'exact'  # High competition, need control
            elif word_count >= 3:
                return 'phrase'  # Specific transactional phrases
            else:
                return 'exact'   # Short transactional terms
        
        # Navigational terms -> Exact
        if intent == 'navigational':
            return 'exact'
        
        # Informational terms -> Broader for discovery
        if intent == 'informational':
            if word_count >= 4:
                return 'phrase'  # Long informational queries
            else:
                return 'broad'   # Short informational terms for discovery
        
        # Commercial investigation
        if intent == 'commercial':
            if budget_level == 'low' or competition > 0.8:
                return 'phrase'  # More control for budget/competition
            elif word_count >= 3:
                return 'phrase'  # Specific commercial terms
            else:
                return 'broad'   # Broader commercial discovery
        
        # Default logic based on keyword characteristics
        if word_count >= 4:
            return 'phrase'  # Long-tail terms
        elif volume < 100:
            return 'broad'   # Low volume terms need broader reach
        elif competition > 0.8:
            return 'exact'   # High competition needs control
        else:
            return 'phrase'  # Default balanced approach
    
    def _is_brand_term(self, keyword: str) -> bool:
        """Check if keyword contains brand terms."""
        brand_indicators = [
            'facebook', 'google', 'amazon', 'microsoft', 'apple', 'nike', 'adidas',
            'coca cola', 'pepsi', 'samsung', 'sony', 'bmw', 'mercedes', 'toyota'
        ]
        
        return any(brand in keyword for brand in brand_indicators)
    
    def _is_product_specific(self, keyword: str) -> bool:
        """Check if keyword is very product-specific."""
        specific_indicators = [
            'model', 'version', 'generation', 'series', 'edition',
            'sku', 'part number', 'isbn'
        ]
        
        # Check for model numbers or very specific product terms
        if any(indicator in keyword for indicator in specific_indicators):
            return True
        
        # Check for number patterns (model numbers, versions)
        import re
        if re.search(r'\b\w+\d+\w*\b', keyword):  # alphanumeric patterns
            return True
        
        return False
    
    def _get_rule_reasoning(self, keyword: str, intent: str, match_type: str, budget_level: str) -> str:
        """Generate reasoning for rule-based match type decision."""
        reasons = []
        
        if intent == 'transactional':
            reasons.append("transactional intent")
        elif intent == 'navigational':
            reasons.append("navigational/brand search")
        elif intent == 'informational':
            reasons.append("informational query")
        else:
            reasons.append("commercial investigation")
        
        word_count = len(keyword.split())
        if word_count >= 4:
            reasons.append("long-tail keyword")
        elif word_count <= 2:
            reasons.append("short keyword")
        
        if budget_level == 'low':
            reasons.append("budget-conscious strategy")
        
        if self._is_brand_term(keyword.lower()):
            reasons.append("contains brand terms")
        
        return f"{match_type} match: {', '.join(reasons)}"
    
    def suggest_negative_keywords(self, keywords: List[Dict], business_context: str = "") -> List[str]:
        """Suggest negative keywords to prevent irrelevant traffic."""
        negative_keywords = set()
        
        # Generic negative keywords for most businesses
        generic_negatives = [
            'free', 'cheap', 'download', 'crack', 'pirate', 'illegal',
            'job', 'jobs', 'career', 'salary', 'hiring', 'employment',
            'diy', 'homemade', 'tutorial', 'how to make',
            'used', 'second hand', 'refurbished'
        ]
        
        negative_keywords.update(generic_negatives)
        
        # Context-specific negatives based on business
        if 'software' in business_context.lower():
            software_negatives = ['free download', 'cracked', 'keygen', 'serial']
            negative_keywords.update(software_negatives)
        
        if 'service' in business_context.lower():
            service_negatives = ['diy', 'do it yourself', 'manual']
            negative_keywords.update(service_negatives)
        
        # Analyze keywords for patterns that might need negatives
        all_keywords = ' '.join([kw['keyword'] for kw in keywords]).lower()
        
        if 'price' in all_keywords or 'cost' in all_keywords:
            negative_keywords.update(['free', 'no cost', 'without charge'])
        
        return sorted(list(negative_keywords))