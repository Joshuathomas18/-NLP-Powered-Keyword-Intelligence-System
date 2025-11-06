"""LLM-based intent classification for keywords."""

import json
from typing import List, Dict, Optional
import os

from ..utils.logging import get_logger
from ..llm.gemini_client import GeminiClient


class IntentClassifier:
    """Classifies keyword search intent using LLM."""
    
    def __init__(self, llm_config: Dict):
        """Initialize intent classifier.
        
        Args:
            llm_config: LLM configuration from config
        """
        self.logger = get_logger('IntentClassifier')
        self.config = llm_config
        
        # Initialize Gemini client (preferred) or OpenAI
        gemini_key = os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # Try Gemini first (free, better for this use case)
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                self.client = GeminiClient(gemini_key, "gemini-2.0-flash-exp")
                self.client_type = "gemini"
                self.logger.info("Using Gemini for intent classification")
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
                self.logger.info("Using OpenAI for intent classification")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                self.client = None
                self.client_type = None
        else:
            self.logger.warning("No LLM API key found. Using rule-based classification.")
            self.client = None
            self.client_type = None
    
    def classify_keywords(self, keywords: List[Dict], batch_size: int = 20) -> List[Dict]:
        """Classify intent for a list of keywords.
        
        Args:
            keywords: List of keyword dictionaries
            batch_size: Number of keywords to process per API call
            
        Returns:
            List of keywords with intent classifications
        """
        if not self.client:
            return self._fallback_classification(keywords)
        
        self.logger.info(f"Classifying intent for {len(keywords)} keywords")
        
        classified_keywords = []
        
        # Process in batches to avoid token limits
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            batch_results = self._classify_batch(batch)
            classified_keywords.extend(batch_results)
        
        return classified_keywords
    
    def _classify_batch(self, keyword_batch: List[Dict]) -> List[Dict]:
        """Classify a batch of keywords."""
        try:
            # Prepare prompt
            keyword_list = [kw['keyword'] for kw in keyword_batch]
            prompt = self._build_classification_prompt(keyword_list)
            
            # Call LLM (Gemini or OpenAI)
            if self.client_type == "gemini":
                # Use Gemini
                full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
                response_text = self.client.generate_content(
                    full_prompt,
                    max_tokens=self.config.get('max_tokens', 800),
                    temperature=0.1
                )
            elif self.client_type == "openai":
                # Use OpenAI
                response = self.client.chat.completions.create(
                    model=self.config.get('model', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.get('max_tokens', 800),
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()
            else:
                return self._fallback_classification(keyword_batch)
            
            # Parse response
            classifications = self._parse_llm_response(response_text, keyword_batch)
            
            return classifications
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            return self._fallback_classification(keyword_batch)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for intent classification."""
        return """You are an expert in search marketing and keyword analysis. Your task is to classify search keywords into one of four intent categories:

1. TRANSACTIONAL: Keywords indicating intent to purchase, buy, or take action
   - Examples: "buy protein powder", "order pizza online", "download software"

2. INFORMATIONAL: Keywords seeking information, answers, or learning
   - Examples: "how to lose weight", "what is python", "best practices for SEO"

3. NAVIGATIONAL: Keywords seeking a specific website or brand
   - Examples: "facebook login", "amazon customer service", "nike official website"

4. COMMERCIAL: Keywords showing research intent before purchase
   - Examples: "best laptop 2024", "protein powder reviews", "vs comparisons"

Respond with a JSON array containing objects with "keyword" and "intent" fields. Be precise and consistent."""
    
    def _build_classification_prompt(self, keywords: List[str]) -> str:
        """Build classification prompt for a batch of keywords."""
        keywords_text = '\n'.join([f"- {kw}" for kw in keywords])
        
        return f"""Classify the search intent for each of these keywords:

{keywords_text}

Return your response as a JSON array with this exact format:
[
  {{"keyword": "example keyword", "intent": "transactional"}},
  {{"keyword": "another keyword", "intent": "informational"}}
]

Ensure every keyword is classified with one of: transactional, informational, navigational, commercial"""
    
    def _parse_llm_response(self, response_text: str, original_keywords: List[Dict]) -> List[Dict]:
        """Parse LLM response and merge with original keyword data."""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                classifications = json.loads(json_text)
                
                # Create lookup for quick matching
                intent_lookup = {item['keyword']: item['intent'] for item in classifications}
                
                # Merge with original keyword data
                classified_keywords = []
                for kw_data in original_keywords:
                    keyword = kw_data['keyword']
                    intent = intent_lookup.get(keyword, 'commercial')  # Default fallback
                    
                    # Add intent to keyword data
                    kw_data_copy = kw_data.copy()
                    kw_data_copy['intent'] = intent
                    kw_data_copy['intent_confidence'] = 0.9  # High confidence for LLM
                    classified_keywords.append(kw_data_copy)
                
                return classified_keywords
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM JSON response: {e}")
        except Exception as e:
            self.logger.warning(f"Error processing LLM response: {e}")
        
        # Fallback if parsing fails
        return self._fallback_classification(original_keywords)
    
    def _fallback_classification(self, keywords: List[Dict]) -> List[Dict]:
        """Fallback classification using rule-based approach."""
        self.logger.info("Using fallback rule-based intent classification")
        
        classified_keywords = []
        
        for kw_data in keywords:
            keyword = kw_data['keyword'].lower()
            intent = self._rule_based_intent(keyword)
            
            kw_data_copy = kw_data.copy()
            kw_data_copy['intent'] = intent
            kw_data_copy['intent_confidence'] = 0.6  # Lower confidence for rules
            classified_keywords.append(kw_data_copy)
        
        return classified_keywords
    
    def _rule_based_intent(self, keyword: str) -> str:
        """Rule-based intent classification."""
        keyword_lower = keyword.lower()
        
        # Transactional signals
        transactional_signals = [
            'buy', 'purchase', 'order', 'shop', 'store', 'cart', 'checkout',
            'price', 'cost', 'cheap', 'discount', 'sale', 'deal', 'offer',
            'download', 'install', 'get', 'subscribe', 'sign up'
        ]
        
        # Informational signals
        informational_signals = [
            'how', 'what', 'why', 'when', 'where', 'guide', 'tutorial',
            'tips', 'help', 'learn', 'course', 'training', 'explain',
            'definition', 'meaning', 'example'
        ]
        
        # Navigational signals
        navigational_signals = [
            'login', 'sign in', 'website', 'official', 'homepage',
            'contact', 'support', 'customer service', 'account'
        ]
        
        # Commercial investigation signals
        commercial_signals = [
            'best', 'top', 'compare', 'comparison', 'review', 'reviews',
            'vs', 'versus', 'alternative', 'option', 'solution',
            'recommendation', 'rating'
        ]
        
        # Check for signals
        if any(signal in keyword_lower for signal in transactional_signals):
            return 'transactional'
        elif any(signal in keyword_lower for signal in informational_signals):
            return 'informational'
        elif any(signal in keyword_lower for signal in navigational_signals):
            return 'navigational'
        elif any(signal in keyword_lower for signal in commercial_signals):
            return 'commercial'
        
        # Default classification based on keyword structure
        words = keyword_lower.split()
        
        # Short keywords often navigational
        if len(words) <= 2 and not any(signal in keyword_lower for signal in transactional_signals):
            return 'navigational'
        
        # Questions are usually informational
        if keyword_lower.startswith(('how', 'what', 'why', 'when', 'where')):
            return 'informational'
        
        # Default to commercial investigation
        return 'commercial'
    
    def classify_single_keyword(self, keyword: str, context: str = "") -> Dict:
        """Classify intent for a single keyword with optional context.
        
        Args:
            keyword: Keyword to classify
            context: Optional context about the business/website
            
        Returns:
            Dictionary with keyword and intent classification
        """
        if not self.client:
            return {
                'keyword': keyword,
                'intent': self._rule_based_intent(keyword),
                'intent_confidence': 0.6
            }
        
        try:
            prompt = f"""Classify the search intent for this keyword: "{keyword}"
            
            {f"Business context: {context}" if context else ""}
            
            Return a JSON object with this format:
            {{"keyword": "{keyword}", "intent": "one_of_four_intents"}}
            
            Intent options: transactional, informational, navigational, commercial"""
            
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(response_text)
            result['intent_confidence'] = 0.9
            return result
            
        except Exception as e:
            self.logger.warning(f"Single keyword classification failed: {e}")
            return {
                'keyword': keyword,
                'intent': self._rule_based_intent(keyword),
                'intent_confidence': 0.6
            }