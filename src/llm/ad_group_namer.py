"""LLM-based ad group naming and theme generation."""

import json
from typing import List, Dict, Optional
from openai import OpenAI
import os
import re

from ..utils.logging import get_logger


class AdGroupNamer:
    """Generates semantic ad group names and themes using LLM."""
    
    def __init__(self, llm_config: Dict):
        """Initialize ad group namer.
        
        Args:
            llm_config: LLM configuration from config
        """
        self.logger = get_logger('AdGroupNamer')
        self.config = llm_config
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("No OpenAI API key found. Using fallback naming.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
    
    def generate_ad_group_names(self, clusters: List[Dict], business_context: str = "") -> List[Dict]:
        """Generate semantic names for ad groups based on keyword clusters.
        
        Args:
            clusters: List of keyword clusters
            business_context: Business context for better naming
            
        Returns:
            List of clusters with generated ad group names
        """
        self.logger.info(f"Generating names for {len(clusters)} ad groups")
        
        named_clusters = []
        
        for cluster in clusters:
            if self.client and len(cluster.get('keywords', [])) >= 3:
                # Use LLM for substantial clusters
                ad_group_name = self._generate_llm_name(cluster, business_context)
            else:
                # Use rule-based for small clusters
                ad_group_name = self._generate_rule_based_name(cluster)
            
            cluster_copy = cluster.copy()
            cluster_copy['ad_group_name'] = ad_group_name
            cluster_copy['original_cluster_id'] = cluster.get('cluster_id', '')
            named_clusters.append(cluster_copy)
        
        return named_clusters
    
    def _generate_llm_name(self, cluster: Dict, business_context: str) -> str:
        """Generate ad group name using LLM."""
        try:
            keywords = [kw['keyword'] for kw in cluster.get('keywords', [])]
            sample_keywords = keywords[:10]  # Use top 10 keywords
            
            # Analyze cluster characteristics
            intents = [kw.get('intent', 'commercial') for kw in cluster.get('keywords', [])]
            dominant_intent = max(set(intents), key=intents.count)
            
            prompt = self._build_naming_prompt(sample_keywords, dominant_intent, business_context)
            
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": self._get_naming_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3  # Slightly higher for creativity
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response to extract clean name
            ad_group_name = self._parse_naming_response(response_text)
            
            # Validate and clean the name
            return self._validate_ad_group_name(ad_group_name)
            
        except Exception as e:
            self.logger.warning(f"LLM naming failed: {e}")
            return self._generate_rule_based_name(cluster)
    
    def _get_naming_system_prompt(self) -> str:
        """System prompt for ad group naming."""
        return """You are a Google Ads expert specializing in creating effective ad group names. Your task is to generate concise, descriptive ad group names that:

1. Clearly describe the keyword theme (2-4 words ideal)
2. Include relevant commercial terms when appropriate
3. Follow Google Ads best practices:
   - 25 characters or less preferred
   - Clear and specific
   - Avoid generic terms
   - Include intent indicators when relevant

4. Consider search intent:
   - Transactional: Include action words (Buy, Shop, Order)
   - Informational: Include learning words (Guide, Tips, How-to)
   - Commercial: Include comparison words (Best, Top, Compare)
   - Navigational: Include brand/specific terms

Examples of good ad group names:
- "Buy Protein Powder"
- "iPhone 15 Reviews" 
- "Local Pizza Delivery"
- "SEO Tools Comparison"

Respond with just the ad group name, no additional text."""
    
    def _build_naming_prompt(self, keywords: List[str], intent: str, business_context: str) -> str:
        """Build prompt for ad group naming."""
        keywords_text = '\n'.join([f"- {kw}" for kw in keywords])
        
        context_text = f"\nBusiness context: {business_context}" if business_context else ""
        
        return f"""Create a concise, descriptive ad group name for these related keywords:

{keywords_text}

Dominant search intent: {intent}
{context_text}

Requirements:
- 2-4 words maximum
- Under 25 characters if possible
- Descriptive of the keyword theme
- Include commercial intent when appropriate
- Professional and clear

Ad group name:"""
    
    def _parse_naming_response(self, response_text: str) -> str:
        """Parse LLM response to extract clean ad group name."""
        # Remove common response prefixes
        response_text = re.sub(r'^(ad group name:|name:|response:|suggested name:)', '', response_text, flags=re.IGNORECASE)
        
        # Remove quotes and extra whitespace
        response_text = response_text.strip().strip('"\'')
        
        # Take first line if multiple lines
        first_line = response_text.split('\n')[0].strip()
        
        return first_line
    
    def _validate_ad_group_name(self, name: str) -> str:
        """Validate and clean ad group name."""
        if not name or len(name) < 2:
            return "Keywords Group"
        
        # Truncate if too long
        if len(name) > 30:
            words = name.split()
            name = ' '.join(words[:3])  # Take first 3 words
        
        # Capitalize properly
        name = name.title()
        
        # Remove invalid characters
        name = re.sub(r'[^\w\s\-]', '', name)
        
        return name
    
    def _generate_rule_based_name(self, cluster: Dict) -> str:
        """Generate ad group name using rule-based approach."""
        keywords = cluster.get('keywords', [])
        if not keywords:
            return "Empty Group"
        
        # Extract common themes
        all_words = []
        for kw_data in keywords:
            words = kw_data['keyword'].lower().split()
            all_words.extend(words)
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Filter out stop words but keep commercial terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        meaningful_words = [
            word for word, count in word_counts.most_common(10)
            if word not in stop_words and len(word) > 2 and count > 1
        ]
        
        if meaningful_words:
            # Create name from most common meaningful words
            name_words = meaningful_words[:3]
            ad_group_name = ' '.join(name_words).title()
        else:
            # Fallback to first keyword theme
            first_keyword = keywords[0]['keyword']
            words = first_keyword.split()[:3]
            ad_group_name = ' '.join(words).title()
        
        # Add intent-based suffix if clear pattern
        intents = [kw.get('intent', 'commercial') for kw in keywords]
        dominant_intent = max(set(intents), key=intents.count)
        
        if dominant_intent == 'transactional' and 'buy' not in ad_group_name.lower():
            ad_group_name = f"Buy {ad_group_name}"
        elif dominant_intent == 'informational' and not any(word in ad_group_name.lower() for word in ['guide', 'tips', 'how']):
            ad_group_name = f"{ad_group_name} Guide"
        
        return self._validate_ad_group_name(ad_group_name)
    
    def generate_pmax_themes(self, top_clusters: List[Dict], business_context: str = "") -> List[Dict]:
        """Generate Performance Max asset themes from top keyword clusters.
        
        Args:
            top_clusters: Top performing keyword clusters
            business_context: Business context for relevant themes
            
        Returns:
            List of PMax theme dictionaries
        """
        self.logger.info(f"Generating PMax themes from {len(top_clusters)} clusters")
        
        themes = []
        
        for i, cluster in enumerate(top_clusters[:8]):  # Max 8 themes
            theme = self._generate_single_theme(cluster, business_context, i + 1)
            themes.append(theme)
        
        return themes
    
    def _generate_single_theme(self, cluster: Dict, business_context: str, theme_number: int) -> Dict:
        """Generate a single PMax theme from a cluster."""
        keywords = [kw['keyword'] for kw in cluster.get('keywords', [])]
        ad_group_name = cluster.get('ad_group_name', f"Theme {theme_number}")
        
        if self.client:
            try:
                theme_content = self._generate_llm_theme(keywords, ad_group_name, business_context)
            except Exception as e:
                self.logger.warning(f"LLM theme generation failed: {e}")
                theme_content = self._generate_rule_based_theme(keywords, ad_group_name)
        else:
            theme_content = self._generate_rule_based_theme(keywords, ad_group_name)
        
        return {
            'theme_name': ad_group_name,
            'keywords': keywords[:10],  # Top 10 keywords
            'headlines': theme_content['headlines'],
            'long_headlines': theme_content['long_headlines'],
            'descriptions': theme_content['descriptions'],
            'callouts': theme_content['callouts'],
            'total_volume': cluster.get('total_volume', 0),
            'keyword_count': len(keywords)
        }
    
    def _generate_llm_theme(self, keywords: List[str], theme_name: str, business_context: str) -> Dict:
        """Generate PMax theme content using LLM."""
        prompt = f"""Create Performance Max ad assets for this keyword theme: "{theme_name}"

Keywords in this theme: {', '.join(keywords[:8])}
Business context: {business_context}

Generate the following ad assets:

1. Headlines (3-5 options, max 30 characters each):
   - Action-oriented and compelling
   - Include key terms from keywords
   - Clear value proposition

2. Long Headlines (2-3 options, max 90 characters each):
   - More descriptive and detailed
   - Include benefits and features

3. Descriptions (3-4 options, max 90 characters each):
   - Compelling reasons to choose your business
   - Include call-to-action

4. Callouts (4-6 options, max 25 characters each):
   - Key features and benefits
   - Unique selling points

Return as JSON with arrays for each asset type."""
        
        response = self.client.chat.completions.create(
            model=self.config.get('model', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "You are a Google Ads copywriter expert. Create compelling ad copy that follows Google Ads policies and character limits."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.4
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            # Try to parse JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                theme_content = json.loads(json_text)
                
                # Validate and clean the content
                return self._validate_theme_content(theme_content)
        
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM theme JSON")
        
        # Fallback to rule-based
        return self._generate_rule_based_theme(keywords, theme_name)
    
    def _generate_rule_based_theme(self, keywords: List[str], theme_name: str) -> Dict:
        """Generate PMax theme content using rule-based approach."""
        # Extract key terms from keywords
        all_words = ' '.join(keywords).lower()
        
        # Generate headlines
        headlines = [
            f"Best {theme_name}",
            f"Top {theme_name}",
            f"Professional {theme_name}",
            f"Quality {theme_name}",
            f"Get {theme_name}"
        ]
        
        # Generate long headlines
        long_headlines = [
            f"Get the Best {theme_name} Solutions Today",
            f"Professional {theme_name} Services & Products",
            f"Top-Rated {theme_name} at Great Prices"
        ]
        
        # Generate descriptions
        descriptions = [
            f"Transform your business with our {theme_name.lower()} solutions.",
            f"Expert {theme_name.lower()} services you can trust.",
            f"Get started with {theme_name.lower()} today.",
            f"Quality {theme_name.lower()} at competitive prices."
        ]
        
        # Generate callouts
        callouts = [
            "Expert Service",
            "Quality Products",
            "Fast Delivery",
            "Great Prices",
            "Trusted Brand",
            "24/7 Support"
        ]
        
        return {
            'headlines': headlines[:5],
            'long_headlines': long_headlines,
            'descriptions': descriptions,
            'callouts': callouts
        }
    
    def _validate_theme_content(self, content: Dict) -> Dict:
        """Validate and clean theme content for character limits."""
        validated = {
            'headlines': [],
            'long_headlines': [],
            'descriptions': [],
            'callouts': []
        }
        
        # Validate headlines (30 char limit)
        for headline in content.get('headlines', []):
            if isinstance(headline, str) and len(headline) <= 30:
                validated['headlines'].append(headline)
        
        # Validate long headlines (90 char limit)
        for long_headline in content.get('long_headlines', []):
            if isinstance(long_headline, str) and len(long_headline) <= 90:
                validated['long_headlines'].append(long_headline)
        
        # Validate descriptions (90 char limit)
        for description in content.get('descriptions', []):
            if isinstance(description, str) and len(description) <= 90:
                validated['descriptions'].append(description)
        
        # Validate callouts (25 char limit)
        for callout in content.get('callouts', []):
            if isinstance(callout, str) and len(callout) <= 25:
                validated['callouts'].append(callout)
        
        # Ensure minimum content
        if len(validated['headlines']) < 3:
            validated['headlines'].extend([
                "Quality Service",
                "Best Solutions",
                "Expert Help"
            ])
        
        return validated