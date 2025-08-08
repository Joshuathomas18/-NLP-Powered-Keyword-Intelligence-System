"""Gemini AI client for advanced LLM features."""

import json
from typing import List, Dict, Optional, Any
import time

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..utils.logging import get_logger


class GeminiClient:
    """Client for Google Gemini AI with advanced features."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model: Model name to use
        """
        self.logger = get_logger('GeminiClient')
        self.api_key = api_key
        self.model_name = model
        self.model = None
        
        if not genai:
            self.logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.logger.info(f"Initialized Gemini client with model: {model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            self.model = None
    
    def generate_content(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.1) -> str:
        """Generate content using Gemini.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text content
        """
        if not self.model:
            self.logger.error("Gemini model not initialized")
            return ""
        
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {e}")
            return ""
    
    def analyze_competitor_website(self, competitor_url: str, brand_context: str) -> Dict:
        """Analyze competitor website for strategic insights.
        
        Args:
            competitor_url: Competitor website URL
            brand_context: Brand context for comparison
            
        Returns:
            Dictionary with competitor analysis
        """
        prompt = f"""Analyze this competitor website and provide strategic insights for Google Ads campaigns.

Competitor URL: {competitor_url}
Our Brand Context: {brand_context}

Please provide a comprehensive analysis in JSON format with:

1. **value_propositions**: Key value propositions and messaging
2. **target_audience**: Who they're targeting (demographics, personas)
3. **pricing_strategy**: Pricing approach and positioning (premium/budget/mid-market)
4. **competitive_advantages**: What they emphasize as strengths
5. **market_positioning**: How they position themselves in the market
6. **keyword_opportunities**: Suggested keywords we could target against them
7. **messaging_gaps**: Areas where we could differentiate
8. **campaign_strategies**: Specific Google Ads strategies to compete

Return ONLY valid JSON without any markdown formatting."""

        response = self.generate_content(prompt, max_tokens=4096, temperature=0.2)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse competitor analysis JSON")
            return {
                "value_propositions": ["Analysis failed"],
                "target_audience": ["Unknown"],
                "pricing_strategy": "Unknown",
                "competitive_advantages": ["Analysis failed"],
                "market_positioning": "Unknown",
                "keyword_opportunities": ["competitor brand name"],
                "messaging_gaps": ["Analysis failed"],
                "campaign_strategies": ["Basic competitor targeting"]
            }
    
    def optimize_budget_allocation(self, keywords: List[Dict], total_budget: float, 
                                  campaign_types: List[str], conversion_rate: float = 0.02) -> Dict:
        """Optimize budget allocation across campaign types.
        
        Args:
            keywords: List of keyword data
            total_budget: Total monthly budget
            campaign_types: List of campaign types (Search, Shopping, PMax)
            conversion_rate: Expected conversion rate
            
        Returns:
            Budget optimization recommendations
        """
        # Prepare keyword summary for analysis
        keyword_summary = []
        for kw in keywords[:50]:  # Limit for prompt size
            keyword_summary.append({
                'keyword': kw.get('keyword', ''),
                'intent': kw.get('intent', 'commercial'),
                'volume': kw.get('volume', 0),
                'cpc_high': kw.get('cpc_high', 0),
                'competition': kw.get('competition', 0.5)
            })
        
        prompt = f"""You are a Google Ads budget optimization expert. Analyze these keywords and provide budget allocation recommendations.

INPUTS:
- Total Monthly Budget: ${total_budget:,.0f}
- Campaign Types: {', '.join(campaign_types)}
- Expected Conversion Rate: {conversion_rate*100:.1f}%
- Keywords Data: {json.dumps(keyword_summary[:20], indent=2)}

ANALYSIS REQUIREMENTS:
1. Analyze keyword intent distribution and volume patterns
2. Consider CPC and competition levels
3. Factor in typical performance by campaign type:
   - Search: High intent, precise targeting
   - Shopping: Product-focused, visual appeal
   - PMax: Broad reach, automation benefits

Provide recommendations in JSON format:
{{
  "budget_allocation": {{
    "search_campaign": {{
      "amount": 0000,
      "percentage": 00,
      "reasoning": "explanation"
    }},
    "shopping_campaign": {{
      "amount": 0000,
      "percentage": 00,
      "reasoning": "explanation"
    }},
    "pmax_campaign": {{
      "amount": 0000,
      "percentage": 00,
      "reasoning": "explanation"
    }}
  }},
  "bid_strategies": {{
    "search": "recommended bid strategy",
    "shopping": "recommended bid strategy", 
    "pmax": "recommended bid strategy"
  }},
  "performance_forecast": {{
    "estimated_clicks": 0000,
    "estimated_conversions": 000,
    "estimated_roas": 0.0
  }},
  "optimization_tips": [
    "tip 1",
    "tip 2",
    "tip 3"
  ]
}}

Return ONLY valid JSON without markdown formatting."""

        response = self.generate_content(prompt, max_tokens=2048, temperature=0.1)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse budget optimization JSON")
            # Return fallback allocation
            search_amount = total_budget * 0.5
            shopping_amount = total_budget * 0.3
            pmax_amount = total_budget * 0.2
            
            return {
                "budget_allocation": {
                    "search_campaign": {
                        "amount": search_amount,
                        "percentage": 50,
                        "reasoning": "High-intent keywords with precise targeting"
                    },
                    "shopping_campaign": {
                        "amount": shopping_amount,
                        "percentage": 30,
                        "reasoning": "Product showcase with visual appeal"
                    },
                    "pmax_campaign": {
                        "amount": pmax_amount,
                        "percentage": 20,
                        "reasoning": "Broad reach across Google properties"
                    }
                },
                "bid_strategies": {
                    "search": "Target CPA",
                    "shopping": "Target ROAS",
                    "pmax": "Maximize Conversions"
                },
                "performance_forecast": {
                    "estimated_clicks": int(total_budget / 2.5),
                    "estimated_conversions": int(total_budget / 2.5 * conversion_rate),
                    "estimated_roas": 4.0
                },
                "optimization_tips": [
                    "Focus budget on high-intent transactional keywords",
                    "Test different bid strategies after initial data collection",
                    "Monitor performance weekly and adjust budgets based on ROAS"
                ]
            }
    
    def generate_campaign_structure(self, keywords: List[Dict], business_context: str, 
                                   competitors: List[str] = None) -> Dict:
        """Generate complete campaign structure with ad groups.
        
        Args:
            keywords: List of keyword data
            business_context: Business description and context
            competitors: List of competitor URLs
            
        Returns:
            Complete campaign structure recommendations
        """
        # Prepare data for analysis
        keyword_intents = {}
        for kw in keywords:
            intent = kw.get('intent', 'commercial')
            if intent not in keyword_intents:
                keyword_intents[intent] = []
            keyword_intents[intent].append(kw.get('keyword', ''))
        
        competitor_info = f"\nCompetitors: {', '.join(competitors)}" if competitors else ""
        
        prompt = f"""You are a Google Ads campaign architect. Create a comprehensive campaign structure for this business.

BUSINESS CONTEXT: {business_context}{competitor_info}

KEYWORD ANALYSIS:
{json.dumps(keyword_intents, indent=2)}

Create a strategic campaign structure in JSON format:

{{
  "search_campaigns": [
    {{
      "campaign_name": "Brand Campaign",
      "campaign_type": "Search",
      "targeting_strategy": "Exact brand terms",
      "ad_groups": [
        {{
          "ad_group_name": "Brand Core Terms",
          "keywords": ["brand keyword 1", "brand keyword 2"],
          "match_types": ["exact", "phrase"],
          "suggested_cpc_range": "low-high",
          "landing_page_theme": "homepage"
        }}
      ]
    }}
  ],
  "shopping_campaigns": [
    {{
      "campaign_name": "Shopping - All Products",
      "campaign_type": "Shopping",
      "product_groups": [
        {{
          "group_name": "Category 1",
          "products": ["product type"],
          "bid_adjustment": "percentage"
        }}
      ]
    }}
  ],
  "pmax_campaigns": [
    {{
      "campaign_name": "Performance Max - Growth",
      "asset_groups": [
        {{
          "theme_name": "Theme 1",
          "target_audience": "audience description",
          "keywords": ["keyword 1", "keyword 2"],
          "messaging_focus": "value proposition"
        }}
      ]
    }}
  ],
  "negative_keywords": [
    "negative keyword 1",
    "negative keyword 2"
  ],
  "campaign_priorities": {{
    "search": 1,
    "shopping": 2,
    "pmax": 3
  }}
}}

Include strategic rationale for each campaign type. Return ONLY valid JSON."""

        response = self.generate_content(prompt, max_tokens=4096, temperature=0.2)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse campaign structure JSON")
            return {
                "search_campaigns": [
                    {
                        "campaign_name": "Search - Brand Terms",
                        "campaign_type": "Search",
                        "targeting_strategy": "Brand protection and high-intent terms",
                        "ad_groups": [
                            {
                                "ad_group_name": "Brand Core",
                                "keywords": [kw.get('keyword', '') for kw in keywords[:10]],
                                "match_types": ["exact", "phrase"],
                                "suggested_cpc_range": "$1.00-$3.00",
                                "landing_page_theme": "homepage"
                            }
                        ]
                    }
                ],
                "shopping_campaigns": [
                    {
                        "campaign_name": "Shopping - All Products",
                        "campaign_type": "Shopping",
                        "product_groups": [
                            {
                                "group_name": "All Products",
                                "products": ["all"],
                                "bid_adjustment": "0%"
                            }
                        ]
                    }
                ],
                "pmax_campaigns": [
                    {
                        "campaign_name": "Performance Max - Growth",
                        "asset_groups": [
                            {
                                "theme_name": "Primary Products",
                                "target_audience": "Interested customers",
                                "keywords": [kw.get('keyword', '') for kw in keywords[:15]],
                                "messaging_focus": "Quality and value"
                            }
                        ]
                    }
                ],
                "negative_keywords": [
                    "free",
                    "cheap",
                    "diy",
                    "job",
                    "career"
                ],
                "campaign_priorities": {
                    "search": 1,
                    "shopping": 2,
                    "pmax": 3
                }
            }
    
    def analyze_market_trends(self, industry: str, keywords: List[str], 
                             business_context: str) -> Dict:
        """Analyze market trends and opportunities.
        
        Args:
            industry: Industry/market category
            keywords: List of primary keywords
            business_context: Business description
            
        Returns:
            Market analysis and trend insights
        """
        prompt = f"""You are a market research expert specializing in Google Ads strategy. Analyze market trends and opportunities for this business.

INDUSTRY: {industry}
BUSINESS: {business_context}
PRIMARY KEYWORDS: {', '.join(keywords[:20])}

Provide comprehensive market analysis in JSON format:

{{
  "market_size": {{
    "size_category": "small/medium/large",
    "growth_trend": "growing/stable/declining",
    "competition_level": "low/medium/high"
  }},
  "seasonal_trends": [
    {{
      "season": "Q1/Q2/Q3/Q4 or specific months",
      "demand_level": "high/medium/low",
      "keyword_opportunities": ["seasonal keyword 1"],
      "campaign_strategy": "strategy description"
    }}
  ],
  "audience_insights": {{
    "primary_demographics": ["demographic 1", "demographic 2"],
    "search_behaviors": ["behavior 1", "behavior 2"],
    "intent_patterns": ["pattern 1", "pattern 2"]
  }},
  "competitive_landscape": {{
    "major_players": ["competitor 1", "competitor 2"],
    "market_gaps": ["gap 1", "gap 2"],
    "differentiation_opportunities": ["opportunity 1"]
  }},
  "keyword_opportunities": {{
    "trending_keywords": ["trending 1", "trending 2"],
    "long_tail_opportunities": ["long tail 1"],
    "local_opportunities": ["local keyword 1"],
    "voice_search_opportunities": ["voice query 1"]
  }},
  "campaign_recommendations": [
    {{
      "campaign_type": "Search/Shopping/PMax",
      "opportunity": "opportunity description",
      "expected_performance": "performance expectation",
      "implementation_priority": "high/medium/low"
    }}
  ]
}}

Focus on actionable insights for Google Ads optimization. Return ONLY valid JSON."""

        response = self.generate_content(prompt, max_tokens=3072, temperature=0.2)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse market analysis JSON")
            return {
                "market_size": {
                    "size_category": "medium",
                    "growth_trend": "stable",
                    "competition_level": "medium"
                },
                "seasonal_trends": [
                    {
                        "season": "Q4",
                        "demand_level": "high",
                        "keyword_opportunities": ["holiday shopping", "year-end"],
                        "campaign_strategy": "Increase budgets for holiday season"
                    }
                ],
                "audience_insights": {
                    "primary_demographics": ["Business professionals", "Tech-savvy users"],
                    "search_behaviors": ["Research before purchase", "Mobile-first"],
                    "intent_patterns": ["Solution-seeking", "Comparison shopping"]
                },
                "competitive_landscape": {
                    "major_players": ["Unknown competitors"],
                    "market_gaps": ["Personalized service", "Better pricing"],
                    "differentiation_opportunities": ["Customer service", "Product quality"]
                },
                "keyword_opportunities": {
                    "trending_keywords": ["ai-powered", "automated"],
                    "long_tail_opportunities": [f"best {keywords[0]} for small business"],
                    "local_opportunities": [f"{keywords[0]} near me"],
                    "voice_search_opportunities": [f"what is the best {keywords[0]}"]
                },
                "campaign_recommendations": [
                    {
                        "campaign_type": "Search",
                        "opportunity": "Target high-intent keywords",
                        "expected_performance": "High CTR and conversion rate",
                        "implementation_priority": "high"
                    }
                ]
            }