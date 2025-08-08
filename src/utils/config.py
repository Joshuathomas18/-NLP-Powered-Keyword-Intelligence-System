"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 800


@dataclass
class EnrichersConfig:
    use_wordstream: bool = True
    use_serpapi: bool = False
    serpapi_key: str = ""


@dataclass
class FiltersConfig:
    min_search_volume: int = 100
    max_cpc: float = 20.0


@dataclass
class ExportConfig:
    csv: bool = True
    json: bool = True
    googleads_json: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-2.0-flash-exp"
    max_tokens: int = 8192


@dataclass
class Config:
    website: str
    competitors: List[str]
    locations: List[str]
    language: str
    seeds: List[str]
    max_keywords: int
    filters: FiltersConfig
    enrichers: EnrichersConfig
    llm: LLMConfig
    gemini: GeminiConfig
    export: ExportConfig
    output_dir: str
    cache_dir: str
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            website=data['website'],
            competitors=data.get('competitors', []),
            locations=data.get('locations', []),
            language=data.get('language', 'en'),
            seeds=data.get('seeds', []),
            max_keywords=data.get('max_keywords', 2000),
            filters=FiltersConfig(**data.get('filters', {})),
            enrichers=EnrichersConfig(**data.get('enrichers', {})),
            llm=LLMConfig(**data.get('llm', {})),
            gemini=GeminiConfig(**data.get('gemini', {})),
            export=ExportConfig(**data.get('export', {})),
            output_dir=data.get('output_dir', './outputs'),
            cache_dir=data.get('cache_dir', './cache'),
            logging=LoggingConfig(**data.get('logging', {}))
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.website:
            raise ValueError("Website URL is required")
        
        if self.max_keywords <= 0:
            raise ValueError("max_keywords must be positive")
        
        if self.filters.max_cpc <= 0:
            raise ValueError("max_cpc must be positive")
        
        # Create output and cache directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)