"""Caching utilities for API responses and content."""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any, Dict
from urllib.parse import urlparse


class CacheManager:
    """Simple file-based cache with TTL support."""
    
    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and parameters."""
        key_data = url
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key_data += str(sorted_params)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached data if available and not expired."""
        cache_key = self._get_cache_key(url, params)
        cache_file = self._get_cache_file(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if expired
            if time.time() > cached_data['expires_at']:
                cache_file.unlink()  # Delete expired cache
                return None
            
            return cached_data['data']
            
        except (json.JSONDecodeError, KeyError, OSError):
            # Invalid cache file, remove it
            if cache_file.exists():
                cache_file.unlink()
            return None
    
    def set(self, url: str, data: Any, params: Optional[Dict] = None, ttl: Optional[int] = None) -> None:
        """Cache data with optional TTL."""
        cache_key = self._get_cache_key(url, params)
        cache_file = self._get_cache_file(cache_key)
        
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        cached_data = {
            'url': url,
            'params': params,
            'data': data,
            'cached_at': time.time(),
            'expires_at': expires_at
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to cache data for {url}: {e}")
    
    def clear_expired(self) -> int:
        """Remove all expired cache files. Returns count of removed files."""
        removed_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                if current_time > cached_data.get('expires_at', 0):
                    cache_file.unlink()
                    removed_count += 1
                    
            except (json.JSONDecodeError, KeyError, OSError):
                # Invalid cache file, remove it
                cache_file.unlink()
                removed_count += 1
        
        return removed_count
    
    def clear_all(self) -> int:
        """Remove all cache files. Returns count of removed files."""
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            removed_count += 1
        return removed_count