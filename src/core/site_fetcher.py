"""Site fetching and content retrieval."""

import requests
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from ..utils.cache import CacheManager
from ..utils.logging import get_logger


class SiteFetcher:
    """Fetches website content while respecting robots.txt and rate limits."""
    
    def __init__(self, cache_manager: CacheManager, delay_between_requests: float = 1.0):
        """Initialize site fetcher.
        
        Args:
            cache_manager: Cache manager for storing responses
            delay_between_requests: Delay between requests to same domain
        """
        self.cache = cache_manager
        self.delay = delay_between_requests
        self.logger = get_logger('SiteFetcher')
        self.last_request_time = {}
        
        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KeywordResearch-Bot/1.0 (+research purposes)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        # Disable SSL verification for problematic sites (for MVP)
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, '/robots.txt')
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(self.session.headers['User-Agent'], url)
        except Exception as e:
            self.logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True  # Assume allowed if can't check
    
    def _respect_rate_limit(self, url: str) -> None:
        """Ensure rate limiting between requests to same domain."""
        domain = urlparse(url).netloc
        
        if domain in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[domain]
            if time_since_last < self.delay:
                sleep_time = self.delay - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def fetch_page(self, url: str, force_refresh: bool = False) -> Optional[Dict]:
        """Fetch single page content.
        
        Args:
            url: URL to fetch
            force_refresh: Bypass cache and fetch fresh content
            
        Returns:
            Dict with 'url', 'status_code', 'content', 'headers' or None if failed
        """
        self.logger.info(f"Fetching page: {url}")
        
        # Check cache first unless forcing refresh
        if not force_refresh:
            cached_content = self.cache.get(url)
            if cached_content:
                self.logger.info(f"Using cached content for {url}")
                return cached_content
        
        # Check robots.txt
        if not self._can_fetch(url):
            self.logger.warning(f"Robots.txt disallows fetching {url}")
            return None
        
        # Rate limiting
        self._respect_rate_limit(url)
        
        try:
            response = self.session.get(url, timeout=10)
            
            content_data = {
                'url': url,
                'status_code': response.status_code,
                'content': response.text if response.status_code == 200 else '',
                'headers': dict(response.headers),
                'fetched_at': time.time()
            }
            
            # Cache successful responses
            if response.status_code == 200:
                self.cache.set(url, content_data, ttl=3600)  # Cache for 1 hour
            
            self.logger.info(f"Fetched {url}: {response.status_code}")
            return content_data
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def fetch_multiple_pages(self, urls: List[str], max_pages: int = 10) -> List[Dict]:
        """Fetch multiple pages with rate limiting.
        
        Args:
            urls: List of URLs to fetch
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of successfully fetched page data
        """
        results = []
        
        for i, url in enumerate(urls[:max_pages]):
            self.logger.info(f"Fetching page {i+1}/{min(len(urls), max_pages)}: {url}")
            
            page_data = self.fetch_page(url)
            if page_data and page_data['status_code'] == 200:
                results.append(page_data)
            
            # Rate limiting between pages
            if i < len(urls) - 1:  # Don't sleep after last request
                time.sleep(self.delay)
        
        self.logger.info(f"Successfully fetched {len(results)} pages")
        return results
    
    def discover_site_pages(self, base_url: str) -> List[str]:
        """Discover important pages from a website.
        
        Args:
            base_url: Base website URL
            
        Returns:
            List of important page URLs to fetch
        """
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Standard pages to check
        candidate_paths = [
            '/',
            '/about',
            '/about-us',
            '/products',
            '/services',
            '/solutions',
            '/pricing',
            '/features',
            '/docs',
            '/documentation',
            '/blog',
            '/news',
            '/company',
            '/contact'
        ]
        
        # Build full URLs
        candidate_urls = [urljoin(base_domain, path) for path in candidate_paths]
        
        # Try to fetch sitemap for more URLs
        sitemap_urls = self._try_fetch_sitemap(base_domain)
        if sitemap_urls:
            candidate_urls.extend(sitemap_urls[:20])  # Limit sitemap URLs
        
        return list(set(candidate_urls))  # Remove duplicates
    
    def _try_fetch_sitemap(self, base_url: str) -> List[str]:
        """Try to fetch URLs from sitemap.xml."""
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')  # Look for sitemap reference
        ]
        
        urls = []
        for sitemap_url in sitemap_urls:
            try:
                response = self.session.get(sitemap_url, timeout=5)
                if response.status_code == 200:
                    # Simple extraction - in production would use proper XML parsing
                    content = response.text
                    if 'sitemap.xml' in sitemap_url or '<urlset' in content:
                        # Extract URLs from sitemap (simplified)
                        import re
                        url_pattern = r'<loc>(.*?)</loc>'
                        found_urls = re.findall(url_pattern, content)
                        urls.extend(found_urls[:10])  # Limit URLs from sitemap
                        break
            except:
                continue
        
        return urls