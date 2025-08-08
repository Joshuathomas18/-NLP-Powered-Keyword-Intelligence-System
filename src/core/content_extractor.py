"""Content extraction from HTML pages."""

import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from readability import Document

from ..utils.logging import get_logger


class ContentExtractor:
    """Extracts clean, structured content from HTML pages."""
    
    def __init__(self):
        self.logger = get_logger('ContentExtractor')
    
    def extract_content(self, page_data: Dict) -> Dict:
        """Extract structured content from page data.
        
        Args:
            page_data: Page data from SiteFetcher
            
        Returns:
            Dict with extracted content: title, meta, headings, body_text, etc.
        """
        if not page_data or page_data.get('status_code') != 200:
            return {}
        
        html_content = page_data.get('content', '')
        url = page_data.get('url', '')
        
        if not html_content:
            return {}
        
        try:
            # Use readability to extract main content
            doc = Document(html_content)
            main_content = doc.summary()
            
            # Parse with BeautifulSoup for structured extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract various content elements
            extracted = {
                'url': url,
                'title': self._extract_title(soup),
                'meta_description': self._extract_meta_description(soup),
                'meta_keywords': self._extract_meta_keywords(soup),
                'headings': self._extract_headings(soup),
                'body_text': self._extract_clean_text(main_content),
                'links': self._extract_internal_links(soup, url),
                'images': self._extract_images(soup),
                'content_length': len(main_content),
                'word_count': len(self._extract_clean_text(main_content).split())
            }
            
            self.logger.info(f"Extracted content from {url}: {extracted['word_count']} words")
            return extracted
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return {}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ""
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        # Try og:description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        return ""
    
    def _extract_meta_keywords(self, soup: BeautifulSoup) -> str:
        """Extract meta keywords."""
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            return meta_keywords['content'].strip()
        return ""
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings by level."""
        headings = {
            'h1': [],
            'h2': [],
            'h3': [],
            'h4': [],
            'h5': [],
            'h6': []
        }
        
        for level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            tags = soup.find_all(level)
            headings[level] = [tag.get_text().strip() for tag in tags if tag.get_text().strip()]
        
        return headings
    
    def _extract_clean_text(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        if not html_content:
            return ""
        
        # Parse with BeautifulSoup to remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract internal links."""
        from urllib.parse import urljoin, urlparse
        
        base_domain = urlparse(base_url).netloc
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if internal link
            if urlparse(full_url).netloc == base_domain:
                links.append({
                    'url': full_url,
                    'text': link.get_text().strip(),
                    'title': link.get('title', '')
                })
        
        return links[:50]  # Limit number of links
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract image information."""
        images = []
        
        for img in soup.find_all('img', src=True):
            images.append({
                'src': img['src'],
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        return images[:20]  # Limit number of images
    
    def extract_keywords_from_content(self, content: Dict) -> List[str]:
        """Extract potential keywords from content using simple heuristics.
        
        This is a basic implementation - in production would use more sophisticated NLP.
        """
        keywords = set()
        
        # Extract from title (high importance)
        title = content.get('title', '')
        if title:
            keywords.update(self._extract_phrases(title, min_length=2))
        
        # Extract from meta description
        meta_desc = content.get('meta_description', '')
        if meta_desc:
            keywords.update(self._extract_phrases(meta_desc, min_length=2))
        
        # Extract from headings
        headings = content.get('headings', {})
        for level, heading_list in headings.items():
            for heading in heading_list:
                if heading:
                    keywords.update(self._extract_phrases(heading, min_length=2))
        
        # Extract from body text (sample to avoid too many keywords)
        body_text = content.get('body_text', '')
        if body_text:
            # Take first 1000 characters to avoid processing huge texts
            sample_text = body_text[:1000]
            keywords.update(self._extract_phrases(sample_text, min_length=2, max_phrases=20))
        
        return list(keywords)
    
    def _extract_phrases(self, text: str, min_length: int = 2, max_phrases: int = 50) -> List[str]:
        """Extract meaningful phrases from text."""
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        phrases = []
        
        # Extract 2-4 word phrases
        for length in range(min_length, min(5, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                
                # Filter out common stop phrases and short words
                if self._is_meaningful_phrase(phrase):
                    phrases.append(phrase)
        
        # Remove duplicates and limit
        return list(set(phrases))[:max_phrases]
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if phrase is meaningful for keyword extraction."""
        # Skip very short phrases
        if len(phrase) < 3:
            return False
        
        # Skip phrases with only common words
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now', 'get', 'got', 'has', 'have',
            'had', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall'
        }
        
        words = phrase.split()
        meaningful_words = [w for w in words if w not in common_words and len(w) > 2]
        
        # Require at least one meaningful word
        return len(meaningful_words) > 0