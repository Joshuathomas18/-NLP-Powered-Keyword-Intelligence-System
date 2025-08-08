"""Keyword embedding generation and storage using Sentence-BERT."""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

from ..utils.logging import get_logger


class EmbeddingStore:
    """Generates and manages keyword embeddings using Sentence-BERT."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./cache"):
        """Initialize embedding store.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings
        """
        self.logger = get_logger('EmbeddingStore')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        try:
            self.logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            self.model = None
        
        self.embeddings_cache = {}
    
    def generate_embeddings(self, keywords: List[Dict], site_content: List[Dict] = None) -> Dict:
        """Generate embeddings for keywords and site content.
        
        Args:
            keywords: List of keyword dictionaries
            site_content: Optional site content for context
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if not self.model:
            self.logger.warning("No embedding model available")
            return {'keyword_embeddings': {}, 'content_embeddings': {}}
        
        self.logger.info(f"Generating embeddings for {len(keywords)} keywords")
        
        # Prepare keyword texts
        keyword_texts = [kw['keyword'] for kw in keywords]
        
        # Check cache first
        cache_file = self.cache_dir / f"embeddings_{hash(str(sorted(keyword_texts)))}.pkl"
        
        if cache_file.exists():
            self.logger.info("Loading embeddings from cache")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Generate keyword embeddings
        keyword_embeddings = {}
        try:
            embeddings = self.model.encode(keyword_texts, show_progress_bar=True)
            
            for i, kw in enumerate(keywords):
                keyword_embeddings[kw['keyword']] = {
                    'embedding': embeddings[i],
                    'metadata': kw
                }
        
        except Exception as e:
            self.logger.error(f"Failed to generate keyword embeddings: {e}")
            return {'keyword_embeddings': {}, 'content_embeddings': {}}
        
        # Generate site content embeddings if provided
        content_embeddings = {}
        if site_content:
            self.logger.info(f"Generating embeddings for {len(site_content)} content pieces")
            
            content_texts = []
            for content in site_content:
                # Combine title, meta description, and body text
                text_parts = [
                    content.get('title', ''),
                    content.get('meta_description', ''),
                    content.get('body_text', '')[:500]  # Limit body text
                ]
                combined_text = ' '.join(filter(None, text_parts))
                content_texts.append(combined_text)
            
            try:
                content_embeds = self.model.encode(content_texts, show_progress_bar=True)
                
                for i, content in enumerate(site_content):
                    content_embeddings[content['url']] = {
                        'embedding': content_embeds[i],
                        'metadata': content
                    }
            
            except Exception as e:
                self.logger.error(f"Failed to generate content embeddings: {e}")
        
        # Prepare result
        result = {
            'keyword_embeddings': keyword_embeddings,
            'content_embeddings': content_embeddings,
            'model_name': self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 384
        }
        
        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.logger.info(f"Cached embeddings to {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to cache embeddings: {e}")
        
        return result
    
    def calculate_similarities(self, keyword_embeddings: Dict, content_embeddings: Dict = None) -> Dict:
        """Calculate similarity matrices between keywords and content.
        
        Args:
            keyword_embeddings: Dictionary of keyword embeddings
            content_embeddings: Optional dictionary of content embeddings
            
        Returns:
            Dictionary with similarity matrices and rankings
        """
        self.logger.info("Calculating embedding similarities")
        
        if not keyword_embeddings:
            return {}
        
        # Extract embeddings as arrays
        keywords = list(keyword_embeddings.keys())
        keyword_vectors = np.array([keyword_embeddings[kw]['embedding'] for kw in keywords])
        
        # Calculate keyword-to-keyword similarity matrix
        keyword_similarity = np.dot(keyword_vectors, keyword_vectors.T)
        
        # Normalize to get cosine similarity
        norms = np.linalg.norm(keyword_vectors, axis=1)
        keyword_similarity = keyword_similarity / np.outer(norms, norms)
        
        result = {
            'keyword_similarity_matrix': keyword_similarity,
            'keyword_list': keywords
        }
        
        # Calculate keyword-to-content similarities if content provided
        if content_embeddings:
            content_urls = list(content_embeddings.keys())
            content_vectors = np.array([content_embeddings[url]['embedding'] for url in content_urls])
            
            # Calculate similarities
            keyword_content_similarity = np.dot(keyword_vectors, content_vectors.T)
            
            # Normalize
            content_norms = np.linalg.norm(content_vectors, axis=1)
            keyword_content_similarity = keyword_content_similarity / np.outer(norms, content_norms)
            
            result.update({
                'keyword_content_similarity': keyword_content_similarity,
                'content_list': content_urls
            })
            
            # Find most relevant content for each keyword
            keyword_relevance = {}
            for i, keyword in enumerate(keywords):
                similarities = keyword_content_similarity[i]
                best_content_idx = np.argmax(similarities)
                
                keyword_relevance[keyword] = {
                    'most_relevant_content': content_urls[best_content_idx],
                    'relevance_score': similarities[best_content_idx],
                    'content_similarities': dict(zip(content_urls, similarities))
                }
            
            result['keyword_relevance'] = keyword_relevance
        
        return result
    
    def find_related_keywords(self, target_keyword: str, keyword_embeddings: Dict, 
                             top_k: int = 10, min_similarity: float = 0.5) -> List[Dict]:
        """Find keywords most similar to a target keyword.
        
        Args:
            target_keyword: The keyword to find relations for
            keyword_embeddings: Dictionary of all keyword embeddings
            top_k: Number of top similar keywords to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of related keywords with similarity scores
        """
        if target_keyword not in keyword_embeddings:
            return []
        
        target_embedding = keyword_embeddings[target_keyword]['embedding']
        related_keywords = []
        
        for keyword, data in keyword_embeddings.items():
            if keyword == target_keyword:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, data['embedding'])
            similarity = similarity / (np.linalg.norm(target_embedding) * np.linalg.norm(data['embedding']))
            
            if similarity >= min_similarity:
                related_keywords.append({
                    'keyword': keyword,
                    'similarity': float(similarity),
                    'metadata': data['metadata']
                })
        
        # Sort by similarity and return top_k
        related_keywords.sort(key=lambda x: x['similarity'], reverse=True)
        return related_keywords[:top_k]
    
    def get_keyword_context_score(self, keyword: str, keyword_embeddings: Dict, 
                                  content_embeddings: Dict) -> float:
        """Calculate how well a keyword fits the site content context.
        
        Args:
            keyword: Keyword to score
            keyword_embeddings: Dictionary of keyword embeddings
            content_embeddings: Dictionary of content embeddings
            
        Returns:
            Context relevance score (0-1)
        """
        if not keyword_embeddings.get(keyword) or not content_embeddings:
            return 0.0
        
        keyword_embedding = keyword_embeddings[keyword]['embedding']
        
        # Calculate similarity to all content pieces
        similarities = []
        for content_data in content_embeddings.values():
            content_embedding = content_data['embedding']
            
            similarity = np.dot(keyword_embedding, content_embedding)
            similarity = similarity / (np.linalg.norm(keyword_embedding) * np.linalg.norm(content_embedding))
            similarities.append(similarity)
        
        # Return average similarity as context score
        return float(np.mean(similarities)) if similarities else 0.0