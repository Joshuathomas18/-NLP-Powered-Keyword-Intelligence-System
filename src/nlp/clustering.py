"""Keyword clustering using embeddings and various clustering algorithms."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import re

from ..utils.logging import get_logger


class KeywordClusterer:
    """Clusters keywords based on semantic similarity using embeddings."""
    
    def __init__(self):
        self.logger = get_logger('KeywordClusterer')
    
    def cluster_keywords(self, keyword_embeddings: Dict, 
                        method: str = "kmeans", 
                        max_clusters: int = 50,
                        min_cluster_size: int = 2) -> Dict:
        """Cluster keywords using specified method.
        
        Args:
            keyword_embeddings: Dictionary of keyword embeddings
            method: Clustering method ("kmeans", "dbscan", "auto")
            max_clusters: Maximum number of clusters for KMeans
            min_cluster_size: Minimum size for clusters
            
        Returns:
            Dictionary with clustering results
        """
        if not keyword_embeddings:
            return {}
        
        self.logger.info(f"Clustering {len(keyword_embeddings)} keywords using {method}")
        
        # Prepare data
        keywords = list(keyword_embeddings.keys())
        embeddings = np.array([keyword_embeddings[kw]['embedding'] for kw in keywords])
        
        # Choose clustering method
        if method == "auto":
            clusters = self._auto_cluster(embeddings, keywords, keyword_embeddings, max_clusters)
        elif method == "dbscan":
            clusters = self._dbscan_cluster(embeddings, keywords, keyword_embeddings, min_cluster_size)
        else:  # default to kmeans
            clusters = self._kmeans_cluster(embeddings, keywords, keyword_embeddings, max_clusters)
        
        # Post-process clusters
        processed_clusters = self._process_clusters(clusters, keyword_embeddings, min_cluster_size)
        
        # Generate cluster summaries
        cluster_summaries = self._generate_cluster_summaries(processed_clusters)
        
        return {
            'clusters': processed_clusters,
            'cluster_summaries': cluster_summaries,
            'total_clusters': len(processed_clusters),
            'method_used': method
        }
    
    def _auto_cluster(self, embeddings: np.ndarray, keywords: List[str], 
                     keyword_embeddings: Dict, max_clusters: int) -> Dict:
        """Automatically determine best clustering method and parameters."""
        n_keywords = len(keywords)
        
        if n_keywords < 10:
            # Too few keywords, use simple grouping
            return self._simple_grouping(keywords, keyword_embeddings)
        
        # Try different KMeans cluster counts
        best_score = -1
        best_k = 2
        best_clusters = {}
        
        for k in range(2, min(max_clusters, n_keywords // 2)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
                    # Store cluster assignments
                    clusters = defaultdict(list)
                    for i, label in enumerate(cluster_labels):
                        clusters[f"cluster_{label}"].append(keywords[i])
                    best_clusters = dict(clusters)
            
            except Exception as e:
                self.logger.warning(f"KMeans with k={k} failed: {e}")
                continue
        
        self.logger.info(f"Auto-clustering selected k={best_k} with silhouette score={best_score:.3f}")
        return best_clusters
    
    def _kmeans_cluster(self, embeddings: np.ndarray, keywords: List[str], 
                       keyword_embeddings: Dict, max_clusters: int) -> Dict:
        """Cluster using KMeans with optimal k selection."""
        n_keywords = len(keywords)
        optimal_k = min(max_clusters, max(2, n_keywords // 5))  # Heuristic for k
        
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group keywords by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[f"cluster_{label}"].append(keywords[i])
            
            return dict(clusters)
            
        except Exception as e:
            self.logger.error(f"KMeans clustering failed: {e}")
            return self._simple_grouping(keywords, keyword_embeddings)
    
    def _dbscan_cluster(self, embeddings: np.ndarray, keywords: List[str], 
                       keyword_embeddings: Dict, min_cluster_size: int) -> Dict:
        """Cluster using DBSCAN for density-based clustering."""
        try:
            # Normalize embeddings for better DBSCAN performance
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings)
            
            # DBSCAN parameters
            eps = 0.5  # Distance threshold
            min_samples = max(2, min_cluster_size)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(normalized_embeddings)
            
            # Group keywords by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise points
                    clusters["unclustered"].append(keywords[i])
                else:
                    clusters[f"cluster_{label}"].append(keywords[i])
            
            return dict(clusters)
            
        except Exception as e:
            self.logger.error(f"DBSCAN clustering failed: {e}")
            return self._simple_grouping(keywords, keyword_embeddings)
    
    def _simple_grouping(self, keywords: List[str], keyword_embeddings: Dict) -> Dict:
        """Fallback simple grouping based on first word and patterns."""
        groups = defaultdict(list)
        
        for keyword in keywords:
            # Group by first word
            first_word = keyword.split()[0] if keyword.split() else keyword
            groups[f"group_{first_word}"].append(keyword)
        
        return dict(groups)
    
    def _process_clusters(self, clusters: Dict, keyword_embeddings: Dict, 
                         min_cluster_size: int) -> List[Dict]:
        """Process and clean up clusters."""
        processed_clusters = []
        
        # Create a copy to avoid dictionary modification during iteration
        clusters_copy = dict(clusters)
        
        for cluster_id, cluster_keywords in clusters_copy.items():
            if len(cluster_keywords) < min_cluster_size and cluster_id != "unclustered":
                # Move small clusters to unclustered
                if "unclustered" not in clusters:
                    clusters["unclustered"] = []
                clusters["unclustered"].extend(cluster_keywords)
                continue
            
            # Calculate cluster statistics
            cluster_data = []
            total_volume = 0
            total_cpc = 0
            sources = Counter()
            
            for keyword in cluster_keywords:
                kw_data = keyword_embeddings[keyword]['metadata']
                cluster_data.append(kw_data)
                
                total_volume += kw_data.get('volume', 0)
                total_cpc += kw_data.get('cpc_high', 0)
                sources[kw_data.get('source', 'unknown')] += 1
            
            # Generate cluster info
            cluster_info = {
                'cluster_id': cluster_id,
                'keywords': cluster_data,
                'keyword_count': len(cluster_keywords),
                'total_volume': total_volume,
                'avg_cpc': total_cpc / len(cluster_keywords) if cluster_keywords else 0,
                'primary_source': sources.most_common(1)[0][0] if sources else 'unknown',
                'centroid_keyword': self._find_centroid_keyword(cluster_keywords, keyword_embeddings)
            }
            
            processed_clusters.append(cluster_info)
        
        # Sort clusters by total volume (most valuable first)
        processed_clusters.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return processed_clusters
    
    def _find_centroid_keyword(self, cluster_keywords: List[str], 
                              keyword_embeddings: Dict) -> str:
        """Find the keyword closest to cluster centroid."""
        if not cluster_keywords:
            return ""
        
        if len(cluster_keywords) == 1:
            return cluster_keywords[0]
        
        # Calculate cluster centroid
        embeddings = [keyword_embeddings[kw]['embedding'] for kw in cluster_keywords]
        centroid = np.mean(embeddings, axis=0)
        
        # Find closest keyword to centroid
        best_keyword = cluster_keywords[0]
        best_distance = float('inf')
        
        for keyword in cluster_keywords:
            embedding = keyword_embeddings[keyword]['embedding']
            distance = np.linalg.norm(embedding - centroid)
            
            if distance < best_distance:
                best_distance = distance
                best_keyword = keyword
        
        return best_keyword
    
    def _generate_cluster_summaries(self, clusters: List[Dict]) -> List[Dict]:
        """Generate human-readable summaries for clusters."""
        summaries = []
        
        for cluster in clusters:
            keywords = [kw['keyword'] for kw in cluster['keywords']]
            
            # Generate cluster name based on common patterns
            cluster_name = self._generate_cluster_name(keywords)
            
            # Identify intent pattern
            intent_pattern = self._identify_intent_pattern(keywords)
            
            # Find common themes
            common_themes = self._extract_common_themes(keywords)
            
            summary = {
                'cluster_id': cluster['cluster_id'],
                'suggested_name': cluster_name,
                'intent_pattern': intent_pattern,
                'common_themes': common_themes,
                'keyword_count': cluster['keyword_count'],
                'total_volume': cluster['total_volume'],
                'centroid_keyword': cluster['centroid_keyword'],
                'sample_keywords': keywords[:5]  # Top 5 as samples
            }
            
            summaries.append(summary)
        
        return summaries
    
    def _generate_cluster_name(self, keywords: List[str]) -> str:
        """Generate a descriptive name for a cluster of keywords."""
        if not keywords:
            return "Empty Cluster"
        
        # Extract common words
        all_words = []
        for keyword in keywords:
            words = keyword.lower().split()
            all_words.extend(words)
        
        # Find most common words (excluding very common stop words)
        word_counts = Counter(all_words)
        common_stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        meaningful_words = [
            word for word, count in word_counts.most_common(10)
            if word not in common_stop_words and len(word) > 2
        ]
        
        if meaningful_words:
            # Create name from top 2-3 common words
            name_words = meaningful_words[:3]
            cluster_name = ' '.join(name_words).title()
            return f"{cluster_name} Keywords"
        
        # Fallback: use first keyword
        return f"{keywords[0].title()} Group"
    
    def _identify_intent_pattern(self, keywords: List[str]) -> str:
        """Identify the primary search intent pattern in a cluster."""
        intent_signals = {
            'transactional': ['buy', 'purchase', 'order', 'shop', 'price', 'cost', 'cheap', 'discount', 'sale'],
            'informational': ['what', 'how', 'why', 'guide', 'tutorial', 'learn', 'tips', 'help'],
            'navigational': ['login', 'sign in', 'website', 'official', 'home', 'contact'],
            'commercial': ['best', 'top', 'compare', 'review', 'vs', 'alternative', 'solution', 'service']
        }
        
        intent_scores = {intent: 0 for intent in intent_signals}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for intent, signals in intent_signals.items():
                for signal in signals:
                    if signal in keyword_lower:
                        intent_scores[intent] += 1
        
        # Return dominant intent
        if max(intent_scores.values()) == 0:
            return 'general'
        
        return max(intent_scores, key=intent_scores.get)
    
    def _extract_common_themes(self, keywords: List[str]) -> List[str]:
        """Extract common themes/topics from keywords."""
        # Extract important terms (nouns and adjectives)
        important_terms = []
        
        for keyword in keywords:
            # Simple extraction of meaningful terms
            words = keyword.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    important_terms.append(word)
        
        # Find most common themes
        term_counts = Counter(important_terms)
        common_themes = [term for term, count in term_counts.most_common(5) if count > 1]
        
        return common_themes[:3]  # Return top 3 themes