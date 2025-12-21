"""
BERT-Based Semantic Clustering Feature Extraction

Simplified version using TF-IDF + K-Means for semantic clustering.
For production, replace TF-IDF with actual BERT embeddings.
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


class SemanticClusterFeatureExtractor:
    """
    Extract semantic cluster-based features for news recommendation.
    
    Implements the methodology from Section 4.4.2 of the research report:
    1. Global Clustering: Create K semantic clusters from all articles
    2. User Profiling: Calculate user's interest distribution across clusters
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        max_features: int = 5000,
        random_state: int = 42,
    ):
        """
        Initialize Semantic Cluster Feature Extractor.
        
        Args:
            n_clusters: Number of semantic clusters (K)
            max_features: Maximum TF-IDF vocabulary size
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.random_state = random_state
        
        self.vectorizer = None
        self.kmeans = None
        self.article_clusters = {}  # article_id -> cluster_id
        self.cluster_centers = None
    
    def fit(
        self,
        articles_df: pl.DataFrame,
        text_column: str = "title",
    ) -> 'SemanticClusterFeatureExtractor':
        """
        Fit global semantic clusters on all articles.
        
        Args:
            articles_df: DataFrame with article texts
            text_column: Column containing text (title, body, or concatenation)
        
        Returns:
            self
        """
        print(f"Fitting semantic clusters on {len(articles_df)} articles...")
        
        # Extract texts
        texts = articles_df[text_column].fill_null("").to_list()
        article_ids = articles_df["article_id"].to_list()
        
        # Create TF-IDF representations
        print("Creating TF-IDF representations...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',  # For multilingual, consider different stopwords
            ngram_range=(1, 2),
            min_df=2,
        )
        article_vectors = self.vectorizer.fit_transform(texts)
        
        # Normalize for better clustering
        article_vectors = normalize(article_vectors, norm='l2')
        
        # K-Means clustering
        print(f"Running K-Means clustering (K={self.n_clusters})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=1024,
            n_init=10,
        )
        cluster_labels = self.kmeans.fit_predict(article_vectors)
        
        # Store article -> cluster mapping
        self.article_clusters = dict(zip(article_ids, cluster_labels))
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Cluster statistics
        cluster_counts = np.bincount(cluster_labels, minlength=self.n_clusters)
        print(f"Cluster distribution: min={cluster_counts.min()}, "
              f"max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
        
        return self
    
    def _calculate_user_cluster_distribution(
        self,
        user_history: List[int],
    ) -> np.ndarray:
        """
        Calculate user's click distribution across clusters.
        
        Args:
            user_history: List of article_ids user has clicked
        
        Returns:
            Distribution vector of length n_clusters
        """
        if not user_history:
            # No history: uniform distribution
            return np.ones(self.n_clusters) / self.n_clusters
        
        # Count clicks per cluster
        cluster_counts = np.zeros(self.n_clusters)
        for article_id in user_history:
            cluster_id = self.article_clusters.get(article_id, -1)
            if cluster_id >= 0:
                cluster_counts[cluster_id] += 1
        
        # Normalize to distribution
        total = cluster_counts.sum()
        if total == 0:
            return np.ones(self.n_clusters) / self.n_clusters
        return cluster_counts / total
    
    def add_cluster_features_to_df(
        self,
        df: pl.DataFrame,
        history_df: pl.DataFrame = None,
        user_col: str = "user_id",
        article_col: str = "article_id",
    ) -> pl.DataFrame:
        """
        Add semantic cluster features to DataFrame.
        
        Args:
            df: Input DataFrame with user-article pairs
            history_df: Optional history DataFrame for user profiling
            user_col: Column name for user IDs
            article_col: Column name for article IDs
        
        Returns:
            DataFrame with added cluster features
        """
        print("Adding semantic cluster features to DataFrame...")
        
        # Build user history mapping
        user_histories = {}
        if history_df is not None:
            print("Building user histories from history_df...")
            for row in tqdm(history_df.iter_rows(named=True), total=len(history_df)):
                user_id = row[user_col]
                history = row.get("article_id_fixed", [])
                if history and len(history) > 0:
                    user_histories[user_id] = history
        
        # Extract user and article IDs
        user_ids = df[user_col].to_list()
        article_ids = df[article_col].to_list()
        
        # Calculate user cluster distributions
        print("Calculating user cluster distributions...")
        user_distributions = []
        for user_id in tqdm(user_ids, desc="User profiles"):
            history = user_histories.get(user_id, [])
            distribution = self._calculate_user_cluster_distribution(history)
            user_distributions.append(distribution)
        user_distributions = np.array(user_distributions)
        
        # Add user cluster distribution features
        for i in range(self.n_clusters):
            df = df.with_columns(
                pl.Series(name=f"bert_user_cluster_{i}", values=user_distributions[:, i])
            )
        
        # Add article cluster ID
        article_cluster_ids = [
            self.article_clusters.get(aid, -1) for aid in article_ids
        ]
        df = df.with_columns(
            pl.Series(name="bert_article_cluster", values=article_cluster_ids)
        )
        
        # Add user-article cluster affinity (user's interest in article's cluster)
        affinities = []
        for i, aid in enumerate(article_ids):
            cluster_id = article_cluster_ids[i]
            if cluster_id >= 0:
                affinity = user_distributions[i, cluster_id]
            else:
                affinity = 0.0
            affinities.append(affinity)
        
        df = df.with_columns(
            pl.Series(name="bert_user_article_cluster_affinity", values=affinities)
        )
        
        # Add cluster match (binary: is article in user's top cluster?)
        user_top_clusters = user_distributions.argmax(axis=1)
        matches = [
            1 if article_cluster_ids[i] == user_top_clusters[i] else 0
            for i in range(len(article_ids))
        ]
        df = df.with_columns(
            pl.Series(name="bert_user_article_cluster_match", values=matches)
        )
        
        num_features = self.n_clusters + 3
        print(f"Added {num_features} semantic cluster features")
        return df
    
    def save_model(self, save_path: Path) -> None:
        """Save the trained clustering model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save all components
        with open(save_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        with open(save_path / "kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
        
        with open(save_path / "article_clusters.pkl", "wb") as f:
            pickle.dump(self.article_clusters, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path) -> None:
        """Load a trained clustering model."""
        load_path = Path(load_path)
        
        with open(load_path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        
        with open(load_path / "kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)
        
        with open(load_path / "article_clusters.pkl", "rb") as f:
            self.article_clusters = pickle.load(f)
        
        self.n_clusters = self.kmeans.n_clusters
        self.cluster_centers = self.kmeans.cluster_centers_
        
        print(f"Model loaded from {load_path}")
        print(f"K={self.n_clusters} clusters, {len(self.article_clusters)} articles")
