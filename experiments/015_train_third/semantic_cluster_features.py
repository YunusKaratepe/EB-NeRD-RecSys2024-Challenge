"""
Semantic Clustering Feature Extraction

Supports two modes:
1. BERT mode: Uses pre-computed BERT embeddings from google_bert_base_multilingual_cased
2. TF-IDF mode: Computes TF-IDF vectors from article text on-the-fly

Both modes apply K-Means clustering for semantic grouping.
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm


class SemanticClusterFeatureExtractor:
    """
    Extract semantic cluster-based features for news recommendation.
    
    Supports two modes:
    - BERT: Uses pre-computed BERT embeddings (768-dim)
    - TF-IDF: Computes TF-IDF vectors from text
    
    Implements the methodology from Section 4.4.2 of the research report:
    1. Global Clustering: Create K semantic clusters from all articles
    2. User Profiling: Calculate user's interest distribution across clusters
    3. Semantic Features: Compute BERT/TF-IDF-based similarity features
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        random_state: int = 42,
        mode: str = "bert",  # "bert" or "tfidf"
        bert_embeddings_path: Optional[str] = None,
        max_tfidf_features: int = 5000,
    ):
        """
        Initialize Semantic Cluster Feature Extractor.
        
        Args:
            n_clusters: Number of semantic clusters (K)
            random_state: Random seed for reproducibility
            mode: "bert" to use pre-computed BERT embeddings, "tfidf" to use TF-IDF
            bert_embeddings_path: Path to pre-computed BERT embeddings parquet file (for BERT mode)
            max_tfidf_features: Maximum number of TF-IDF features (for TF-IDF mode)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.mode = mode.lower()
        self.bert_embeddings_path = bert_embeddings_path or "input/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
        self.max_tfidf_features = max_tfidf_features
        
        self.kmeans = None
        self.article_clusters = {}  # article_id -> cluster_id
        self.cluster_centers = None
        self.bert_embeddings_df = None  # Cache BERT embeddings
        self.article_embeddings_cache = {}  # article_id -> embedding vector
        self.tfidf_vectorizer = None  # For TF-IDF mode
        self.embedding_dim = 768 if mode == "bert" else max_tfidf_features
        self.user_profile_embeddings_cache = {}  # Cache user profiles to avoid recomputation
    
    def fit(
        self,
        articles_df: pl.DataFrame,
        text_column: str = "title",
    ) -> 'SemanticClusterFeatureExtractor':
        """
        Fit global semantic clusters on all articles.
        
        Args:
            articles_df: DataFrame with article IDs and text
            text_column: Column name for text (used in TF-IDF mode)
        
        Returns:
            self
        """
        print(f"Fitting semantic clusters in {self.mode.upper()} mode on {len(articles_df)} articles...")
        
        if self.mode == "bert":
            article_vectors = self._load_bert_embeddings(articles_df)
            # For BERT mode, we need to filter articles_df to only those with embeddings
            # Join to ensure alignment
            bert_df_temp = pl.read_parquet(self.bert_embeddings_path)
            articles_df = articles_df.join(
                bert_df_temp.select(["article_id"]), 
                on="article_id", 
                how="inner"
            )
            print(f"After filtering for BERT embeddings: {len(articles_df)} articles")
        elif self.mode == "tfidf":
            article_vectors = self._compute_tfidf_vectors(articles_df, text_column)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'bert' or 'tfidf'")
        
        # Extract article IDs
        article_ids = articles_df["article_id"].to_list()
        
        print(f"Article vectors shape: {article_vectors.shape}")
        print(f"Embedding dimension: {article_vectors.shape[1]}")
        
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
        
        # Cache article embeddings for later use
        print("Caching article embeddings...")
        for article_id, embedding in zip(article_ids, article_vectors):
            self.article_embeddings_cache[article_id] = embedding
        
        # Cluster statistics
        cluster_counts = np.bincount(cluster_labels, minlength=self.n_clusters)
        print(f"Cluster distribution: min={cluster_counts.min()}, "
              f"max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
        
        return self
    
    def _load_bert_embeddings(self, articles_df: pl.DataFrame) -> np.ndarray:
        """Load pre-computed BERT embeddings for articles."""
        print(f"Loading BERT embeddings from {self.bert_embeddings_path}...")
        bert_df = pl.read_parquet(self.bert_embeddings_path)
        self.bert_embeddings_df = bert_df  # Cache for later use
        
        print(f"Loaded BERT embeddings for {len(bert_df)} articles")
        
        # Join with articles_df to maintain order
        articles_with_embeddings = articles_df.join(
            bert_df, 
            on="article_id", 
            how="inner"
        )
        
        if len(articles_with_embeddings) < len(articles_df):
            missing_count = len(articles_df) - len(articles_with_embeddings)
            print(f"WARNING: {missing_count} articles have no BERT embeddings")
        
        # BERT embeddings are stored as list/array in a single column
        embedding_col = 'google-bert/bert-base-multilingual-cased'
        
        if embedding_col not in articles_with_embeddings.columns:
            raise ValueError(f"Expected embedding column '{embedding_col}' not found. Available columns: {articles_with_embeddings.columns}")
        
        # Convert embeddings to numpy array
        embeddings_list = articles_with_embeddings[embedding_col].to_list()
        article_vectors = np.array(embeddings_list, dtype=np.float32)
        
        return article_vectors
    
    def _compute_tfidf_vectors(self, articles_df: pl.DataFrame, text_column: str) -> np.ndarray:
        """Compute TF-IDF vectors from article text."""
        print(f"Computing TF-IDF vectors from '{text_column}' column...")
        
        # Extract text
        texts = articles_df[text_column].fill_null("").to_list()
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        
        article_vectors = self.tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
        
        print(f"TF-IDF vectorization complete: {article_vectors.shape}")
        return article_vectors
    
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
            DataFrame with added cluster and BERT-based semantic features
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
        
        # Build user profile embeddings (mean of history embeddings)
        # Check if we have cached profiles already
        if not self.user_profile_embeddings_cache:
            print("Building user profile embeddings (first time - will be cached)...")
            for user_id, history in tqdm(user_histories.items(), desc="User profiles"):
                history_embeddings = []
                for article_id in history:
                    if article_id in self.article_embeddings_cache:
                        history_embeddings.append(self.article_embeddings_cache[article_id])
                if history_embeddings:
                    self.user_profile_embeddings_cache[user_id] = np.mean(history_embeddings, axis=0)
                else:
                    # No valid history embeddings - use zero vector
                    self.user_profile_embeddings_cache[user_id] = np.zeros(self.embedding_dim)
        else:
            print("Using cached user profile embeddings...")
        
        # Get user profile and article embeddings for each row
        print("Extracting embeddings for each user-article pair...")
        user_embeds = []
        article_embeds = []
        for uid, aid in tqdm(zip(user_ids, article_ids), total=len(user_ids), desc="Embeddings"):
            user_embed = self.user_profile_embeddings_cache.get(uid, np.zeros(self.embedding_dim))
            article_embed = self.article_embeddings_cache.get(aid, np.zeros(self.embedding_dim))
            user_embeds.append(user_embed)
            article_embeds.append(article_embed)
        
        user_embeds = np.array(user_embeds)
        article_embeds = np.array(article_embeds)
        
        # Calculate user cluster distributions (frequency-based)
        print("Calculating user cluster distributions...")
        user_distributions = []
        for user_id in tqdm(user_ids, desc="Cluster distributions"):
            history = user_histories.get(user_id, [])
            distribution = self._calculate_user_cluster_distribution(history)
            user_distributions.append(distribution)
        user_distributions = np.array(user_distributions)
        
        # ===== NEW BERT-BASED SEMANTIC FEATURES =====
        print(f"Calculating {self.mode.upper()}-based semantic features...")
        
        # 1. Cosine Similarity: user profile vs article
        print("Computing cosine similarities...")
        cosine_sims = []
        for user_embed, article_embed in zip(user_embeds, article_embeds):
            # Normalize and compute cosine similarity
            user_norm = np.linalg.norm(user_embed)
            article_norm = np.linalg.norm(article_embed)
            if user_norm > 0 and article_norm > 0:
                sim = np.dot(user_embed, article_embed) / (user_norm * article_norm)
            else:
                sim = 0.0
            cosine_sims.append(sim)
        
        df = df.with_columns(
            pl.Series(name=f"{self.mode}_user_article_cosine_sim", values=cosine_sims)
        )
        
        # 2. Cluster Center Distance: article to its cluster center
        print("Computing cluster center distances...")
        cluster_distances = []
        for aid, article_embed in zip(article_ids, article_embeds):
            cluster_id = self.article_clusters.get(aid, -1)
            if cluster_id >= 0 and cluster_id < len(self.cluster_centers):
                cluster_center = self.cluster_centers[cluster_id]
                distance = np.linalg.norm(article_embed - cluster_center)
            else:
                distance = 0.0
            cluster_distances.append(distance)
        
        df = df.with_columns(
            pl.Series(name=f"{self.mode}_article_cluster_distance", values=cluster_distances)
        )
        
        # 3. Max Similarity to User History: max cosine sim to any history article
        print("Computing max similarity to user history (batched)...")
        max_history_sims = []
        max_history_length = 20  # Limit to last 20 articles for speed
        
        # Pre-normalize article embeddings once
        article_embeds_normalized = normalize(article_embeds, norm='l2')
        
        for uid, article_embed_norm in tqdm(zip(user_ids, article_embeds_normalized), total=len(user_ids), desc="Max history sim"):
            history = user_histories.get(uid, [])
            if history:
                # Limit history length - use last N items
                history = history[-max_history_length:] if len(history) > max_history_length else history
                
                # Collect history embeddings
                history_embeds = [
                    self.article_embeddings_cache[hist_aid] 
                    for hist_aid in history 
                    if hist_aid in self.article_embeddings_cache
                ]
                
                if history_embeds:
                    history_matrix = np.array(history_embeds)
                    # Normalize history embeddings
                    history_matrix_norm = normalize(history_matrix, norm='l2')
                    # Vectorized cosine similarity (dot product of normalized vectors)
                    sims = np.dot(history_matrix_norm, article_embed_norm)
                    max_sim = float(np.max(sims))
                else:
                    max_sim = 0.0
            else:
                max_sim = 0.0
            max_history_sims.append(max_sim)
        
        df = df.with_columns(
            pl.Series(name=f"{self.mode}_max_history_similarity", values=max_history_sims)
        )
        
        # Add user cluster distribution features
        for i in range(self.n_clusters):
            df = df.with_columns(
                pl.Series(name=f"{self.mode}_user_cluster_{i}", values=user_distributions[:, i])
            )
        
        # Add article cluster ID
        article_cluster_ids = [
            self.article_clusters.get(aid, -1) for aid in article_ids
        ]
        df = df.with_columns(
            pl.Series(name=f"{self.mode}_article_cluster", values=article_cluster_ids)
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
            pl.Series(name=f"{self.mode}_user_article_cluster_affinity", values=affinities)
        )
        
        # Add cluster match (binary: is article in user's top cluster?)
        user_top_clusters = user_distributions.argmax(axis=1)
        matches = [
            1 if article_cluster_ids[i] == user_top_clusters[i] else 0
            for i in range(len(article_ids))
        ]
        df = df.with_columns(
            pl.Series(name=f"{self.mode}_user_article_cluster_match", values=matches)
        )
        
        num_features = self.n_clusters + 6  # Old: 3, New: +3 semantic features = 6 total
        print(f"Added {num_features} semantic cluster features ({self.mode.upper()} mode)")
        print(f"  - {self.n_clusters} user cluster distributions")
        print(f"  - 1 article cluster ID")
        print(f"  - 1 user-article cluster affinity")
        print(f"  - 1 cluster match (binary)")
        print(f"  - 1 {self.mode.upper()} cosine similarity (user profile vs article)")
        print(f"  - 1 {self.mode.upper()} cluster center distance")
        print(f"  - 1 {self.mode.upper()} max history similarity")
        return df
    
    def save_model(self, save_path: Path) -> None:
        """Save the trained clustering model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save kmeans, article clusters, embeddings cache, and vectorizer (if TF-IDF mode)
        with open(save_path / "kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
        
        with open(save_path / "article_clusters.pkl", "wb") as f:
            pickle.dump(self.article_clusters, f)
        
        with open(save_path / "article_embeddings_cache.pkl", "wb") as f:
            pickle.dump(self.article_embeddings_cache, f)
        
        if self.mode == "tfidf" and self.tfidf_vectorizer is not None:
            with open(save_path / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Save config
        config = {
            'n_clusters': self.n_clusters,
            'mode': self.mode,
            'bert_embeddings_path': self.bert_embeddings_path,
            'max_tfidf_features': self.max_tfidf_features,
            'embedding_dim': self.embedding_dim,
        }
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print(f"Model saved to {save_path} (mode: {self.mode})")
    
    def load_model(self, load_path: Path) -> None:
        """Load a trained clustering model."""
        load_path = Path(load_path)
        
        # Load config
        with open(load_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        self.n_clusters = config['n_clusters']
        self.mode = config.get('mode', 'bert')  # Default to bert for old models
        self.bert_embeddings_path = config.get('bert_embeddings_path', self.bert_embeddings_path)
        self.max_tfidf_features = config.get('max_tfidf_features', 5000)
        self.embedding_dim = config.get('embedding_dim', 768)
        
        # Load kmeans, article clusters, and embeddings cache
        with open(load_path / "kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)
        
        with open(load_path / "article_clusters.pkl", "rb") as f:
            self.article_clusters = pickle.load(f)
        
        with open(load_path / "article_embeddings_cache.pkl", "rb") as f:
            self.article_embeddings_cache = pickle.load(f)
        
        # Load TF-IDF vectorizer if in TF-IDF mode
        if self.mode == "tfidf":
            tfidf_path = load_path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
        
        self.cluster_centers = self.kmeans.cluster_centers_
        
        print(f"Model loaded from {load_path} (mode: {self.mode})")
        print(f"K={self.n_clusters} clusters, {len(self.article_clusters)} articles")
        print(f"Cached {len(self.article_embeddings_cache)} article embeddings")
