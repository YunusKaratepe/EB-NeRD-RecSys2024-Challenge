"""
Graph-Based Feature Extraction Module
Implements bipartite graph construction and structural embeddings (Node2Vec/DeepWalk)
for user-article interactions as described in the research report.
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import polars as pl
from node2vec import Node2Vec
from tqdm.auto import tqdm


class GraphFeatureExtractor:
    """
    Extracts structural embeddings from user-article bipartite graph.
    
    Implements the methodology described in Section 4.4.1 of the research report:
    - Constructs bipartite graph from user-article interactions
    - Applies Node2Vec algorithm to learn structural embeddings
    - Generates embedding features for both users and articles
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4,
        window: int = 10,
        min_count: int = 1,
        batch_words: int = 4,
    ):
        """
        Initialize Graph Feature Extractor.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            walk_length: Length of random walks
            num_walks: Number of random walks per node
            p: Return parameter (controls likelihood of returning to previous node)
            q: In-out parameter (controls exploration vs exploitation)
            workers: Number of parallel workers
            window: Context window size for Word2Vec
            min_count: Minimum word count for Word2Vec
            batch_words: Batch size for Word2Vec training
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        
        self.graph = None
        self.node2vec_model = None
        self.user_embeddings = {}
        self.article_embeddings = {}
    
    def build_bipartite_graph(
        self,
        interactions_df: pl.DataFrame,
        user_col: str = "user_id",
        article_col: str = "article_id",
        weight_col: str = None,
    ) -> nx.Graph:
        """
        Build bipartite graph from user-article interactions.
        
        Args:
            interactions_df: DataFrame with user-article interactions
            user_col: Column name for user IDs
            article_col: Column name for article IDs
            weight_col: Optional column for edge weights (e.g., reading time)
        
        Returns:
            NetworkX bipartite graph
        """
        print("Building bipartite graph...")
        G = nx.Graph()
        
        # Add nodes with bipartite labels
        unique_users = interactions_df[user_col].unique().to_list()
        unique_articles = interactions_df[article_col].unique().to_list()
        
        # Prefix to distinguish users from articles
        user_nodes = [f"u_{uid}" for uid in unique_users]
        article_nodes = [f"a_{aid}" for aid in unique_articles]
        
        G.add_nodes_from(user_nodes, bipartite=0)
        G.add_nodes_from(article_nodes, bipartite=1)
        
        # Add edges
        print("Adding edges...")
        if weight_col is not None:
            edges = [
                (f"u_{row[user_col]}", f"a_{row[article_col]}", row[weight_col])
                for row in interactions_df.select([user_col, article_col, weight_col]).iter_rows(named=True)
            ]
            G.add_weighted_edges_from(edges)
        else:
            edges = [
                (f"u_{row[user_col]}", f"a_{row[article_col]}")
                for row in interactions_df.select([user_col, article_col]).iter_rows(named=True)
            ]
            G.add_edges_from(edges)
        
        print(f"Graph built: {len(user_nodes)} users, {len(article_nodes)} articles, {G.number_of_edges()} edges")
        self.graph = G
        return G
    
    def train_node2vec(self, save_path: Path = None) -> None:
        """
        Train Node2Vec model on the bipartite graph.
        
        Args:
            save_path: Optional path to save the trained model
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_bipartite_graph() first.")
        
        print("Training Node2Vec model...")
        print(f"Parameters: dim={self.embedding_dim}, walk_length={self.walk_length}, "
              f"num_walks={self.num_walks}, p={self.p}, q={self.q}")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
        )
        
        # Train Word2Vec model
        model = node2vec.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words,
        )
        
        self.node2vec_model = model
        
        # Extract embeddings
        print("Extracting embeddings...")
        for node in tqdm(self.graph.nodes()):
            embedding = model.wv[node]
            if node.startswith("u_"):
                user_id = node[2:]  # Remove 'u_' prefix
                self.user_embeddings[user_id] = embedding
            elif node.startswith("a_"):
                article_id = node[2:]  # Remove 'a_' prefix
                self.article_embeddings[article_id] = embedding
        
        print(f"Extracted {len(self.user_embeddings)} user embeddings and "
              f"{len(self.article_embeddings)} article embeddings")
        
        # Save model if path provided
        if save_path is not None:
            self.save_model(save_path)
    
    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get embedding for a specific user."""
        return self.user_embeddings.get(str(user_id), np.zeros(self.embedding_dim))
    
    def get_article_embedding(self, article_id: int) -> np.ndarray:
        """Get embedding for a specific article."""
        return self.article_embeddings.get(str(article_id), np.zeros(self.embedding_dim))
    
    def add_graph_features_to_df(
        self,
        df: pl.DataFrame,
        user_col: str = "user_id",
        article_col: str = "article_id",
        include_embeddings: bool = False,
    ) -> pl.DataFrame:
        """
        Add graph-based features to a DataFrame.
        
        Args:
            df: Input DataFrame with user and article IDs
            user_col: Column name for user IDs
            article_col: Column name for article IDs
            include_embeddings: If True, adds all embedding dimensions. If False, only interaction features.
        
        Returns:
            DataFrame with added graph features
        """
        print("Adding graph features to DataFrame...")
        
        # Prepare user and article embeddings
        user_ids = df[user_col].to_list()
        article_ids = df[article_col].to_list()
        
        user_embeds = np.array([self.get_user_embedding(uid) for uid in tqdm(user_ids, desc="User embeddings")])
        article_embeds = np.array([self.get_article_embedding(aid) for aid in tqdm(article_ids, desc="Article embeddings")])
        
        if include_embeddings:
            # Add user embedding features
            for i in range(self.embedding_dim):
                df = df.with_columns(
                    pl.Series(name=f"g_user_emb_{i}", values=user_embeds[:, i])
                )
            
            # Add article embedding features
            for i in range(self.embedding_dim):
                df = df.with_columns(
                    pl.Series(name=f"g_article_emb_{i}", values=article_embeds[:, i])
                )
        
        # Add interaction features (dot product, cosine similarity, euclidean distance)
        dot_product = np.sum(user_embeds * article_embeds, axis=1)
        df = df.with_columns(pl.Series(name="g_dot_product", values=dot_product))
        
        user_norm = np.linalg.norm(user_embeds, axis=1)
        article_norm = np.linalg.norm(article_embeds, axis=1)
        cosine_sim = dot_product / (user_norm * article_norm + 1e-8)
        df = df.with_columns(pl.Series(name="g_cosine_sim", values=cosine_sim))
        
        euclidean_dist = np.linalg.norm(user_embeds - article_embeds, axis=1)
        df = df.with_columns(pl.Series(name="g_euclidean_dist", values=euclidean_dist))
        
        num_features = 3 if not include_embeddings else (2 * self.embedding_dim + 3)
        print(f"Added {num_features} graph features (embeddings={'included' if include_embeddings else 'excluded'})")
        return df
    
    def save_model(self, save_path: Path) -> None:
        """Save the trained model and embeddings."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Node2Vec model
        if self.node2vec_model is not None:
            self.node2vec_model.save(str(save_path / "node2vec.model"))
        
        # Save embeddings
        with open(save_path / "user_embeddings.pkl", "wb") as f:
            pickle.dump(self.user_embeddings, f)
        
        with open(save_path / "article_embeddings.pkl", "wb") as f:
            pickle.dump(self.article_embeddings, f)
        
        # Save graph using pickle
        with open(save_path / "bipartite_graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path) -> None:
        """Load a trained model and embeddings."""
        load_path = Path(load_path)
        
        # Load embeddings
        with open(load_path / "user_embeddings.pkl", "rb") as f:
            self.user_embeddings = pickle.load(f)
        
        with open(load_path / "article_embeddings.pkl", "rb") as f:
            self.article_embeddings = pickle.load(f)
        
        # Load graph if exists (optional, only needed for retraining)
        graph_path = load_path / "bipartite_graph.gpickle"
        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
        
        print(f"Model loaded from {load_path}")
        print(f"Loaded {len(self.user_embeddings)} user embeddings and "
              f"{len(self.article_embeddings)} article embeddings")


def build_interaction_history(
    behaviors_df: pl.DataFrame,
    history_df: pl.DataFrame = None,
) -> pl.DataFrame:
    """
    Build complete interaction history from behaviors and history files.
    
    Args:
        behaviors_df: Behaviors DataFrame with clicked articles
        history_df: Optional history DataFrame with past interactions
    
    Returns:
        DataFrame with user-article interactions
    """
    interactions = []
    
    # Extract clicks from behaviors
    for row in behaviors_df.iter_rows(named=True):
        user_id = row["user_id"]
        clicked_articles = row.get("article_ids_clicked", [])
        if clicked_articles is not None and len(clicked_articles) > 0:
            for article_id in clicked_articles:
                interactions.append({"user_id": user_id, "article_id": article_id})
    
    # Add history if available
    if history_df is not None:
        for row in history_df.iter_rows(named=True):
            user_id = row["user_id"]
            article_history = row.get("article_id_fixed", [])
            if article_history is not None and len(article_history) > 0:
                for article_id in article_history:
                    interactions.append({"user_id": user_id, "article_id": article_id})
    
    return pl.DataFrame(interactions).unique()
