"""
Single-stage training:
- Train on train dataset only
- Validate on first 50% of validation dataset (for early stopping)
- Test on second 50% of validation dataset (held-out test set)
"""

import gc
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import utils
import wandb
from ebrec.evaluation import MetricEvaluator
from ebrec.evaluation import MultiprocessingAucScore as AucScore
from ebrec.evaluation import MultiprocessingMrrScore as MrrScore
from ebrec.evaluation import MultiprocessingNdcgScore as NdcgScore
from utils.logger import get_logger
from wandb.integration.lightgbm import wandb_callback, log_summary
from graph_features import GraphFeatureExtractor, build_interaction_history
from semantic_cluster_features import SemanticClusterFeatureExtractor

pl.Config.set_ascii_tables(True)

logger = None

group_cols = ["impression_id", "user_id"]


def get_need_cols(cfg: DictConfig, cols: list) -> pl.DataFrame:
    need_cols = ["impression_id", "user_id", "article_id", "label"] + [
        col for col in cols if col not in cfg.exp.lgbm.unuse_cols
    ]
    if cfg.exp.article_stats_cols is False:
        need_cols = [
            col for col in need_cols if col not in cfg.leak_features.article_stats_cols
        ]
    if cfg.exp.past_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.past_impression_cols
        ]
    if cfg.exp.future_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.future_impression_cols
        ]
    return need_cols


def process_df(cfg: DictConfig, df: pl.DataFrame) -> pl.DataFrame:
    mul_cols_dict = cfg.exp.lgbm.mul_cols_dict
    if mul_cols_dict is not None and len(mul_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) * pl.col(cols[1])).alias(name)
                for name, cols in mul_cols_dict.items()
            ]
        )

    div_cols_dict = cfg.exp.lgbm.div_cols_dict
    if div_cols_dict is not None and len(div_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in div_cols_dict.items()
            ]
        )

    norm_cols_dict = cfg.exp.lgbm.norm_cols_dict
    if norm_cols_dict is not None and len(norm_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in norm_cols_dict.items()
            ]
        ).drop([cols[0] for cols in norm_cols_dict.values()])

    need_cols = get_need_cols(cfg, df.columns)
    df = df[need_cols]
    return df


def train_and_valid(
    cfg: DictConfig, train_df: pl.DataFrame, validation_df: pl.DataFrame
) -> lgb.Booster:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    logger.info(f"{len(feature_cols)=} {feature_cols=}")
    label_col = cfg.exp.lgbm.label_col

    if cfg.exp.lgbm.params.two_rounds:
        # bool特徴をintに変換
        bool_cols = [
            col
            for col in [label_col] + feature_cols
            if train_df[col].dtype == pl.Boolean
        ]
        train_df = train_df.with_columns(
            [train_df[col].cast(pl.Int8) for col in bool_cols]
        )
        validation_df = validation_df.with_columns(
            [validation_df[col].cast(pl.Int8) for col in bool_cols]
        )

        with utils.trace("write csv"):
            # テキストファイルへ書き出し
            train_df[[label_col] + feature_cols].write_csv(
                "tmp_train.csv", include_header=False
            )
            validation_df[[label_col] + feature_cols].write_csv(
                "tmp_validation.csv", include_header=False
            )

    print("make lgb.Dataset")
    lgb_train_dataset = lgb.Dataset(
        "tmp_train.csv"
        if cfg.exp.lgbm.params.two_rounds
        else train_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(train_df[label_col]),
        feature_name=feature_cols,
    )
    lgb_valid_dataset = lgb.Dataset(
        "tmp_validation.csv"
        if cfg.exp.lgbm.params.two_rounds
        else validation_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(validation_df[label_col]),
        feature_name=feature_cols,
        categorical_feature=cfg.exp.lgbm.cat_cols,
    )
    """
    if cfg.exp.lgbm.params.two_rounds:
        lgb_train_dataset.construct()
        lgb_valid_dataset.construct()
    """

    if cfg.exp.lgbm.params.objective == "lambdarank":
        print("make train group")
        train_group = (
            train_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("make validation group")
        valid_group = (
            validation_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("set group")
        lgb_train_dataset.set_group(train_group)
        lgb_valid_dataset.set_group(valid_group)
        cfg.exp.lgbm.params["ndcg_eval_at"] = cfg.exp.lgbm.ndcg_eval_at

    print("train")
    bst = lgb.train(
        OmegaConf.to_container(cfg.exp.lgbm.params, resolve=True),
        lgb_train_dataset,
        num_boost_round=cfg.exp.lgbm.num_boost_round,
        valid_sets=[lgb_valid_dataset],
        valid_names=["valid"],
        callbacks=[
            wandb_callback(),
            lgb.early_stopping(
                stopping_rounds=cfg.exp.lgbm.early_stopping_round,
                verbose=True,
                first_metric_only=cfg.exp.lgbm.params.first_metric_only,
            ),
            lgb.log_evaluation(cfg.exp.lgbm.verbose_eval),
        ],
    )
    # log_summary(bst, save_model_checkpoint=True)
    log_summary(bst, save_model_checkpoint=False)
    logger.info(
        f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
    )
    return bst


def predict(
    cfg: DictConfig, bst: lgb.Booster, test_df: pd.DataFrame, num_iteration: int
) -> pd.DataFrame:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in test_df.columns if col not in unuse_cols]
    # batch size で分割して予測
    batch_size = 100000
    y_pred = np.zeros(len(test_df))
    for i in tqdm(range(0, len(test_df), batch_size)):
        y_pred[i : i + batch_size] = bst.predict(
            test_df[feature_cols][i : i + batch_size], num_iteration=num_iteration
        )
    return y_pred


def save_model(cfg: DictConfig, bst: lgb.Booster, output_path: Path, name: int) -> None:
    with open(output_path / f"model_dict_{name}.pkl", "wb") as f:
        pickle.dump({"model": bst}, f)

    # save feature importance (top 100)
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = lgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=100)
    fig.tight_layout()
    fig.savefig(output_path / f"importance_{name}.png")
    plt.close(fig)

    # Get importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": bst.feature_name(),
            "importance": bst.feature_importance(importance_type="gain"),
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Save top 20 most important features
    fig_top, ax_top = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20).sort_values("importance", ascending=True)
    ax_top.barh(top_20["feature"], top_20["importance"])
    ax_top.set_xlabel("Importance (gain)")
    ax_top.set_title(f"Top 20 Most Important Features - {name}")
    fig_top.tight_layout()
    fig_top.savefig(output_path / f"importance_top20_{name}.png")
    plt.close(fig_top)
    
    # Save bottom 20 least important features
    fig_bottom, ax_bottom = plt.subplots(figsize=(10, 8))
    bottom_20 = importance_df.tail(20).sort_values("importance", ascending=True)
    ax_bottom.barh(bottom_20["feature"], bottom_20["importance"])
    ax_bottom.set_xlabel("Importance (gain)")
    ax_bottom.set_title(f"Bottom 20 Least Important Features - {name}")
    fig_bottom.tight_layout()
    fig_bottom.savefig(output_path / f"importance_bottom20_{name}.png")
    plt.close(fig_bottom)

    # Log importance to console
    pd.set_option("display.max_rows", None)
    logger.info(importance_df)
    logger.info(importance_df["feature"].to_list())


def make_result_df(df: pl.DataFrame, pred: np.ndarray):
    assert len(df) == len(pred)
    return (
        df.select(["impression_id", "user_id", "article_id"])
        .with_columns(pl.Series(name="pred", values=pred))
        .with_columns(
            pl.col("pred")
            .rank(method="ordinal", descending=True)
            .over(["impression_id", "user_id"])
            .alias("rank")
        )
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("rank"), pl.col("pred"))
        .select(["impression_id", "rank", "pred"])
    )


def main_stage(cfg: DictConfig, output_path) -> None:
    print("main_stage")
    
    # Set all random seeds for reproducibility
    seed = cfg.exp.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed} for reproducibility")
    
    dataset_path = Path(cfg.exp.dataset_path)

    size_name = cfg.exp.size_name
    if "train" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                print(f"{train_df.shape=}")
                random.seed(cfg.exp.seed)
                train_impression_ids = sorted(
                    train_df["impression_id"].unique().to_list()
                )
                use_train_impression_ids = random.sample(
                    train_impression_ids,
                    int(len(train_impression_ids) * cfg.exp.sampling_rate),
                )
                train_df = train_df.filter(
                    pl.col("impression_id").is_in(use_train_impression_ids)
                )
                print(f"{train_df.shape=}")
                gc.collect()
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            
            # Split validation into validation and test (50/50) BY TIME
            print(f"Original validation shape: {full_validation_df.shape}")
            
            # Get impression_ids from loaded data
            impression_ids = full_validation_df["impression_id"].unique().to_list()
            
            # Load behaviors to get impression_time
            behaviors_path = Path(cfg.dir.input_dir) / f"ebnerd_{cfg.exp.size_name}" / "validation" / "behaviors.parquet"
            behaviors_df = pl.read_parquet(behaviors_path).select(["impression_id", "impression_time"])
            
            # Sort by time
            impression_time_df = behaviors_df.filter(
                pl.col("impression_id").is_in(impression_ids)
            ).sort("impression_time")
            
            all_validation_impression_ids = impression_time_df["impression_id"].to_list()
            
            # Split 50/50 by time (first 50% = validation, second 50% = test)
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            
            print(f"Validation split shape: {validation_df.shape}")
            print(f"Test split shape: {test_df.shape}")
            del full_validation_df
            gc.collect()

        # Graph-based feature extraction (if enabled) - MUST BE BEFORE process_df
        logger.info(f"use_graph_features setting: {cfg.exp.get('use_graph_features', False)}")
        if cfg.exp.get("use_graph_features", False):
            with utils.trace("graph feature extraction"):
                logger.info(f"BEFORE graph features - train_df shape: {train_df.shape}, columns: {len(train_df.columns)}")
                graph_model_path = output_path / "graph_model"
                
                # Check if pre-trained graph model exists
                if (graph_model_path / "user_embeddings.pkl").exists():
                    logger.info("Loading pre-trained graph model...")
                    graph_extractor = GraphFeatureExtractor(
                        embedding_dim=cfg.exp.get("graph_embedding_dim", 64)
                    )
                    graph_extractor.load_model(graph_model_path)
                else:
                    logger.info("Training new graph model from train_df...")
                    # Build interaction DataFrame directly from train_df
                    # Filter to only clicked items (label=1)
                    interactions_df = train_df.filter(pl.col("label") == 1).select(["user_id", "article_id"]).unique()
                    logger.info(f"Built interactions from train_df: {len(interactions_df)} unique user-article pairs")
                    
                    # Initialize and train graph extractor
                    graph_extractor = GraphFeatureExtractor(
                        embedding_dim=cfg.exp.get("graph_embedding_dim", 64),
                        walk_length=cfg.exp.get("graph_walk_length", 30),
                        num_walks=cfg.exp.get("graph_num_walks", 200),
                        workers=cfg.exp.get("graph_workers", 4),
                    )
                    
                    # Build bipartite graph
                    graph_extractor.build_bipartite_graph(interactions_df)
                    
                    # Train Node2Vec
                    graph_extractor.train_node2vec(save_path=graph_model_path)
                    
                    del interactions_df
                    gc.collect()
                
                # Add graph features to datasets
                include_embeddings = cfg.exp.get("graph_include_embeddings", False)
                logger.info(f"Adding graph features (include_embeddings={include_embeddings})...")
                
                logger.info("Adding graph features to train data...")
                train_df = graph_extractor.add_graph_features_to_df(train_df, include_embeddings=include_embeddings)
                logger.info(f"AFTER graph features - train_df shape: {train_df.shape}, columns: {len(train_df.columns)}")
                gc.collect()
                
                logger.info("Adding graph features to validation data...")
                validation_df = graph_extractor.add_graph_features_to_df(validation_df, include_embeddings=include_embeddings)
                gc.collect()
                
                logger.info("Adding graph features to test data...")
                test_df = graph_extractor.add_graph_features_to_df(test_df, include_embeddings=include_embeddings)
                gc.collect()
                
                logger.info(f"Graph features added successfully! Train shape: {train_df.shape}")

        # Semantic cluster feature extraction (if enabled)
        logger.info(f"use_semantic_clusters setting: {cfg.exp.get('use_semantic_clusters', False)}")
        if cfg.exp.get("use_semantic_clusters", False):
            with utils.trace("semantic cluster feature extraction"):
                logger.info(f"BEFORE semantic clusters - train_df shape: {train_df.shape}, columns: {len(train_df.columns)}")
                semantic_model_path = output_path / "semantic_model"
                
                # Check if pre-trained semantic model exists
                if (semantic_model_path / "article_clusters.pkl").exists():
                    logger.info("Loading pre-trained semantic cluster model...")
                    semantic_extractor = SemanticClusterFeatureExtractor(
                        n_clusters=cfg.exp.get("semantic_n_clusters", 30)
                    )
                    semantic_extractor.load_model(semantic_model_path)
                else:
                    logger.info("Training new semantic cluster model...")
                    # Load articles dataset based on size
                    articles_path = Path(cfg.dir.input_dir) / f"ebnerd_{size_name}" / "articles.parquet"
                    if not articles_path.exists():
                        logger.warning(f"Articles file not found: {articles_path}")
                        logger.warning("Skipping semantic cluster feature extraction")
                    else:
                        articles_df = pl.read_parquet(articles_path)
                        
                        # Initialize and train semantic extractor using BERT embeddings
                        semantic_extractor = SemanticClusterFeatureExtractor(
                            n_clusters=cfg.exp.get("semantic_n_clusters", 30),
                            random_state=cfg.exp.seed,
                        )
                        
                        # Train on articles (BERT embeddings loaded automatically)
                        semantic_extractor.fit(articles_df)
                        
                        # Save model
                        semantic_extractor.save_model(semantic_model_path)
                        
                        del articles_df
                        gc.collect()
                
                # Add semantic cluster features to datasets
                if 'semantic_extractor' in locals():
                    # Load history for user profiling from original dataset
                    history_path = Path(cfg.dir.input_dir) / f"ebnerd_{size_name}" / "train" / "history.parquet"
                    if history_path.exists():
                        history_df = pl.read_parquet(history_path)
                    else:
                        history_df = None
                        logger.warning("History file not found, user profiling will use empty histories")
                    
                    logger.info("Adding semantic cluster features to train data...")
                    train_df = semantic_extractor.add_cluster_features_to_df(train_df, history_df)
                    logger.info(f"AFTER semantic clusters - train_df shape: {train_df.shape}, columns: {len(train_df.columns)}")
                    gc.collect()
                    
                    logger.info("Adding semantic cluster features to validation data...")
                    validation_df = semantic_extractor.add_cluster_features_to_df(validation_df, history_df)
                    gc.collect()
                    
                    logger.info("Adding semantic cluster features to test data...")
                    test_df = semantic_extractor.add_cluster_features_to_df(test_df, history_df)
                    gc.collect()
                    
                    if history_df is not None:
                        del history_df
                        gc.collect()
                    
                    logger.info(f"Semantic cluster features added successfully! Train shape: {train_df.shape}")

        with utils.trace("process dataframes"):
            train_df = process_df(cfg, train_df)
            gc.collect()
            validation_df = process_df(cfg, validation_df)
            gc.collect()
            test_df = process_df(cfg, test_df)
            gc.collect()

        with utils.trace("train and valid"):
            bst = train_and_valid(cfg, train_df, validation_df)
            save_model(cfg, bst, output_path, name="model")

        del train_df, validation_df
        gc.collect()

    if "predict" in cfg.exp.first_modes:
        bst = None
        with open(output_path / "model_dict_model.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict validation and test"):
            # Recreate the same split BY TIME
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
            )
            
            # Get impression_ids from loaded data
            impression_ids = full_validation_df["impression_id"].unique().to_list()
            
            # Load behaviors to get impression_time
            behaviors_path = Path(cfg.dir.input_dir) / f"ebnerd_{cfg.exp.size_name}" / "validation" / "behaviors.parquet"
            behaviors_df = pl.read_parquet(behaviors_path).select(["impression_id", "impression_time"])
            
            # Sort by time
            impression_time_df = behaviors_df.filter(
                pl.col("impression_id").is_in(impression_ids)
            ).sort("impression_time")
            
            all_validation_impression_ids = impression_time_df["impression_id"].to_list()
            
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            del full_validation_df
            
            # Add graph features if enabled - MUST BE BEFORE process_df
            if cfg.exp.get("use_graph_features", False):
                graph_model_path = output_path / "graph_model"
                if (graph_model_path / "user_embeddings.pkl").exists():
                    logger.info("Loading graph model for prediction...")
                    graph_extractor = GraphFeatureExtractor(
                        embedding_dim=cfg.exp.get("graph_embedding_dim", 64)
                    )
                    graph_extractor.load_model(graph_model_path)
                    
                    include_embeddings = cfg.exp.get("graph_include_embeddings", False)
                    logger.info(f"Adding graph features for prediction (include_embeddings={include_embeddings})...")
                    validation_df = graph_extractor.add_graph_features_to_df(validation_df, include_embeddings=include_embeddings)
                    test_df = graph_extractor.add_graph_features_to_df(test_df, include_embeddings=include_embeddings)
            
            # Add semantic cluster features if enabled - MUST BE BEFORE process_df
            if cfg.exp.get("use_semantic_clusters", False):
                semantic_model_path = output_path / "semantic_model"
                if not (semantic_model_path / "article_clusters.pkl").exists():
                    raise FileNotFoundError(
                        f"Semantic clustering is enabled but model not found at {semantic_model_path}. "
                        "Please run training first to generate the semantic model."
                    )
                
                logger.info("Loading semantic cluster model for prediction...")
                semantic_extractor = SemanticClusterFeatureExtractor(
                    n_clusters=cfg.exp.get("semantic_n_clusters", 30)
                )
                semantic_extractor.load_model(semantic_model_path)
                
                # Load history for user profiling
                history_path = dataset_path / size_name / "history.parquet"
                if history_path.exists():
                    history_df = pl.read_parquet(str(history_path))
                else:
                    history_df = None
                    logger.warning("History file not found for prediction, using empty histories")
                
                logger.info("Adding semantic cluster features to validation data...")
                validation_df = semantic_extractor.add_cluster_features_to_df(validation_df, history_df)
                logger.info("Adding semantic cluster features to test data...")
                test_df = semantic_extractor.add_cluster_features_to_df(test_df, history_df)
                
                if history_df is not None:
                    del history_df
                gc.collect()
            
            validation_df = process_df(cfg, validation_df)
            test_df = process_df(cfg, test_df)
            
            y_valid_pred = predict(
                cfg, bst, validation_df, num_iteration=bst.best_iteration
            )
            validation_result_df = make_result_df(validation_df, y_valid_pred)
            validation_result_df.write_parquet(
                output_path / "validation_result.parquet"
            )
            print(f"Validation: {validation_result_df}")
            del validation_df, validation_result_df
            gc.collect()
            
            y_test_pred = predict(cfg, bst, test_df, num_iteration=bst.best_iteration)
            test_result_df = make_result_df(test_df, y_test_pred)
            test_result_df.write_parquet(output_path / "test_result.parquet")
            print(f"Test: {test_result_df}")
            del test_df, test_result_df
            gc.collect()

    if "eval" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            # Recreate the same split BY TIME
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
            )
            
            # Get impression_ids from loaded data
            impression_ids = full_validation_df["impression_id"].unique().to_list()
            
            # Load behaviors to get impression_time
            behaviors_path = Path(cfg.dir.input_dir) / f"ebnerd_{cfg.exp.size_name}" / "validation" / "behaviors.parquet"
            behaviors_df = pl.read_parquet(behaviors_path).select(["impression_id", "impression_time"])
            
            # Sort by time
            impression_time_df = behaviors_df.filter(
                pl.col("impression_id").is_in(impression_ids)
            ).sort("impression_time")
            
            all_validation_impression_ids = impression_time_df["impression_id"].to_list()
            
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            del full_validation_df
            
            # Add graph features if enabled - MUST BE BEFORE process_df
            if cfg.exp.get("use_graph_features", False):
                graph_model_path = output_path / "graph_model"
                if (graph_model_path / "user_embeddings.pkl").exists():
                    logger.info("Loading graph model for eval...")
                    graph_extractor = GraphFeatureExtractor(
                        embedding_dim=cfg.exp.get("graph_embedding_dim", 64)
                    )
                    graph_extractor.load_model(graph_model_path)
                    
                    include_embeddings = cfg.exp.get("graph_include_embeddings", False)
                    logger.info(f"Adding graph features for eval (include_embeddings={include_embeddings})...")
                    validation_df = graph_extractor.add_graph_features_to_df(validation_df, include_embeddings=include_embeddings)
                    test_df = graph_extractor.add_graph_features_to_df(test_df, include_embeddings=include_embeddings)
            
            # Add semantic cluster features if enabled - MUST BE BEFORE process_df
            if cfg.exp.get("use_semantic_clusters", False):
                semantic_model_path = output_path / "semantic_model"
                if not (semantic_model_path / "article_clusters.pkl").exists():
                    raise FileNotFoundError(
                        f"Semantic clustering is enabled but model not found at {semantic_model_path}. "
                        "Please run training first to generate the semantic model."
                    )
                
                logger.info("Loading semantic cluster model for eval...")
                semantic_extractor = SemanticClusterFeatureExtractor(
                    n_clusters=cfg.exp.get("semantic_n_clusters", 30)
                )
                semantic_extractor.load_model(semantic_model_path)
                
                # Load history for user profiling
                history_path = dataset_path / size_name / "history.parquet"
                if history_path.exists():
                    history_df = pl.read_parquet(str(history_path))
                else:
                    history_df = None
                    logger.warning("History file not found for eval, using empty histories")
                
                logger.info("Adding semantic cluster features for eval...")
                validation_df = semantic_extractor.add_cluster_features_to_df(validation_df, history_df)
                test_df = semantic_extractor.add_cluster_features_to_df(test_df, history_df)
                
                if history_df is not None:
                    del history_df
                gc.collect()
            
            validation_df = process_df(cfg, validation_df)
            test_df = process_df(cfg, test_df)
            
            validation_result_df = pl.read_parquet(
                output_path / "validation_result.parquet",
            )
            test_result_df = pl.read_parquet(
                output_path / "test_result.parquet",
            )
            
        with utils.trace("prepare eval validation"):
            labels = (
                validation_df.select(["impression_id", "user_id", "label"])
                .group_by(["impression_id", "user_id"], maintain_order=True)
                .agg(pl.col("label").cast(pl.Int32))["label"]
                .to_list()
            )
            predictions = validation_result_df.with_columns(
                pl.col("rank").list.eval(1 / pl.element())
            )["rank"].to_list()

        with utils.trace("eval validation"):
            metric_functions = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]

            metrics = MetricEvaluator(
                labels=labels,
                predictions=predictions,
                metric_functions=metric_functions,
            )
            metrics.evaluate()
            val_result_dict = metrics.evaluations

            logger.info(f"VALIDATION: {val_result_dict}")
            wandb.log({"val_" + k: v for k, v in val_result_dict.items()})
        
        with utils.trace("prepare eval test"):
            test_labels = (
                test_df.select(["impression_id", "user_id", "label"])
                .group_by(["impression_id", "user_id"], maintain_order=True)
                .agg(pl.col("label").cast(pl.Int32))["label"]
                .to_list()
            )
            test_predictions = test_result_df.with_columns(
                pl.col("rank").list.eval(1 / pl.element())
            )["rank"].to_list()

        with utils.trace("eval test"):
            metric_functions = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]

            test_metrics = MetricEvaluator(
                labels=test_labels,
                predictions=test_predictions,
                metric_functions=metric_functions,
            )
            test_metrics.evaluate()
            test_result_dict = test_metrics.evaluations

            logger.info(f"TEST: {test_result_dict}")
            wandb.log({"test_" + k: v for k, v in test_result_dict.items()})
        
        # Save results to a separate file for easy access
        with utils.trace("save results to file"):
            results_file = output_path / "results.txt"
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("EVALUATION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("VALIDATION METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric_name, metric_value in val_result_dict.items():
                    f.write(f"  {metric_name:20s}: {metric_value:.6f}\n")
                
                f.write("\n")
                f.write("TEST METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric_name, metric_value in test_result_dict.items():
                    f.write(f"  {metric_name:20s}: {metric_value:.6f}\n")
                
                f.write("\n")
                f.write("=" * 60 + "\n")
            
            logger.info(f"Results saved to: {results_file}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    # Add timestamp and seed to output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = cfg.exp.seed
    exp_name_with_meta = f"{exp_name}_seed{seed}_{timestamp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name_with_meta
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")

    logger.info(f"exp_name: {exp_name}")
    logger.info(f"ouput_path: {output_path}")
    logger.info(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="recsys2024",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug or cfg.exp.size_name == "demo" else "online",
    )

    main_stage(cfg, output_path)


if __name__ == "__main__":
    main()
