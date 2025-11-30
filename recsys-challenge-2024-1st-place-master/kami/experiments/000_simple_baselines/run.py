"""
Simple baseline models that don't require training:
1. Popularity Baseline: Rank by global article popularity
2. Recency Baseline: Rank by article freshness (newest first)
3. Popularity + Recency: Weighted combination
"""

import gc
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parents[2]))

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils


def make_result_df(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Create result dataframe with rankings based on scores"""
    result_df = (
        df.select(["impression_id", "user_id", "article_id", score_col])
        .with_columns(
            pl.col(score_col)
            .rank(method="ordinal", descending=True)
            .over("impression_id")
            .alias("rank")
        )
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg([pl.col("article_id"), pl.col("rank")])
    )
    return result_df


def popularity_baseline(cfg: DictConfig, df: pl.DataFrame, articles_df: pl.DataFrame) -> pl.DataFrame:
    """Rank articles by their global popularity (total pageviews)"""
    print("\n=== Popularity Baseline ===")
    
    # Use total_pageviews as popularity score (higher = more popular)
    popularity_scores = articles_df.select(["article_id", "total_pageviews"])
    
    # Join with candidates
    df_with_scores = df.join(popularity_scores, on="article_id", how="left")
    
    # Fill missing with 0 (articles with no pageviews)
    df_with_scores = df_with_scores.with_columns(
        pl.col("total_pageviews").fill_null(0)
    )
    
    return df_with_scores


def recency_baseline(cfg: DictConfig, df: pl.DataFrame, articles_df: pl.DataFrame) -> pl.DataFrame:
    """Rank articles by recency (newest first)"""
    print("\n=== Recency Baseline ===")
    
    # Use published_time as recency score (higher timestamp = more recent)
    recency_scores = articles_df.select(["article_id", "published_time"])
    
    # Join with candidates
    df_with_scores = df.join(recency_scores, on="article_id", how="left")
    
    # Fill missing with min timestamp
    min_time = articles_df["published_time"].min()
    df_with_scores = df_with_scores.with_columns(
        pl.col("published_time").fill_null(min_time)
    )
    
    return df_with_scores


def popularity_recency_baseline(cfg: DictConfig, df: pl.DataFrame, articles_df: pl.DataFrame) -> pl.DataFrame:
    """Combine popularity and recency with weights"""
    print("\n=== Popularity + Recency Baseline ===")
    
    # Get both scores
    scores = articles_df.select(["article_id", "total_pageviews", "published_time"])
    
    # Normalize to [0, 1] range
    max_pageviews = scores["total_pageviews"].max()
    max_time = scores["published_time"].max()
    min_time = scores["published_time"].min()
    
    scores = scores.with_columns([
        (pl.col("total_pageviews") / max_pageviews).alias("norm_popularity"),
        ((pl.col("published_time") - min_time) / (max_time - min_time)).alias("norm_recency")
    ])
    
    # Combine with weights
    pop_weight = cfg.exp.get("popularity_weight", 0.7)
    rec_weight = 1.0 - pop_weight
    
    scores = scores.with_columns(
        (pop_weight * pl.col("norm_popularity") + rec_weight * pl.col("norm_recency")).alias("combined_score")
    )
    
    # Join with candidates
    df_with_scores = df.join(
        scores.select(["article_id", "combined_score"]), 
        on="article_id", 
        how="left"
    )
    
    # Fill missing with 0
    df_with_scores = df_with_scores.with_columns(
        pl.col("combined_score").fill_null(0)
    )
    
    return df_with_scores


def evaluate_baseline(cfg: DictConfig, validation_df: pl.DataFrame, result_df: pl.DataFrame, baseline_name: str):
    """Evaluate baseline on validation set"""
    print(f"\nEvaluating {baseline_name}...")
    
    # Prepare labels
    labels = (
        validation_df.select(["impression_id", "user_id", "label"])
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("label").cast(pl.Int32))["label"]
        .to_list()
    )
    
    # Prepare predictions (convert rank to scores: lower rank = higher score)
    predictions = result_df.with_columns(
        pl.col("rank").list.eval(1 / pl.element())
    )["rank"].to_list()
    
    # Calculate metrics
    from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
    
    metrics = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]
    evaluator = MetricEvaluator(labels=labels, predictions=predictions, metric_functions=metrics)
    evaluator.evaluate()
    results = evaluator.evaluations
    
    print(f"\n{baseline_name} Results:")
    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.4f}")
    
    return results


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    size_name = cfg.exp.size_name
    dataset_path = Path(cfg.exp.dataset_path)
    
    # Load articles metadata
    print("\nLoading articles metadata...")
    # Try different possible paths
    possible_paths = [
        Path(cfg.dir.input_dir) / "ebnerd_large" / "articles.parquet",
        Path(cfg.dir.input_dir) / "ebnerd_small" / "articles.parquet",
        Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet",
    ]
    
    articles_path = None
    for path in possible_paths:
        if path.exists():
            articles_path = path
            break
    
    if articles_path is None:
        raise FileNotFoundError(f"Could not find articles.parquet in any of: {possible_paths}")
    
    print(f"Loading articles from: {articles_path}")
    articles_df = pl.read_parquet(articles_path)
    print(f"Articles shape: {articles_df.shape}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    validation_df = pl.read_parquet(
        str(dataset_path / size_name / "validation_dataset.parquet")
    )
    print(f"Validation shape: {validation_df.shape}")
    
    baselines = {
        "popularity": (popularity_baseline, "total_pageviews"),
        "recency": (recency_baseline, "published_time"),
        "popularity_recency": (popularity_recency_baseline, "combined_score")
    }
    
    all_results = {}
    
    for baseline_name, (baseline_func, score_col) in baselines.items():
        print(f"\n{'='*60}")
        print(f"Running {baseline_name} baseline...")
        print(f"{'='*60}")
        
        # Generate scores
        df_with_scores = baseline_func(cfg, validation_df, articles_df)
        
        # Create result dataframe
        result_df = make_result_df(df_with_scores, score_col)
        
        # Save results
        result_path = output_path / f"validation_result_{baseline_name}.parquet"
        result_df.write_parquet(result_path)
        print(f"Saved results to {result_path}")
        
        # Evaluate
        results = evaluate_baseline(cfg, validation_df, result_df, baseline_name)
        all_results[baseline_name] = results
        
        del df_with_scores, result_df
        gc.collect()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - All Baselines")
    print(f"{'='*60}")
    print(f"{'Baseline':<25} {'AUC':<10} {'nDCG@5':<10} {'nDCG@10':<10} {'MRR':<10}")
    print("-" * 60)
    
    for baseline_name, results in all_results.items():
        auc = results.get('auc', 0)
        ndcg5 = results.get('ndcg@5', 0)
        ndcg10 = results.get('ndcg@10', 0)
        mrr = results.get('mrr', 0)
        print(f"{baseline_name:<25} {auc:<10.4f} {ndcg5:<10.4f} {ndcg10:<10.4f} {mrr:<10.4f}")
    
    # Save results to file
    with open(output_path / "baseline_results.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("SUMMARY - All Baselines\n")
        f.write("="*60 + "\n")
        f.write(f"{'Baseline':<25} {'AUC':<10} {'nDCG@5':<10} {'nDCG@10':<10} {'MRR':<10}\n")
        f.write("-" * 60 + "\n")
        for baseline_name, results in all_results.items():
            auc = results.get('auc', 0)
            ndcg5 = results.get('ndcg@5', 0)
            ndcg10 = results.get('ndcg@10', 0)
            mrr = results.get('mrr', 0)
            f.write(f"{baseline_name:<25} {auc:<10.4f} {ndcg5:<10.4f} {ndcg10:<10.4f} {mrr:<10.4f}\n")
    
    print(f"\nResults saved to {output_path / 'baseline_results.txt'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
