"""
Quick cold-start analysis for medium dataset experiments.
"""

import polars as pl
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore


def analyze_experiment(exp_path: str, size_name: str = "medium"):
    """Analyze cold-start performance for an experiment."""
    exp_path = Path(exp_path)
    
    # Load articles from input folder
    articles_path = Path(f"input/ebnerd_{size_name}/articles.parquet")
    
    if not articles_path.exists():
        print(f"ERROR: Could not find articles.parquet at {articles_path}")
        return
    
    articles_df = pl.read_parquet(articles_path)
    
    # Identify cold-start articles (bottom 20%)
    threshold = articles_df["total_inviews"].quantile(0.2)
    cold_articles = set(
        articles_df.filter(pl.col("total_inviews") <= threshold)["article_id"].to_list()
    )
    
    # Load test dataset with labels (second 50% of validation set)
    test_data = pl.read_parquet(f"output/preprocess/dataset067/{size_name}/test_dataset.parquet")
    
    # Load test results (predictions on second 50% of validation set)
    test_result = pl.read_parquet(exp_path / "test_result.parquet")
    
    # Explode predictions to match test_data structure
    # Group test_data by impression_id to get article_ids in order
    test_grouped = (
        test_data
        .group_by("impression_id", maintain_order=True)
        .agg([
            pl.col("article_id"),
            pl.col("label")
        ])
    )
    
    # Join with results and explode
    merged_grouped = test_grouped.join(
        test_result.select(["impression_id", "pred"]),
        on="impression_id",
        how="inner"
    )
    
    # Explode to get row per candidate
    merged = merged_grouped.select([
        pl.col("impression_id").repeat_by(pl.col("article_id").list.len()).explode(),
        pl.col("article_id").explode(),
        pl.col("label").explode(),
        pl.col("pred").explode().alias("prediction")
    ])
    
    # Separate cold-start vs popular items
    cold_mask = merged["article_id"].is_in(cold_articles)
    cold_data = merged.filter(cold_mask)
    popular_data = merged.filter(~cold_mask)
    
    # Calculate metrics
    def calc_metrics(df):
        labels = df["label"].to_list()
        preds = df["prediction"].to_list()
        
        evaluator = MetricEvaluator(
            labels=labels,
            predictions=preds,
            metric_functions=[AucScore(), NdcgScore(k=10), MrrScore()],
        )
        evaluator.evaluate()
        return evaluator.evaluations
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_path.name}")
    print(f"{'='*80}")
    print(f"Total articles: {len(articles_df)}")
    print(f"Cold-start articles (≤{threshold:.1f} inviews): {len(cold_articles)} ({len(cold_articles)/len(articles_df)*100:.1f}%)")
    print(f"\nTest impressions with cold items: {cold_data['impression_id'].n_unique()}")
    print(f"Test impressions with popular items: {popular_data['impression_id'].n_unique()}")
    
    if len(cold_data) > 0:
        print(f"\n--- COLD-START ITEMS ---")
        cold_metrics = calc_metrics(cold_data)
        for metric_name, metric_value in cold_metrics.items():
            print(f"  {metric_name:20s}: {metric_value:.6f}")
    
    if len(popular_data) > 0:
        print(f"\n--- POPULAR ITEMS ---")
        pop_metrics = calc_metrics(popular_data)
        for metric_name, metric_value in pop_metrics.items():
            print(f"  {metric_name:20s}: {metric_value:.6f}")
    
    print(f"\n--- ALL ITEMS ---")
    all_metrics = calc_metrics(merged)
    for metric_name, metric_value in all_metrics.items():
        print(f"  {metric_name:20s}: {metric_value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment paths")
    parser.add_argument("--size", default="medium", help="Dataset size (small/medium)")
    args = parser.parse_args()
    
    for exp in args.experiments:
        analyze_experiment(exp, args.size)
