"""
Analyze model performance on cold-start (least popular) items.
This shows how well the model handles items with sparse interaction history.
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore


def calculate_metrics(labels, predictions):
    """Calculate AUC, nDCG@5, nDCG@10, MRR."""
    evaluator = MetricEvaluator(
        labels=labels,
        predictions=predictions,
        metric_functions=[
            AucScore(),
            NdcgScore(k=5),
            NdcgScore(k=10),
            MrrScore(),
        ],
    )
    evaluator.evaluate()
    return evaluator.evaluations


def analyze_cold_start_performance(
    experiment_path: str,
    articles_path: str = "input/ebnerd_testset/ebnerd_testset/articles.parquet",
    cold_start_percentile: float = 0.1,
):
    """
    Analyze performance on cold-start items.
    
    Args:
        experiment_path: Path to experiment output (e.g., 'output/experiments/2025-12-21/small067_001')
        articles_path: Path to articles.parquet
        cold_start_percentile: Bottom percentile to consider as cold-start (default 0.1 = bottom 10%)
    """
    experiment_path = Path(experiment_path)
    articles_path = Path(articles_path)
    
    print("=" * 80)
    print(f"COLD-START ANALYSIS: Bottom {cold_start_percentile*100:.0f}% Items")
    print(f"Experiment: {experiment_path}")
    print("=" * 80)
    
    # Load articles and calculate popularity
    print("\n1. Loading articles and identifying cold-start items...")
    articles_df = pl.read_parquet(articles_path)
    
    # Use total_inviews as popularity metric
    if "total_inviews" not in articles_df.columns:
        print("ERROR: total_inviews column not found in articles.parquet")
        return
    
    # Calculate threshold for cold-start items
    popularity_threshold = articles_df["total_inviews"].quantile(cold_start_percentile)
    cold_start_articles = articles_df.filter(
        pl.col("total_inviews") <= popularity_threshold
    )["article_id"].to_list()
    
    print(f"   Total articles: {len(articles_df)}")
    print(f"   Popularity threshold (inviews): {popularity_threshold:.1f}")
    print(f"   Cold-start articles: {len(cold_start_articles)} ({len(cold_start_articles)/len(articles_df)*100:.1f}%)")
    
    # Load test results
    print("\n2. Loading test results...")
    test_result_path = experiment_path / "test_result.parquet"
    if not test_result_path.exists():
        print(f"ERROR: test_result.parquet not found at {test_result_path}")
        return
    
    test_result_df = pl.read_parquet(test_result_path)
    print(f"   Total test impressions: {len(test_result_df)}")
    
    # Load original test data to get labels
    print("\n3. Loading test dataset with labels...")
    # We need to reconstruct the test split
    size_name = "small"
    dataset_path = Path("output/preprocess/dataset067")
    
    import random
    random.seed(42)
    
    full_validation_df = pl.read_parquet(
        str(dataset_path / size_name / "validation_dataset.parquet")
    )
    
    all_validation_impression_ids = sorted(
        full_validation_df["impression_id"].unique().to_list()
    )
    
    # Apply sampling rate if needed (0.1)
    sampling_rate = 0.1
    if sampling_rate:
        all_validation_impression_ids = random.sample(
            all_validation_impression_ids,
            int(len(all_validation_impression_ids) * sampling_rate),
        )
    
    split_idx = len(all_validation_impression_ids) // 2
    test_impression_ids = all_validation_impression_ids[split_idx:]
    
    test_df = full_validation_df.filter(
        pl.col("impression_id").is_in(test_impression_ids)
    )
    
    print(f"   Test dataset rows: {len(test_df)}")
    
    # Ensure test_df and test_result_df are aligned
    print("\n   Aligning test data and results...")
    test_impression_ids_set = set(test_df["impression_id"].unique().to_list())
    result_impression_ids_set = set(test_result_df["impression_id"].unique().to_list())
    
    # Only keep impressions that exist in both
    common_impression_ids = list(test_impression_ids_set & result_impression_ids_set)
    print(f"   Common impressions: {len(common_impression_ids)}")
    
    if len(common_impression_ids) == 0:
        print("ERROR: No common impression IDs between test data and results!")
        return
    
    # Filter both to common impressions
    test_df = test_df.filter(pl.col("impression_id").is_in(common_impression_ids))
    test_result_df = test_result_df.filter(pl.col("impression_id").is_in(common_impression_ids))
    
    print(f"   Aligned test rows: {len(test_df)}")
    print(f"   Aligned result rows: {len(test_result_df)}")
    
    # Calculate metrics on ALL items first
    print("\n4. Calculating metrics on ALL test items...")
    all_labels = (
        test_df.select(["impression_id", "user_id", "label"])
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("label").cast(pl.Int32))["label"]
        .to_list()
    )
    all_predictions = test_result_df.with_columns(
        pl.col("rank").list.eval(1 / pl.element())
    )["rank"].to_list()
    
    all_metrics = calculate_metrics(all_labels, all_predictions)
    
    print("\n   ALL ITEMS:")
    print(f"   AUC       : {all_metrics['auc']:.6f}")
    print(f"   nDCG@5    : {all_metrics['ndcg@5']:.6f}")
    print(f"   nDCG@10   : {all_metrics['ndcg@10']:.6f}")
    print(f"   MRR       : {all_metrics['mrr']:.6f}")
    
    # Filter to impressions that contain at least one cold-start item
    print(f"\n5. Filtering to impressions with cold-start items...")
    
    # Mark cold-start items in test_df
    test_df = test_df.with_columns(
        pl.col("article_id").is_in(cold_start_articles).alias("is_cold_start")
    )
    
    # Find impressions that have at least one cold-start item
    impressions_with_cold_start = test_df.filter(
        pl.col("is_cold_start")
    )["impression_id"].unique().to_list()
    
    # Filter both datasets to these impressions
    cold_test_df = test_df.filter(
        pl.col("impression_id").is_in(impressions_with_cold_start)
    )
    cold_test_result_df = test_result_df.filter(
        pl.col("impression_id").is_in(impressions_with_cold_start)
    )
    
    # Count cold-start items in these impressions
    num_cold_items = cold_test_df.filter(pl.col("is_cold_start")).shape[0]
    num_total_items = cold_test_df.shape[0]
    
    print(f"   Impressions with cold-start items: {len(impressions_with_cold_start)}")
    
    if len(impressions_with_cold_start) == 0 or num_cold_items == 0:
        print(f"\n   WARNING: No cold-start items found in test set!")
        print(f"   Try increasing --percentile (e.g., 0.2 or 0.3)")
        print(f"   Current threshold: {popularity_threshold:.1f} inviews")
        print("\n" + "=" * 80)
        return None
    
    print(f"   Cold-start items in these impressions: {num_cold_items} ({num_cold_items/num_total_items*100:.1f}%)")
    print(f"   Total items in these impressions: {num_total_items}")
    
    # Calculate metrics on impressions containing cold-start items
    print(f"\n6. Calculating metrics on impressions WITH cold-start items...")
    cold_labels = (
        cold_test_df.select(["impression_id", "user_id", "label"])
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("label").cast(pl.Int32))["label"]
        .to_list()
    )
    cold_predictions = cold_test_result_df.with_columns(
        pl.col("rank").list.eval(1 / pl.element())
    )["rank"].to_list()
    
    cold_metrics = calculate_metrics(cold_labels, cold_predictions)
    
    print("\n   IMPRESSIONS WITH COLD-START ITEMS:")
    print(f"   AUC       : {cold_metrics['auc']:.6f}")
    print(f"   nDCG@5    : {cold_metrics['ndcg@5']:.6f}")
    print(f"   nDCG@10   : {cold_metrics['ndcg@10']:.6f}")
    print(f"   MRR       : {cold_metrics['mrr']:.6f}")
    
    # Calculate performance gap
    print("\n7. Performance Gap (All Items vs Impressions with Cold-Start):")
    print(f"   Δ AUC     : {all_metrics['auc'] - cold_metrics['auc']:.6f} ({(all_metrics['auc'] - cold_metrics['auc'])/all_metrics['auc']*100:.2f}%)")
    print(f"   Δ nDCG@10 : {all_metrics['ndcg@10'] - cold_metrics['ndcg@10']:.6f} ({(all_metrics['ndcg@10'] - cold_metrics['ndcg@10'])/all_metrics['ndcg@10']*100:.2f}%)")
    
    print("\n" + "=" * 80)
    
    return {
        "all": all_metrics,
        "cold_start": cold_metrics,
        "cold_start_impressions_ratio": len(impressions_with_cold_start) / len(test_impression_ids),
        "cold_start_items_ratio": num_cold_items / num_total_items,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cold-start performance")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment output (e.g., output/experiments/2025-12-21/small067_001)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.1,
        help="Bottom percentile for cold-start (default: 0.1 = bottom 10%%)",
    )
    
    args = parser.parse_args()
    
    analyze_cold_start_performance(
        experiment_path=args.experiment,
        cold_start_percentile=args.percentile,
    )
