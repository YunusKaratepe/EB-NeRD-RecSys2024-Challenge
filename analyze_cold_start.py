"""
Analyze model performance on cold-start (least popular) items.
This shows how well the model handles items with sparse interaction history.
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys
import json

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
    size: str = "small",
    cold_start_percentile: float = 0.1,
):
    """
    Analyze performance on cold-start items.
    
    Args:
        experiment_path: Path to experiment output (e.g., 'output/experiments/015_train_third/small067_001_seed7_...')
        size: Dataset size - 'small' or 'medium' (default: 'small')
        cold_start_percentile: Bottom percentile to consider as cold-start (default 0.1 = bottom 10%)
    """
    experiment_path = Path(experiment_path)
    
    # Determine articles path based on size
    if size == "small":
        articles_path = Path("input/ebnerd_small/articles.parquet")
    elif size == "medium":
        articles_path = Path("input/ebnerd_medium/articles.parquet")
    else:
        raise ValueError(f"Unknown size: {size}. Use 'small' or 'medium'.")
    
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
    # Load from the test_result to see which impressions were actually predicted
    result_impression_ids = test_result_df["impression_id"].unique().to_list()
    
    # Load behaviors to get impression timestamps
    print("   Loading behaviors to get timestamps...")
    if size == "small":
        behaviors_path = Path("input/ebnerd_small/validation/behaviors.parquet")
    elif size == "medium":
        behaviors_path = Path("input/ebnerd_medium/validation/behaviors.parquet")
    else:
        raise ValueError(f"Unknown size: {size}")
    
    behaviors_df = pl.read_parquet(behaviors_path).select(["impression_id", "impression_time"])
    
    # Load full validation dataset
    dataset_path = Path("output/preprocess/dataset067")
    full_validation_df = pl.read_parquet(
        str(dataset_path / size / "validation_dataset.parquet")
    )
    
    # Join with behaviors to get timestamps
    full_validation_df = full_validation_df.join(
        behaviors_df,
        on="impression_id",
        how="left"
    )
    
    # Apply golden rule: Sort by time, use most recent 50% as test set
    print("   Applying golden rule: sorting by time, taking most recent 50% as test set...")
    
    # Get unique impression IDs sorted by time
    sorted_impressions = (
        full_validation_df
        .select(["impression_id", "impression_time"])
        .unique(subset=["impression_id"], maintain_order=True)
        .sort("impression_time")
    )
    
    total_impressions = len(sorted_impressions)
    
    # Take most recent 50% as test set
    test_split_idx = total_impressions // 2
    test_impression_ids = sorted_impressions["impression_id"][test_split_idx:].to_list()
    
    print(f"   Total unique impressions in validation: {total_impressions}")
    print(f"   Test set impressions (most recent 50%): {len(test_impression_ids)}")
    
    # Filter to test split AND impressions in results
    test_df = full_validation_df.filter(
        pl.col("impression_id").is_in(test_impression_ids) &
        pl.col("impression_id").is_in(result_impression_ids)
    )
    
    print(f"   Test impressions (from results): {len(result_impression_ids)}")
    print(f"   Test dataset rows: {len(test_df)}")
    
    # Data is already aligned since we filtered test_df by result impression IDs
    if len(test_df) == 0:
        print("ERROR: No matching impression IDs found in validation dataset!")
        return
    
    # Filter to impressions that contain at least one cold-start item
    print(f"\n4. Filtering to impressions with cold-start items...")
    
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
    print(f"\n5. Calculating metrics on impressions WITH cold-start items...")
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
    
    print("\n   COLD-START PERFORMANCE:")
    print(f"   AUC       : {cold_metrics['auc']:.6f}")
    print(f"   nDCG@5    : {cold_metrics['ndcg@5']:.6f}")
    print(f"   nDCG@10   : {cold_metrics['ndcg@10']:.6f}")
    print(f"   MRR       : {cold_metrics['mrr']:.6f}")
    
    # Save results to experiment folder in human-readable format
    output_file = experiment_path / f"cold_start_analysis_p{int(cold_start_percentile*100)}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COLD-START ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Experiment Path: {experiment_path}\n")
        f.write(f"  Dataset Size: {size}\n")
        f.write(f"  Cold-Start Percentile: {cold_start_percentile*100:.0f}% (bottom)\n")
        f.write(f"  Popularity Threshold (inviews): {popularity_threshold:.1f}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Total Articles: {len(articles_df):,}\n")
        f.write(f"  Cold-Start Articles: {len(cold_start_articles):,} ({len(cold_start_articles)/len(articles_df)*100:.1f}%)\n")
        f.write(f"  Total Test Impressions: {len(result_impression_ids):,}\n")
        f.write(f"  Impressions with Cold-Start Items: {len(impressions_with_cold_start):,} ({len(impressions_with_cold_start)/len(result_impression_ids)*100:.1f}%)\n")
        f.write(f"  Cold-Start Items in These Impressions: {num_cold_items:,} ({num_cold_items/num_total_items*100:.1f}%)\n")
        f.write(f"  Total Items in These Impressions: {num_total_items:,}\n\n")
        
        f.write("COLD-START PERFORMANCE METRICS:\n")
        f.write(f"  AUC       : {cold_metrics['auc']:.6f}\n")
        f.write(f"  nDCG@5    : {cold_metrics['ndcg@5']:.6f}\n")
        f.write(f"  nDCG@10   : {cold_metrics['ndcg@10']:.6f}\n")
        f.write(f"  MRR       : {cold_metrics['mrr']:.6f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 80)
    
    return {
        "cold_start_percentile": cold_start_percentile,
        "popularity_threshold": float(popularity_threshold),
        "cold_start_articles_count": len(cold_start_articles),
        "cold_start_impressions_count": len(impressions_with_cold_start),
        "metrics": cold_metrics,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze cold-start performance for news recommendation experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment output (e.g., output/experiments/015_train_third/medium067_001_seed7_20251221_143045)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="Dataset size (default: small)",
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
        size=args.size,
        cold_start_percentile=args.percentile,
    )