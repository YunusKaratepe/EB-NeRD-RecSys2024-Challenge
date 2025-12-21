"""
Create a medium-sized dataset (20% of train) by sampling the most recent impressions.
This allows training on large dataset features without running out of RAM.
"""

import polars as pl
from pathlib import Path
import shutil


def create_medium_dataset(
    source_dataset: str = "input/ebnerd_large",
    target_dataset: str = "input/ebnerd_medium",
    sample_ratio: float = 0.2,
):
    """
    Create a medium-sized dataset by sampling recent training data from raw ebnerd data.
    
    Args:
        source_dataset: Path to source raw dataset (default: input/ebnerd_large)
        target_dataset: Path to save medium dataset (default: input/ebnerd_medium)
        sample_ratio: Fraction of training data to keep (default: 0.2 = 20%)
    """
    source_path = Path(source_dataset)
    target_path = Path(target_dataset)
    
    print("=" * 80)
    print(f"CREATING MEDIUM DATASET ({sample_ratio*100:.0f}% of source)")
    print("=" * 80)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print()
    
    # Create target directory structure
    for subdir in ["train", "validation"]:
        (target_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # 1. Process training behaviors
    print("1. Loading and sampling training behaviors...")
    train_behaviors_path = source_path / "train" / "behaviors.parquet"
    if not train_behaviors_path.exists():
        print(f"ERROR: Training behaviors not found at {train_behaviors_path}")
        return
    
    train_behaviors = pl.read_parquet(train_behaviors_path)
    print(f"   Original size: {len(train_behaviors):,} impressions")
    
    # Sample most recent impressions (end of training period)
    train_behaviors = train_behaviors.sort("impression_time", descending=True)
    n_keep = int(len(train_behaviors) * sample_ratio)
    sampled_train = train_behaviors.head(n_keep)
    
    # 2. Process training history and filter behaviors to users with history
    print("\n2. Loading training history and filtering behaviors...")
    train_history_path = source_path / "train" / "history.parquet"
    if train_history_path.exists():
        train_history = pl.read_parquet(train_history_path)
        print(f"   Original history: {len(train_history):,} users")
        
        # Get users from sampled behaviors
        sampled_user_ids = sampled_train["user_id"].unique()
        
        # Filter history to sampled users
        sampled_history = train_history.filter(
            pl.col("user_id").is_in(sampled_user_ids)
        )
        
        # Filter behaviors to only users that have history
        users_with_history = sampled_history["user_id"].unique()
        sampled_train = sampled_train.filter(
            pl.col("user_id").is_in(users_with_history)
        )
        
        print(f"   Sampled history users: {len(sampled_history):,}")
        print(f"   Behaviors after filtering to users with history: {len(sampled_train):,} impressions")
        
        output_history = target_path / "train" / "history.parquet"
        sampled_history.write_parquet(output_history)
        print(f"   Saved: {output_history}")
    else:
        print("   WARNING: Training history not found")
        sampled_history = None
    
    # Save filtered training behaviors
    output_train_behaviors = target_path / "train" / "behaviors.parquet"
    sampled_train.write_parquet(output_train_behaviors)
    print(f"   Saved filtered behaviors: {output_train_behaviors}")
    
    # 3. Sample validation (get least recent to be closest to end of training period)
    print("\n3. Loading and sampling validation behaviors...")
    val_behaviors_path = source_path / "validation" / "behaviors.parquet"
    if val_behaviors_path.exists():
        val_behaviors = pl.read_parquet(val_behaviors_path)
        print(f"   Original validation size: {len(val_behaviors):,} impressions")
        
        # Sample LEAST recent (earliest) validation impressions (closest to training end)
        val_behaviors = val_behaviors.sort("impression_time", descending=False)  # Changed to ascending
        n_keep_val = int(len(val_behaviors) * sample_ratio)
        sampled_val = val_behaviors.head(n_keep_val)
        
        print(f"   Sampled validation size: {len(sampled_val):,} impressions ({sample_ratio*100:.0f}%)")
        print(f"   Validation time range: {sampled_val['impression_time'].min()} to {sampled_val['impression_time'].max()}")
        
        # Sample validation history for users in sampled validation
        val_history_path = source_path / "validation" / "history.parquet"
        if val_history_path.exists():
            val_history = pl.read_parquet(val_history_path)
            sampled_val_user_ids = sampled_val["user_id"].unique()
            sampled_val_history = val_history.filter(
                pl.col("user_id").is_in(sampled_val_user_ids)
            )
            
            # Filter validation behaviors to only users that have history
            users_with_val_history = sampled_val_history["user_id"].unique()
            sampled_val = sampled_val.filter(
                pl.col("user_id").is_in(users_with_val_history)
            )
            
            print(f"   Validation behaviors after filtering to users with history: {len(sampled_val):,} impressions")
            
            output_val_history = target_path / "validation" / "history.parquet"
            sampled_val_history.write_parquet(output_val_history)
            print(f"   Saved validation history for {len(sampled_val_history):,} users: {output_val_history}")
        
        # Save filtered validation behaviors
        output_val_behaviors = target_path / "validation" / "behaviors.parquet"
        sampled_val.write_parquet(output_val_behaviors)
        print(f"   Saved filtered behaviors: {output_val_behaviors}")
    else:
        print("   WARNING: Validation behaviors not found")
    
    # 4. Filter articles to only those appearing in sampled data
    print("\n4. Filtering articles to sampled impressions...")
    articles_path = source_path / "articles.parquet"
    if articles_path.exists():
        articles_df = pl.read_parquet(articles_path)
        print(f"   Original articles: {len(articles_df):,}")
        
        # Collect all article IDs from sampled train and validation
        train_article_ids = set()
        for article_list in sampled_train["article_ids_inview"].to_list():
            if article_list:
                train_article_ids.update(article_list)
        
        val_article_ids = set()
        if 'sampled_val' in locals():
            for article_list in sampled_val["article_ids_inview"].to_list():
                if article_list:
                    val_article_ids.update(article_list)
        
        # Also include articles from histories
        if 'sampled_history' in locals():
            for article_list in sampled_history["article_id_fixed"].to_list():
                if article_list:
                    train_article_ids.update(article_list)
        
        if 'sampled_val_history' in locals():
            for article_list in sampled_val_history["article_id_fixed"].to_list():
                if article_list:
                    val_article_ids.update(article_list)
        
        all_article_ids = train_article_ids | val_article_ids
        
        # Filter articles
        filtered_articles = articles_df.filter(
            pl.col("article_id").is_in(list(all_article_ids))
        )
        
        print(f"   Filtered articles: {len(filtered_articles):,} ({len(filtered_articles)/len(articles_df)*100:.1f}%)")
        
        target_articles = target_path / "articles.parquet"
        filtered_articles.write_parquet(target_articles)
        print(f"   Saved: {target_articles}")
    else:
        print("   WARNING: Articles not found")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original train: {len(train_behaviors):,} impressions")
    print(f"Medium train:   {len(sampled_train):,} impressions ({len(sampled_train)/len(train_behaviors)*100:.1f}%)")
    if 'val_behaviors' in locals():
        print(f"\nOriginal validation: {len(val_behaviors):,} impressions")
        print(f"Medium validation:   {len(sampled_val):,} impressions ({len(sampled_val)/len(val_behaviors)*100:.1f}%)")
    print(f"\nRaw dataset saved to: {target_path}")
    print("Next: Run preprocessing on this dataset to create features")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create medium-sized dataset from large raw data")
    parser.add_argument(
        "--source",
        type=str,
        default="input/ebnerd_large",
        help="Source raw dataset path (default: input/ebnerd_large)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="input/ebnerd_medium",
        help="Target dataset path (default: input/ebnerd_medium)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Sample ratio (default: 0.2 = 20%%)",
    )
    
    args = parser.parse_args()
    
    create_medium_dataset(
        source_dataset=args.source,
        target_dataset=args.target,
        sample_ratio=args.ratio,
    )
