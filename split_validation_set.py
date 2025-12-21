"""
Split validation set into val/test using the golden rule:
- Sort by impression_time
- First 50% → validation set
- Last 50% (most recent) → test set
"""

import polars as pl
from pathlib import Path
import shutil

def split_validation_set(size: str = "medium"):
    """
    Apply golden rule to split validation data.
    
    Args:
        size: Dataset size - 'small' or 'medium'
    """
    print("=" * 80)
    print(f"SPLITTING {size.upper()} VALIDATION SET USING GOLDEN RULE")
    print("=" * 80)
    
    # Define paths
    input_base = Path(f"input/ebnerd_{size}")
    validation_dir = input_base / "validation"
    
    output_val_dir = Path(f"{size}_val")
    output_test_dir = Path(f"{size}_test")
    
    # Create output directories
    output_val_dir.mkdir(exist_ok=True)
    output_test_dir.mkdir(exist_ok=True)
    
    print(f"\nInput: {validation_dir}")
    print(f"Output VAL: {output_val_dir}")
    print(f"Output TEST: {output_test_dir}")
    
    # Load behaviors
    print("\n1. Loading behaviors...")
    behaviors_path = validation_dir / "behaviors.parquet"
    behaviors_df = pl.read_parquet(behaviors_path)
    print(f"   Total impressions: {len(behaviors_df)}")
    
    # Sort by time
    print("\n2. Sorting by impression_time...")
    behaviors_df = behaviors_df.sort("impression_time")
    
    # Split 50/50
    total_impressions = len(behaviors_df)
    split_idx = total_impressions // 2
    
    val_behaviors = behaviors_df[:split_idx]
    test_behaviors = behaviors_df[split_idx:]
    
    print(f"   Validation impressions (first 50%): {len(val_behaviors)}")
    print(f"   Test impressions (last 50%): {len(test_behaviors)}")
    
    # Get unique user IDs for each split
    val_users = set(val_behaviors["user_id"].unique().to_list())
    test_users = set(test_behaviors["user_id"].unique().to_list())
    
    print(f"\n   Unique users in validation: {len(val_users)}")
    print(f"   Unique users in test: {len(test_users)}")
    print(f"   Overlapping users: {len(val_users & test_users)}")
    
    # Save behaviors
    print("\n3. Saving behaviors...")
    val_behaviors.write_parquet(output_val_dir / "behaviors.parquet")
    test_behaviors.write_parquet(output_test_dir / "behaviors.parquet")
    print("   ✓ Behaviors saved")
    
    # Load and split history
    print("\n4. Processing history...")
    history_path = validation_dir / "history.parquet"
    if history_path.exists():
        history_df = pl.read_parquet(history_path)
        print(f"   Total history entries: {len(history_df)}")
        
        # Filter history for each split's users
        val_history = history_df.filter(pl.col("user_id").is_in(val_users))
        test_history = history_df.filter(pl.col("user_id").is_in(test_users))
        
        print(f"   Validation history entries: {len(val_history)}")
        print(f"   Test history entries: {len(test_history)}")
        
        val_history.write_parquet(output_val_dir / "history.parquet")
        test_history.write_parquet(output_test_dir / "history.parquet")
        print("   ✓ History saved")
    else:
        print("   ⚠ history.parquet not found, skipping")
    
    # Copy articles (same for both splits)
    print("\n5. Copying articles...")
    articles_sources = [
        validation_dir / "articles.parquet",
        input_base / "articles.parquet",
    ]
    
    articles_path = None
    for source in articles_sources:
        if source.exists():
            articles_path = source
            break
    
    if articles_path:
        shutil.copy2(articles_path, output_val_dir / "articles.parquet")
        shutil.copy2(articles_path, output_test_dir / "articles.parquet")
        print(f"   ✓ Copied from {articles_path}")
    else:
        print("   ⚠ articles.parquet not found")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SPLIT COMPLETE")
    print("=" * 80)
    print(f"\n{output_val_dir}/")
    print(f"  - behaviors.parquet ({len(val_behaviors):,} impressions)")
    if history_path.exists():
        print(f"  - history.parquet ({len(val_history):,} entries)")
    if articles_path:
        print(f"  - articles.parquet")
    
    print(f"\n{output_test_dir}/")
    print(f"  - behaviors.parquet ({len(test_behaviors):,} impressions)")
    if history_path.exists():
        print(f"  - history.parquet ({len(test_history):,} entries)")
    if articles_path:
        print(f"  - articles.parquet")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split validation set using golden rule (50/50 time-based split)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=["small", "medium"],
        help="Dataset size (default: medium)",
    )
    
    args = parser.parse_args()
    
    split_validation_set(size=args.size)
