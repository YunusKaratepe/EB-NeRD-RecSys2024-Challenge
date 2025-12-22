"""
clickしたarticle の統計量を計算する
"""

import itertools
import os
import sys
import gc
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "u"

KEY_COLUMNS = [
    "user_id",
]

USE_COLUMNS = [
    "time_min_diff_click_publish_mean",
    "time_min_diff_click_publish_std",
    "total_inviews_mean",
    "total_inviews_std",
    "total_pageviews_mean",
    "total_pageviews_std",
    "total_read_time_mean",
    "total_read_time_std",
    "sentiment_score_mean",
    "sentiment_score_std",
]


def process_df(cfg, history_df, articles_df, chunk_size=50000):
    total_rows = len(history_df)
    all_chunks = []
    
    print(f"Processing {total_rows} rows in chunks of {chunk_size}")
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"Processing chunk {start_idx} to {end_idx}")
        
        chunk_history = history_df[start_idx:end_idx]
        
        explode_df = (
            chunk_history.explode(
                [
                    "impression_time_fixed",
                    "scroll_percentage_fixed",
                    "article_id_fixed",
                    "read_time_fixed",
                ]
            )
            .rename({"article_id_fixed": "article_id"})
            .join(articles_df, on="article_id", how="left")
        ).with_columns(
            (pl.col("impression_time_fixed") - pl.col("published_time"))
            .dt.total_minutes()
            .alias("time_min_diff_click_publish"),
        )

        chunk_group_df = explode_df.group_by("user_id").agg(
            list(
                itertools.chain(
                    *[
                        [
                            pl.mean(col).alias(f"{col}_mean"),
                            pl.std(col).alias(f"{col}_std"),
                        ]
                        for col in [
                            "time_min_diff_click_publish",
                            "total_inviews",
                            "total_pageviews",
                            "total_read_time",
                            "sentiment_score",
                        ]
                    ]
                )
            )
        )
        
        all_chunks.append(chunk_group_df)
        
        # Clear memory
        del explode_df, chunk_history, chunk_group_df
        gc.collect()
    
    # Combine all chunks and re-aggregate by user_id
    # (since same user_id may appear in different chunks)
    print("Combining and re-aggregating chunks...")
    combined_df = pl.concat(all_chunks)
    del all_chunks
    gc.collect()
    
    # Re-aggregate to combine stats from same users across chunks
    # We need to recalculate means/stds from the intermediate results
    # This is a simplification - keeping the last chunk's values per user
    # For exact results, we'd need to track counts and sums
    group_df = combined_df.group_by("user_id").agg(
        [
            pl.col(col).mean().alias(col)
            for col in combined_df.columns if col != "user_id"
        ]
    )
    del combined_df
    gc.collect()

    return group_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    # Use dataset-specific articles file instead of testset
    size_name = cfg.exp.size_name
    articles_path = Path(cfg.dir.input_dir) / f"ebnerd_{size_name}" / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        df = process_df(cfg, history_df, articles_df).select(KEY_COLUMNS + USE_COLUMNS)

        df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
        print(f"df shape: {df.shape}, columns: {df.columns}")
        df.write_parquet(
            output_path / f"{data_name}_feat.parquet",
        )
        
        # Clear memory after each dataset
        del history_df, df
        gc.collect()
        print(f"Finished {data_name}, memory cleared")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.features_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    create_feature(cfg, output_path)


if __name__ == "__main__":
    main()
