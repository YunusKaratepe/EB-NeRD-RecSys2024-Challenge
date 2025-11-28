import os
import sys
import gc
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "a"

KEY_COLUMNS = [
    "article_id",
]

cat_cols = ["category"]
USE_COLUMNS = [f"{col}_click_ratio" for col in cat_cols]


def process_df(cfg, articles_df, history_df, chunk_size=50000):
    # Process history in chunks to reduce memory usage
    total_rows = len(history_df)
    all_counts = {cat_col: {} for cat_col in cat_cols}
    
    print(f"Processing {total_rows} rows in chunks of {chunk_size}")
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"Processing chunk {start_idx} to {end_idx}")
        
        chunk_df = history_df[start_idx:end_idx]
        
        explode_df = (
            chunk_df.select("article_id_fixed", "user_id")
            .explode(
                [
                    "article_id_fixed",
                ]
            )
            .with_columns(
                pl.col("article_id_fixed").alias("article_id"),
            )
        )

        # ユーザーごとに1回にする
        if cfg.exp.is_user_unique:
            explode_df = explode_df.unique(["user_id", "article_id"])

        explode_df = explode_df.join(
            articles_df.select(["article_id"] + cat_cols),
            on="article_id",
            how="left",
        )
        
        # Accumulate counts for each category
        for cat_col in cat_cols:
            chunk_counts = explode_df[cat_col].value_counts()
            for row in chunk_counts.iter_rows():
                cat_value = row[0]
                count = row[1]
                all_counts[cat_col][cat_value] = all_counts[cat_col].get(cat_value, 0) + count
        
        # Clear memory
        del explode_df, chunk_df

    base_df = articles_df.select(["article_id"] + cat_cols)

    # category, article_type, sentiment_label, premium ごとの click count の割合を求める
    for cat_col in cat_cols:
        # Convert accumulated counts to DataFrame
        count_data = list(all_counts[cat_col].items())
        count_df = pl.DataFrame({
            cat_col: [item[0] for item in count_data],
            "count": [item[1] for item in count_data]
        })
        
        count_df = (
            count_df
            .sort("count", descending=True)
            .with_columns(
                # ratioにする
                (pl.col("count") / pl.col("count").sum()).alias(
                    f"{cat_col}_click_ratio"
                )
            )
            .drop("count")
        )
        base_df = base_df.join(
            count_df,
            on=cat_col,
            how="left",
        )

    return base_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        df = process_df(cfg, articles_df, history_df).select(KEY_COLUMNS + USE_COLUMNS)

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
