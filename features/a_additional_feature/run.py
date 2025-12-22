import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import OrdinalEncoder

PREFIX = "a"

KEY_COLUMNS = [
    "article_id",
]

USE_COLUMNS = [
    "inviews_per_pageviews",
    "read_time_per_pageviews",
    "read_time_per_inviews",
    "subcategory_len",
    "image_ids_len",
]


def process_df(cfg, df):
    df = df.with_columns(
        [
            (pl.col("total_inviews") / pl.col("total_pageviews")).alias(
                "inviews_per_pageviews"
            ),
            (pl.col("total_read_time") / pl.col("total_pageviews")).alias(
                "read_time_per_pageviews"
            ),
            (pl.col("total_read_time") / pl.col("total_inviews")).alias(
                "read_time_per_inviews"
            ),
            pl.col("subcategory").list.len().alias("subcategory_len"),
            pl.col("image_ids").list.len().alias("image_ids_len"),
        ]
    )
    return df


def create_feature(cfg: DictConfig, output_path):
    # Use dataset-specific articles file instead of testset
    size_name = cfg.exp.size_name
    articles_path = Path(cfg.dir.input_dir) / f"ebnerd_{size_name}" / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    df = process_df(cfg, articles_df).select(KEY_COLUMNS + USE_COLUMNS)
    df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
    print(f"df shape: {df.shape}, columns: {df.columns}")

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        df.write_parquet(
            output_path / f"{data_name}_feat.parquet",
        )


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
