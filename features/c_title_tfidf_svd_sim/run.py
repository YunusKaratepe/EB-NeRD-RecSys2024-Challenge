import os
import sys
import gc
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from utils.data import get_data_dirs

PREFIX = "c"

USE_COLUMNS = [
    "title_count_svd_sim",
    "title_count_svd_rn",
]
TARGET_COL = "title"
n_components = 50


def process_df(cfg, article_embeddings, articles_df, history_df, candidate_df):
    # map
    article_id_map = dict(
        zip(
            articles_df["article_id"].to_list(),
            articles_df.with_row_index()["index"].to_list(),
        )
    )
    user_id_map = dict(
        zip(
            history_df["user_id"].to_list(),
            history_df.with_row_index()["index"].to_list(),
        )
    )
    map_history_df = (
        history_df.select(["user_id", "article_id_fixed"])
        .rename({"article_id_fixed": "article_id"})
        .with_columns(
            pl.col("article_id")
            .list.eval(pl.element().replace(article_id_map))
            .alias("new_article_id")
        )
    )

    # user embedding
    user_embeddings = []
    for new_article_id_list in tqdm(map_history_df["new_article_id"].to_list()):
        user_embeddings.append(article_embeddings[new_article_id_list].mean(axis=0))
    user_embeddings = normalize(np.array(user_embeddings), norm="l2")
    
    # Create mean embedding for users with no history (cold-start)
    mean_embedding = user_embeddings.mean(axis=0)

    # Map article and user IDs to indices
    # For users not in history, use a special index (len of user_embeddings)
    candidate_df = candidate_df.with_columns(
        [
            pl.col("article_id").replace(article_id_map).alias("article_rn"),
            pl.col("user_id").replace(user_id_map, default=-1).alias("user_rn"),
        ]
    )
    
    # Check for users without history
    users_without_history = (candidate_df["user_rn"] == -1).sum()
    if users_without_history > 0:
        print(f"Warning: {users_without_history} candidates have users without history (will use mean embedding)")

    # 要素積の合計を類似度とする（consin similarity）
    # Process in chunks to avoid huge array allocation
    print(f"{article_embeddings.shape=}, {user_embeddings.shape=}")
    total_rows = len(candidate_df)
    chunk_size = 1000000
    all_similarities = []
    
    print(f"Processing {total_rows} candidates in chunks of {chunk_size}")
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"Processing chunk {start_idx} to {end_idx}")
        
        chunk_article_rn = candidate_df["article_rn"][start_idx:end_idx].to_list()
        chunk_user_rn = candidate_df["user_rn"][start_idx:end_idx].to_list()
        
        # Handle users without history by using mean embedding
        chunk_user_embeddings = np.array([
            user_embeddings[idx] if idx >= 0 else mean_embedding
            for idx in chunk_user_rn
        ])
        
        chunk_similarity = np.asarray(
            (
                article_embeddings[chunk_article_rn]
                * chunk_user_embeddings
            ).sum(axis=1)
        ).flatten()
        
        all_similarities.append(chunk_similarity)
        
        del chunk_article_rn, chunk_user_rn, chunk_similarity, chunk_user_embeddings
        gc.collect()
    
    similarity = np.concatenate(all_similarities)
    del all_similarities
    gc.collect()

    df = (
        candidate_df.with_columns(
            pl.Series(name="title_count_svd_sim", values=similarity)
        )
        .with_columns(
            pl.col("title_count_svd_sim")
            .rank(descending=True)
            .over("user_rn")
            .alias("title_count_svd_rn")
        )
        .select(USE_COLUMNS)
    )

    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)

    print("make article embeddings")
    # Danish stopwords for proper text processing
    danish_stopwords = [
        'og', 'i', 'det', 'at', 'en', 'til', 'er', 'som', 'på', 'de',
        'med', 'han', 'af', 'for', 'ikke', 'der', 'var', 'mig', 'sig',
        'den', 'har', 'ham', 'hun', 'nu', 'over', 'da', 'fra', 'du',
        'ud', 'sin', 'dem', 'os', 'op', 'man', 'hans', 'hvor', 'eller',
        'hvad', 'skal', 'selv', 'her', 'alle', 'vil', 'blev', 'kunne',
        'ind', 'når', 'være', 'dog', 'noget', 'ville', 'jo', 'deres',
        'efter', 'ned', 'skulle', 'denne', 'end', 'dette', 'mit', 'også',
        'under', 'have', 'dig', 'anden', 'hende', 'mine', 'alt', 'meget',
        'sit', 'sine', 'vor', 'mod', 'disse', 'hvis', 'din', 'nogle',
        'hos', 'blive', 'mange', 'ad', 'bliver', 'hendes', 'været',
        'thi', 'jer', 'sådan'
    ]
    vectorizer = TfidfVectorizer(stop_words=danish_stopwords, min_df=2, max_df=0.8)
    article_matrix = vectorizer.fit_transform(articles_df[TARGET_COL].to_list())
    decomposer = TruncatedSVD(n_components=n_components)
    article_embeddings = normalize(decomposer.fit_transform(article_matrix), norm="l2")

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        candidate_df = pl.read_parquet(
            Path(cfg.dir.candidate_dir) / size_name / f"{data_name}_candidate.parquet"
        )

        df = process_df(cfg, article_embeddings, articles_df, history_df, candidate_df)

        df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
        print(f"df shape: {df.shape}, columns: {df.columns}")
        df.write_parquet(
            output_path / f"{data_name}_feat.parquet",
        )
        
        # Clear memory after each dataset
        del history_df, candidate_df, df
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
