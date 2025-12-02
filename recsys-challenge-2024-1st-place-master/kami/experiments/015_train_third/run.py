"""
Single-stage training:
- Train on train dataset only
- Validate on first 50% of validation dataset (for early stopping)
- Test on second 50% of validation dataset (held-out test set)
"""

import gc
import os
import pickle
import random
import sys
from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import utils
import wandb
from ebrec.evaluation import MetricEvaluator
from ebrec.evaluation import MultiprocessingAucScore as AucScore
from ebrec.evaluation import MultiprocessingMrrScore as MrrScore
from ebrec.evaluation import MultiprocessingNdcgScore as NdcgScore
from ebrec.utils._python import write_submission_file
from utils.logger import get_logger
from wandb.integration.lightgbm import wandb_callback, log_summary

pl.Config.set_ascii_tables(True)

logger = None

group_cols = ["impression_id", "user_id"]


def get_need_cols(cfg: DictConfig, cols: list) -> pl.DataFrame:
    need_cols = ["impression_id", "user_id", "article_id", "label"] + [
        col for col in cols if col not in cfg.exp.lgbm.unuse_cols
    ]
    if cfg.exp.article_stats_cols is False:
        need_cols = [
            col for col in need_cols if col not in cfg.leak_features.article_stats_cols
        ]
    if cfg.exp.past_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.past_impression_cols
        ]
    if cfg.exp.future_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.future_impression_cols
        ]
    return need_cols


def process_df(cfg: DictConfig, df: pl.DataFrame) -> pl.DataFrame:
    mul_cols_dict = cfg.exp.lgbm.mul_cols_dict
    if mul_cols_dict is not None and len(mul_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) * pl.col(cols[1])).alias(name)
                for name, cols in mul_cols_dict.items()
            ]
        )

    div_cols_dict = cfg.exp.lgbm.div_cols_dict
    if div_cols_dict is not None and len(div_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in div_cols_dict.items()
            ]
        )

    norm_cols_dict = cfg.exp.lgbm.norm_cols_dict
    if norm_cols_dict is not None and len(norm_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in norm_cols_dict.items()
            ]
        ).drop([cols[0] for cols in norm_cols_dict.values()])

    need_cols = get_need_cols(cfg, df.columns)
    df = df[need_cols]
    return df


def train_and_valid(
    cfg: DictConfig, train_df: pl.DataFrame, validation_df: pl.DataFrame
) -> lgb.Booster:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    logger.info(f"{len(feature_cols)=} {feature_cols=}")
    label_col = cfg.exp.lgbm.label_col

    if cfg.exp.lgbm.params.two_rounds:
        # bool特徴をintに変換
        bool_cols = [
            col
            for col in [label_col] + feature_cols
            if train_df[col].dtype == pl.Boolean
        ]
        train_df = train_df.with_columns(
            [train_df[col].cast(pl.Int8) for col in bool_cols]
        )
        validation_df = validation_df.with_columns(
            [validation_df[col].cast(pl.Int8) for col in bool_cols]
        )

        with utils.trace("write csv"):
            # テキストファイルへ書き出し
            train_df[[label_col] + feature_cols].write_csv(
                "tmp_train.csv", include_header=False
            )
            validation_df[[label_col] + feature_cols].write_csv(
                "tmp_validation.csv", include_header=False
            )

    print("make lgb.Dataset")
    lgb_train_dataset = lgb.Dataset(
        "tmp_train.csv"
        if cfg.exp.lgbm.params.two_rounds
        else train_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(train_df[label_col]),
        feature_name=feature_cols,
    )
    lgb_valid_dataset = lgb.Dataset(
        "tmp_validation.csv"
        if cfg.exp.lgbm.params.two_rounds
        else validation_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(validation_df[label_col]),
        feature_name=feature_cols,
        categorical_feature=cfg.exp.lgbm.cat_cols,
    )
    """
    if cfg.exp.lgbm.params.two_rounds:
        lgb_train_dataset.construct()
        lgb_valid_dataset.construct()
    """

    if cfg.exp.lgbm.params.objective == "lambdarank":
        print("make train group")
        train_group = (
            train_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("make validation group")
        valid_group = (
            validation_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("set group")
        lgb_train_dataset.set_group(train_group)
        lgb_valid_dataset.set_group(valid_group)
        cfg.exp.lgbm.params["ndcg_eval_at"] = cfg.exp.lgbm.ndcg_eval_at

    print("train")
    bst = lgb.train(
        OmegaConf.to_container(cfg.exp.lgbm.params, resolve=True),
        lgb_train_dataset,
        num_boost_round=cfg.exp.lgbm.num_boost_round,
        valid_sets=[lgb_valid_dataset],
        valid_names=["valid"],
        callbacks=[
            wandb_callback(),
            lgb.early_stopping(
                stopping_rounds=cfg.exp.lgbm.early_stopping_round,
                verbose=True,
                first_metric_only=cfg.exp.lgbm.params.first_metric_only,
            ),
            lgb.log_evaluation(cfg.exp.lgbm.verbose_eval),
        ],
    )
    # log_summary(bst, save_model_checkpoint=True)
    log_summary(bst, save_model_checkpoint=False)
    logger.info(
        f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
    )
    return bst


def predict(
    cfg: DictConfig, bst: lgb.Booster, test_df: pd.DataFrame, num_iteration: int
) -> pd.DataFrame:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in test_df.columns if col not in unuse_cols]
    # batch size で分割して予測
    batch_size = 100000
    y_pred = np.zeros(len(test_df))
    for i in tqdm(range(0, len(test_df), batch_size)):
        y_pred[i : i + batch_size] = bst.predict(
            test_df[feature_cols][i : i + batch_size], num_iteration=num_iteration
        )
    return y_pred


def save_model(cfg: DictConfig, bst: lgb.Booster, output_path: Path, name: int) -> None:
    with open(output_path / f"model_dict_{name}.pkl", "wb") as f:
        pickle.dump({"model": bst}, f)

    # save feature importance (top 100)
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = lgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=100)
    fig.tight_layout()
    fig.savefig(output_path / f"importance_{name}.png")
    plt.close(fig)

    # Get importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": bst.feature_name(),
            "importance": bst.feature_importance(importance_type="gain"),
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Save top 20 most important features
    fig_top, ax_top = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20).sort_values("importance", ascending=True)
    ax_top.barh(top_20["feature"], top_20["importance"])
    ax_top.set_xlabel("Importance (gain)")
    ax_top.set_title(f"Top 20 Most Important Features - {name}")
    fig_top.tight_layout()
    fig_top.savefig(output_path / f"importance_top20_{name}.png")
    plt.close(fig_top)
    
    # Save bottom 20 least important features
    fig_bottom, ax_bottom = plt.subplots(figsize=(10, 8))
    bottom_20 = importance_df.tail(20).sort_values("importance", ascending=True)
    ax_bottom.barh(bottom_20["feature"], bottom_20["importance"])
    ax_bottom.set_xlabel("Importance (gain)")
    ax_bottom.set_title(f"Bottom 20 Least Important Features - {name}")
    fig_bottom.tight_layout()
    fig_bottom.savefig(output_path / f"importance_bottom20_{name}.png")
    plt.close(fig_bottom)

    # Log importance to console
    pd.set_option("display.max_rows", None)
    logger.info(importance_df)
    logger.info(importance_df["feature"].to_list())


def make_result_df(df: pl.DataFrame, pred: np.ndarray):
    assert len(df) == len(pred)
    return (
        df.select(["impression_id", "user_id", "article_id"])
        .with_columns(pl.Series(name="pred", values=pred))
        .with_columns(
            pl.col("pred")
            .rank(method="ordinal", descending=True)
            .over(["impression_id", "user_id"])
            .alias("rank")
        )
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("rank"), pl.col("pred"))
        .select(["impression_id", "rank", "pred"])
    )


def main_stage(cfg: DictConfig, output_path) -> None:
    print("main_stage")
    dataset_path = Path(cfg.exp.dataset_path)

    size_name = cfg.exp.size_name
    if "train" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                print(f"{train_df.shape=}")
                random.seed(cfg.exp.seed)
                train_impression_ids = sorted(
                    train_df["impression_id"].unique().to_list()
                )
                use_train_impression_ids = random.sample(
                    train_impression_ids,
                    int(len(train_impression_ids) * cfg.exp.sampling_rate),
                )
                train_df = train_df.filter(
                    pl.col("impression_id").is_in(use_train_impression_ids)
                )
                print(f"{train_df.shape=}")
                gc.collect()
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            
            # Split validation into validation and test (50/50)
            print(f"Original validation shape: {full_validation_df.shape}")
            random.seed(cfg.exp.seed)
            all_validation_impression_ids = sorted(
                full_validation_df["impression_id"].unique().to_list()
            )
            
            if cfg.exp.sampling_rate:
                all_validation_impression_ids = random.sample(
                    all_validation_impression_ids,
                    int(len(all_validation_impression_ids) * cfg.exp.sampling_rate),
                )
            
            # Split 50/50
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            
            print(f"Validation split shape: {validation_df.shape}")
            print(f"Test split shape: {test_df.shape}")
            del full_validation_df
            gc.collect()

            train_df = process_df(cfg, train_df)
            gc.collect()
            validation_df = process_df(cfg, validation_df)
            gc.collect()
            test_df = process_df(cfg, test_df)
            gc.collect()

        with utils.trace("train and valid"):
            bst = train_and_valid(cfg, train_df, validation_df)
            save_model(cfg, bst, output_path, name="model")

        del train_df, validation_df
        gc.collect()

    if "predict" in cfg.exp.first_modes:
        bst = None
        with open(output_path / "model_dict_model.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict validation and test"):
            # Recreate the same split
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
            )
            
            random.seed(cfg.exp.seed)
            all_validation_impression_ids = sorted(
                full_validation_df["impression_id"].unique().to_list()
            )
            if cfg.exp.sampling_rate:
                all_validation_impression_ids = random.sample(
                    all_validation_impression_ids,
                    int(len(all_validation_impression_ids) * cfg.exp.sampling_rate),
                )
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            del full_validation_df
            
            validation_df = process_df(cfg, validation_df)
            y_valid_pred = predict(
                cfg, bst, validation_df, num_iteration=bst.best_iteration
            )
            validation_result_df = make_result_df(validation_df, y_valid_pred)
            validation_result_df.write_parquet(
                output_path / "validation_result.parquet"
            )
            print(f"Validation: {validation_result_df}")
            del validation_df, validation_result_df
            gc.collect()
            
            test_df = process_df(cfg, test_df)
            y_test_pred = predict(cfg, bst, test_df, num_iteration=bst.best_iteration)
            test_result_df = make_result_df(test_df, y_test_pred)
            test_result_df.write_parquet(output_path / "test_result.parquet")
            print(f"Test: {test_result_df}")
            del test_df, test_result_df
            gc.collect()

    if "eval" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            # Recreate the same split
            full_validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
            )
            
            random.seed(cfg.exp.seed)
            all_validation_impression_ids = sorted(
                full_validation_df["impression_id"].unique().to_list()
            )
            if cfg.exp.sampling_rate:
                all_validation_impression_ids = random.sample(
                    all_validation_impression_ids,
                    int(len(all_validation_impression_ids) * cfg.exp.sampling_rate),
                )
            split_idx = len(all_validation_impression_ids) // 2
            validation_impression_ids = all_validation_impression_ids[:split_idx]
            test_impression_ids = all_validation_impression_ids[split_idx:]
            
            validation_df = full_validation_df.filter(
                pl.col("impression_id").is_in(validation_impression_ids)
            )
            test_df = full_validation_df.filter(
                pl.col("impression_id").is_in(test_impression_ids)
            )
            del full_validation_df
            
            validation_df = process_df(cfg, validation_df)
            test_df = process_df(cfg, test_df)
            
            validation_result_df = pl.read_parquet(
                output_path / "validation_result.parquet",
            )
            test_result_df = pl.read_parquet(
                output_path / "test_result.parquet",
            )
            
        with utils.trace("prepare eval validation"):
            labels = (
                validation_df.select(["impression_id", "user_id", "label"])
                .group_by(["impression_id", "user_id"], maintain_order=True)
                .agg(pl.col("label").cast(pl.Int32))["label"]
                .to_list()
            )
            predictions = validation_result_df.with_columns(
                pl.col("rank").list.eval(1 / pl.element())
            )["rank"].to_list()

        with utils.trace("eval validation"):
            metric_functions = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]

            metrics = MetricEvaluator(
                labels=labels,
                predictions=predictions,
                metric_functions=metric_functions,
            )
            metrics.evaluate()
            val_result_dict = metrics.evaluations

            logger.info(f"VALIDATION: {val_result_dict}")
            wandb.log({"val_" + k: v for k, v in val_result_dict.items()})
        
        with utils.trace("prepare eval test"):
            test_labels = (
                test_df.select(["impression_id", "user_id", "label"])
                .group_by(["impression_id", "user_id"], maintain_order=True)
                .agg(pl.col("label").cast(pl.Int32))["label"]
                .to_list()
            )
            test_predictions = test_result_df.with_columns(
                pl.col("rank").list.eval(1 / pl.element())
            )["rank"].to_list()

        with utils.trace("eval test"):
            metric_functions = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]

            test_metrics = MetricEvaluator(
                labels=test_labels,
                predictions=test_predictions,
                metric_functions=metric_functions,
            )
            test_metrics.evaluate()
            test_result_dict = test_metrics.evaluations

            logger.info(f"TEST: {test_result_dict}")
            wandb.log({"test_" + k: v for k, v in test_result_dict.items()})


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")

    logger.info(f"exp_name: {exp_name}")
    logger.info(f"ouput_path: {output_path}")
    logger.info(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="recsys2024",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug or cfg.exp.size_name == "demo" else "online",
    )

    main_stage(cfg, output_path)


if __name__ == "__main__":
    main()
