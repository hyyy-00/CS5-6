"""Final three-model XGBRanker walk-forward evaluation.

The final model uses within-week 0--4 relevance labels for training and the
continuous next-week log return for evaluation. It produces rankings and
IC/RankIC metrics only; it does not run a trading backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRanker


TOTAL_EXPECTED_WEEKS = 522
FORMAL_START_WEEK = 156
REFIT_INTERVAL_WEEKS = 26
RELEVANCE_BINS = 5
RANDOM_SEED = 5703
FOLD_BOUNDARIES = (156, 247, 338, 430, 522)

FIVE_FEATURES = (
    "revenue_growth_yoy",
    "operating_profitability",
    "cashflow_to_assets",
    "debt_to_assets",
    "asset_growth_yoy",
)
TWO_FEATURES = ("operating_profitability", "debt_to_assets")


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    features: tuple[str, ...]
    training_window_weeks: int
    params: dict[str, Any]

    @property
    def ranked_features(self) -> tuple[str, ...]:
        return tuple(f"csrank__{feature}" for feature in self.features)


COMMON_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "tree_method": "hist",
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "eval_metric": "ndcg@10",
}

COMPONENTS = (
    ComponentSpec(
        "model_1_w156_five_factors",
        FIVE_FEATURES,
        156,
        COMMON_PARAMS
        | {
            "max_depth": 2,
            "learning_rate": 0.025,
            "n_estimators": 360,
            "min_child_weight": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
            "gamma": 0.0,
        },
    ),
    ComponentSpec(
        "model_2_w52_two_factors",
        TWO_FEATURES,
        52,
        COMMON_PARAMS
        | {
            "max_depth": 3,
            "learning_rate": 0.04,
            "n_estimators": 240,
            "min_child_weight": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
            "gamma": 0.0,
        },
    ),
    ComponentSpec(
        "model_3_w52_five_factors_regularized",
        FIVE_FEATURES,
        52,
        COMMON_PARAMS
        | {
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 280,
            "min_child_weight": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "gamma": 0.05,
        },
    ),
)


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    raw_features = tuple(
        dict.fromkeys(feature for spec in COMPONENTS for feature in spec.features)
    )
    required = {"date", "ticker", "adj_close", "target", *raw_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("date+ticker must be unique")
    numeric = frame[["adj_close", "target", *raw_features]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Selected numeric columns contain NaN or infinity")

    dates = pd.Index(sorted(frame["date"].unique()))
    if len(dates) != TOTAL_EXPECTED_WEEKS:
        raise ValueError(
            f"Expected {TOTAL_EXPECTED_WEEKS} unique weeks, found {len(dates)}"
        )
    gaps = pd.Series(dates[1:] - dates[:-1]).dt.days
    if not gaps.between(3, 11).all():
        raise ValueError("Unique dates do not have the expected weekly spacing")

    week_map = {date: index for index, date in enumerate(dates)}
    frame["week_index"] = frame["date"].map(week_map).astype(int)
    frame["label_end_week"] = frame["week_index"] + 1
    frame = frame.sort_values(["week_index", "ticker"], kind="stable").reset_index(
        drop=True
    )

    by_ticker = frame.sort_values(["ticker", "week_index"], kind="stable")
    next_week = by_ticker.groupby("ticker", sort=False)["week_index"].shift(-1)
    next_adj_close = by_ticker.groupby("ticker", sort=False)["adj_close"].shift(-1)
    verifiable = next_week.eq(by_ticker["week_index"] + 1) & next_adj_close.notna()
    expected = np.log(
        next_adj_close[verifiable] / by_ticker.loc[verifiable, "adj_close"]
    )
    errors = (by_ticker.loc[verifiable, "target"] - expected).abs()
    if (errors > 1e-8).any():
        raise ValueError("target does not match the documented next-week log return")

    frame["target_verification_status"] = "supplied_upstream"
    verified_keys = pd.MultiIndex.from_frame(
        by_ticker.loc[verifiable, ["date", "ticker"]]
    )
    frame_keys = pd.MultiIndex.from_frame(frame[["date", "ticker"]])
    frame.loc[frame_keys.isin(verified_keys), "target_verification_status"] = (
        "verified_from_next_row"
    )

    grouped = frame.groupby("week_index", sort=False)
    for feature in raw_features:
        frame[f"csrank__{feature}"] = grouped[feature].rank(
            method="average", pct=True
        )

    audit = pd.DataFrame(
        [
            ["input_rows", int(len(frame)), "verified"],
            ["unique_weeks", int(frame["week_index"].nunique()), "verified"],
            ["unique_tickers", int(frame["ticker"].nunique()), "verified"],
            ["target_rows_verified", int(verifiable.sum()), "verified"],
            ["target_rows_supplied_upstream", int((~verifiable).sum()), "accepted"],
            ["maximum_verified_target_error", float(errors.max()), "verified"],
            ["target2_used", False, "verified"],
            ["training_relevance_bins", RELEVANCE_BINS, "configured"],
        ],
        columns=["check", "value", "status"],
    )
    return frame, audit


def relevance_labels(frame: pd.DataFrame) -> np.ndarray:
    grouped = frame.groupby("week_index", sort=False)["target"]
    ranks = grouped.rank(method="average", ascending=True)
    group_size = grouped.transform("count")
    labels = np.floor((ranks - 1.0) * RELEVANCE_BINS / group_size)
    labels = labels.clip(0, RELEVANCE_BINS - 1).astype(np.int32)
    if not labels.between(0, RELEVANCE_BINS - 1).all():
        raise AssertionError("Invalid relevance label")
    return labels.to_numpy(dtype=np.int32)


def fit_ranker(
    training: pd.DataFrame,
    spec: ComponentSpec,
    n_jobs: int,
) -> XGBRanker:
    ordered = training.sort_values(["week_index", "ticker"], kind="stable")
    groups = ordered.groupby("week_index", sort=False).size().to_numpy(dtype=np.uint32)
    model = XGBRanker(
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        verbosity=0,
        **spec.params,
    )
    model.fit(
        ordered[list(spec.ranked_features)].to_numpy(dtype=np.float32),
        relevance_labels(ordered),
        group=groups,
        verbose=False,
    )
    return model


def build_ensemble(component_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for number, spec in enumerate(COMPONENTS, start=1):
        source = component_predictions[spec.component_id]
        source = source.loc[source["week_index"].ge(FORMAL_START_WEEK)].copy()
        source = source[
            [
                "week_index",
                "date",
                "ticker",
                "target",
                "target_verification_status",
                "actual_rank",
                "model_id",
                "refit_week",
                "raw_ranking_score",
                "percentile_score",
                "predicted_rank",
            ]
        ].rename(
            columns={
                "model_id": f"model_{number}_id",
                "refit_week": f"model_{number}_refit_week",
                "raw_ranking_score": f"model_{number}_raw_score",
                "percentile_score": f"model_{number}_percentile",
                "predicted_rank": f"model_{number}_predicted_rank",
            }
        )
        if merged is None:
            merged = source
        else:
            source = source.drop(
                columns=["target", "target_verification_status", "actual_rank"]
            )
            merged = merged.merge(
                source,
                on=["week_index", "date", "ticker"],
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise AssertionError("No component predictions")

    refit_columns = [f"model_{number}_refit_week" for number in range(1, 4)]
    if not merged[refit_columns].nunique(axis=1).eq(1).all():
        raise AssertionError("Component refit schedules are not aligned")
    merged["refit_week"] = merged["model_1_refit_week"]
    merged["prediction_block"] = (
        (merged["refit_week"] - FORMAL_START_WEEK) // REFIT_INTERVAL_WEEKS + 1
    ).astype(int)
    merged["ensemble_model_id"] = merged["refit_week"].map(
        lambda value: f"ensemble_refit_{int(value):04d}"
    )
    percentile_columns = [f"model_{number}_percentile" for number in range(1, 4)]
    merged["ensemble_ranking_score"] = merged[percentile_columns].mean(axis=1)
    merged["ensemble_predicted_rank"] = merged.groupby("week_index", sort=False)[
        "ensemble_ranking_score"
    ].rank(method="first", ascending=False).astype(int)
    merged["rank_difference"] = merged["ensemble_predicted_rank"] - merged["actual_rank"]
    merged.insert(6, "training_label", "weekly_relevance_0_to_4")

    if sorted(merged["week_index"].unique()) != list(
        range(FORMAL_START_WEEK, TOTAL_EXPECTED_WEEKS)
    ):
        raise AssertionError("Formal ensemble prediction weeks are incomplete")
    if merged.duplicated(["date", "ticker"]).any():
        raise AssertionError("Formal ensemble contains duplicate predictions")
    return merged.sort_values(["week_index", "ticker"], kind="stable").reset_index(
        drop=True
    )

