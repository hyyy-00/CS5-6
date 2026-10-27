"""Development-only XGBoostRanker label and parameter search.

Keeps the six-calendar-year history, 26-week refits, four factor sets,
preprocessing, selection gates and eight metrics identical to the baseline.
Freezes all four configurations before later-stage evaluation.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(p for p in SCRIPT_DIR.parents if (p / "data").is_dir())
DEFAULT_INPUT = ROOT / "data" / "test_45_factors_weekly.csv"
PREDICTIONS = ROOT / "research" / "predictions"
DATE, TICKER, TARGET = "date", "ticker", "target"
MODEL_TARGET = "_model_target"
TRAINING_YEARS = 6
REFIT_WEEKS = 26
# These are calendar anchors only.  The actual stage boundaries are resolved
# to the first observed weekly signal date on or after each anchor.
DEV_START = pd.Timestamp("2022-09-01")
VALID_START = pd.Timestamp("2024-09-01")
TEST_START = pd.Timestamp("2025-09-01")
TOP_K = 5
MIN_RANK_ICIR = 0.05
MIN_POSITIVE_RANK_IC = 0.50
RELEVANCE_LEVELS = 5
STAGES = ["development", "validation", "final_test"]
METRIC_COLUMNS = ["IC", "ICIR", "Rank IC", "Rank ICIR", "Annualized Return",
                  "Information Ratio", "Top5 Excess", "Positive Rank IC"]
RESULT_COLUMNS = ["Model Name", "Dataset", "Stage", *METRIC_COLUMNS]

SEVEN = [
    "revenue_growth_yoy", "ret_120d", "cashflow_to_assets", "ret_1d",
    "ret_5d", "close_location_value", "body_range_ratio",
]
FIVE = [
    "revenue_growth_yoy", "operating_profitability", "cashflow_to_assets",
    "debt_to_assets", "asset_growth_yoy",
]
MACRO = [
    "macro_vix_level", "macro_vix_change_5obs",
    "macro_yield_spread_10y_2y", "macro_dgs10_change_5obs",
]
NON_FACTORS = {DATE, TICKER, "open", "high", "low", "close", "adj_close",
               "volume", "target", "target2"}



def pct(value):
    return "n.a." if value is None or not np.isfinite(value) else f"{value:.4%}"


def ratio(value):
    return "n.a." if value is None or not np.isfinite(value) else f"{value:.4f}"


def factor_sets(source: Path):
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    all_factors = [c for c in header if c not in NON_FACTORS]
    if len(all_factors) != 45:
        raise ValueError(f"Expected 45 factors, found {len(all_factors)}")
    if sorted(set(FIVE + SEVEN + MACRO) - set(all_factors)):
        raise ValueError("The CSV does not contain all selected or macro factors")
    return {"5": FIVE, "7": SEVEN, "7_macro": SEVEN + MACRO, "45": all_factors}


def load_panel(source: Path, features: list[str]) -> pd.DataFrame:
    data = pd.read_csv(source, parse_dates=[DATE], encoding="utf-8-sig")
    required = {DATE, TICKER, TARGET, "adj_close", *features}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = data.sort_values([TICKER, DATE]).reset_index(drop=True)
    next_date = data.groupby(TICKER, sort=False)[DATE].shift(-1)
    data["label_end_date"] = next_date
    data[MODEL_TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    gap = (next_date - data[DATE]).dt.days
    data["label_end_date"] = data["label_end_date"].where(gap.between(5, 14))
    for col in features + [TARGET, "adj_close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    usable = data[MODEL_TARGET].notna() & data["label_end_date"].notna()
    usable &= data[features].notna().any(axis=1)
    panel = data.loc[usable].copy().reset_index(drop=True)
    if panel.empty:
        raise ValueError("No usable rows remain after Target/date checks")
    return panel


def resolve_stage_dates(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Map calendar anchors to dates that actually exist in the weekly panel."""
    observed = pd.Series(panel[DATE].dropna().sort_values().unique())
    resolved = {}
    for name, anchor in [("development", DEV_START),
                         ("validation", VALID_START),
                         ("final_test", TEST_START)]:
        candidates = observed[observed >= anchor]
        if candidates.empty:
            raise ValueError(f"No observed weekly signal date on/after {anchor.date()} for {name}")
        resolved[name] = pd.Timestamp(candidates.iloc[0])
    if not (resolved["development"] < resolved["validation"] < resolved["final_test"]):
        raise ValueError(f"Resolved stage dates are not strictly increasing: {resolved}")
    return resolved


def prepared(train, predict, features):
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    x_pred = predict[features].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train, x_pred = x_train.fillna(med), x_pred.fillna(med)
    lo, hi = x_train.quantile(0.01), x_train.quantile(0.99)
    return x_train.clip(lo, hi, axis=1).to_numpy(), x_pred.clip(lo, hi, axis=1).to_numpy()


def rank_series(pred, method):
    values = {}
    for date, group in pred.groupby(DATE, sort=True):
        group = group[["y_pred", "y_true"]].dropna()
        if len(group) >= 2 and group.y_pred.nunique() > 1 and group.y_true.nunique() > 1:
            values[date] = group.y_pred.corr(group.y_true, method=method)
    return pd.Series(values, dtype=float)


def annualize(values):
    values = pd.Series(values).dropna().astype(float)
    return float(np.expm1(np.log1p(values).mean() * 52)) if len(values) and (values > -1).all() else np.nan


def stage_metrics(pred):
    ic, ric = rank_series(pred, "pearson"), rank_series(pred, "spearman")
    top, market = {}, {}
    for date, group in pred.groupby(DATE, sort=True):
        group = group.dropna(subset=["y_pred", "y_true"])
        if len(group) < TOP_K:
            continue
        actual = np.expm1(group.y_true)
        chosen = group.nlargest(TOP_K, "y_pred")
        top[date] = float(np.expm1(chosen.y_true).mean())
        market[date] = float(actual.mean())
    top, market = pd.Series(top), pd.Series(market)
    excess = top - market.reindex(top.index)
    def ir(series):
        series = series.dropna()
        return float(series.mean() / series.std(ddof=1)) if len(series) > 1 and series.std(ddof=1) > 0 else np.nan
    return {
        "IC": ic.mean(), "ICIR": ir(ic), "Rank IC": ric.mean(),
        "Rank ICIR": ir(ric), "Annualized Return": annualize(top),
        "Information Ratio": ir(excess) * np.sqrt(52),
        "Top5 Excess": annualize(excess),
        "Positive Rank IC": (ric > 0).mean() if len(ric) else np.nan,
    }


def rolling_folds(panel, rolling_years):
    # A rolling experiment must have a complete history window before its
    # first prediction.  With --rolling-years 6, predictions therefore start
    # six years after the first observation; the calendar validation/test
    # boundaries remain unchanged for comparability.
    first_fit = panel[DATE].min() + pd.DateOffset(years=rolling_years)
    eligible = panel.loc[panel[DATE] >= first_fit].copy()
    eligible["_refit_block"] = ((eligible[DATE] - first_fit).dt.days //
                                 (7 * REFIT_WEEKS)).astype(int)
    rows = []
    for _, part in eligible.groupby("_refit_block", sort=True):
        fit_date = part[DATE].min()
        train = panel[(panel[DATE] >= fit_date - pd.DateOffset(years=rolling_years)) &
                      (panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date)]
        if not train.empty:
            rows.append({"stage": part["stage"].iloc[0], "fit_date": fit_date,
                         "end": part[DATE].max()})
    return rows


def add_stages(panel, stage_dates):
    first_fit = stage_dates["development"]
    panel = panel.copy()
    panel["stage"] = np.select(
        [panel[DATE] < first_fit,
         panel[DATE] < stage_dates["validation"],
         panel[DATE] < stage_dates["final_test"]],
        ["initial_history", "development", "validation"], default="final_test")
    return panel


DATASET_NAMES = {"5": "5 factors", "7": "7 factors",
                 "7_macro": "7 factors + 4 macro factors", "45": "45 factors"}
LABEL_MODES = ["quintile", "decile", "dense_rank", "top5"]
SEARCH_VERSION = "development_only_36_v1"


def candidate_grid():
    """Fixed before searching: 12 exact baselines and 24 new configurations."""
    grid = []
    for depth, child, trees in itertools.product([2, 3], [5, 20], [50, 150, 300]):
        grid.append({"candidate_id": len(grid) + 1, "family": "baseline",
                     "label_mode": "quintile", "ndcg_exp_gain": True,
                     "max_depth": depth, "min_child_weight": child,
                     "n_estimators": trees, "reg_lambda": 10, "pair_topk": 5})
    # Vary only labels and model hyperparameters. The experiment protocol,
    # feature sets, preprocessing, refit dates and selection gate do not change.
    presets = [(2, 5, 10, 50, 5), (2, 1, 1, 100, 5),
               (3, 1, 1, 150, 5), (3, 0.1, 0.1, 150, 10),
               (4, 1, 5, 150, 10), (2, 5, 10, 300, 10)]
    for label in LABEL_MODES:
        for preset, (depth, child, reg, trees, topk) in enumerate(presets, 1):
            grid.append({"candidate_id": len(grid) + 1, "family": "optimized",
                         "preset": preset, "label_mode": label, "ndcg_exp_gain": False,
                         "max_depth": depth, "min_child_weight": child,
                         "n_estimators": trees, "reg_lambda": reg, "pair_topk": topk})
    return grid


def params(candidate):
    return {"objective": "rank:ndcg", "eval_metric": "ndcg@5", "tree_method": "hist",
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": candidate["pair_topk"],
            "ndcg_exp_gain": candidate["ndcg_exp_gain"],
            "n_jobs": 4, "random_state": 42, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0,
            **{key: candidate[key] for key in
               ["max_depth", "min_child_weight", "n_estimators", "reg_lambda"]}}


def relevance_labels(train, mode):
    """Train-window-only, nonnegative integer labels; equal returns stay tied."""
    grouped = train.groupby(DATE, sort=False)[MODEL_TARGET]
    if mode in ["quintile", "decile"]:
        levels = 5 if mode == "quintile" else 10
        labels = (np.ceil(grouped.rank(method="average", pct=True) * levels) - 1).clip(
            0, levels - 1)
    elif mode == "dense_rank":
        labels = grouped.rank(method="dense", ascending=True) - 1
    elif mode == "top5":
        labels = (TOP_K + 1 - grouped.rank(method="min", ascending=False)).clip(lower=0)
    else:
        raise ValueError(f"Unknown label mode: {mode}")
    return labels.to_numpy(dtype=np.int32)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")


def experiment_protocol():
    return {"training_years": TRAINING_YEARS, "refit_weeks": REFIT_WEEKS,
            "top_k": TOP_K, "selection_stage": "development",
            "selection_policy": "xgboost_105_signal_weeks",
            "min_rank_ic": 0, "min_rank_icir": MIN_RANK_ICIR,
            "min_positive_rank_ic": MIN_POSITIVE_RANK_IC,
            "selection_order": ["Top5 Excess", "Rank IC", "Rank ICIR", "candidate_id"],
            "metric_columns": METRIC_COLUMNS,
            "preprocessing": "training_median_and_1pct_99pct_winsorisation"}


def prepare_folds(panel, features, folds, stages):
    """Cache preprocessing once per refit; stages controls prediction rows only."""
    cache = []
    for fold in folds:
        fit_date = fold["fit_date"]
        predict = panel[(panel[DATE] >= fit_date) & (panel[DATE] <= fold["end"]) &
                        panel.stage.isin(stages)]
        if predict.empty:
            continue
        train = panel[(panel[DATE] >= fit_date - pd.DateOffset(years=TRAINING_YEARS)) &
                      (panel[DATE] < fit_date) & (panel.label_end_date < fit_date)]
        train = train.sort_values([DATE, TICKER], kind="stable")
        if train.empty or train.duplicated([DATE, TICKER]).any():
            raise ValueError("Training panel is empty or has duplicate date/ticker rows")
        x_train, x_pred = prepared(train, predict, features)
        labels = {mode: relevance_labels(train, mode) for mode in LABEL_MODES}
        out = predict[[DATE, TICKER, "stage", "label_end_date"]].copy()
        out["y_true"] = predict[MODEL_TARGET].to_numpy(float)
        out["fit_date"] = fit_date
        cache.append({"x_train": x_train, "x_pred": x_pred, "labels": labels,
                      "group": train.groupby(DATE, sort=True).size().to_numpy(dtype=np.uint32),
                      "prediction_template": out,
                      "fit_date": fit_date,
                      "train_signal_min": train[DATE].min(),
                      "train_signal_max": train[DATE].max(),
                      "train_label_end_max": train.label_end_date.max(),
                      "train_rows": len(train), "train_weeks": train[DATE].nunique()})
    if not cache:
        raise ValueError(f"No prediction folds for stages {stages}")
    return cache


def predict_candidate(cache, candidate):
    parts = []
    for fold in cache:
        model = xgb.XGBRanker(**params(candidate))
        # No eval_set and no early stopping: later-stage outcomes cannot
        # influence training or the predetermined number of trees.
        model.fit(fold["x_train"], fold["labels"][candidate["label_mode"]],
                  group=fold["group"])
        out = fold["prediction_template"].copy()
        out["y_pred"] = model.predict(fold["x_pred"])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def choose_candidate(scores):
    scores = scores.copy()
    scores["passes_gate"] = ((scores["Rank IC"] > 0) &
                              (scores["Rank ICIR"] >= MIN_RANK_ICIR) &
                              (scores["Positive Rank IC"] >= MIN_POSITIVE_RANK_IC))
    scores["selected"] = False
    eligible = scores[scores.passes_gate & np.isfinite(scores["Top5 Excess"])]
    if eligible.empty:
        return None, scores
    best = eligible.sort_values(["Top5 Excess", "Rank IC", "Rank ICIR", "candidate_id"],
                                ascending=[False, False, False, True]).iloc[0]
    selected_id = int(best.candidate_id)
    scores.loc[scores.candidate_id == selected_id, "selected"] = True
    return selected_id, scores


def load_experiment(source):
    groups = factor_sets(source)
    panels, schedules, stage_dates = {}, {}, {}
    for name, features in groups.items():
        panel = load_panel(source, features)
        stage_dates[name] = resolve_stage_dates(panel)
        panels[name] = add_stages(panel, stage_dates[name])
        schedules[name] = rolling_folds(panels[name], TRAINING_YEARS)
    return groups, panels, schedules, stage_dates


def tune(source, run_dir, groups, panels, schedules, stage_dates):
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new --run-dir: {run_dir}")
    run_dir.mkdir(parents=True)
    grid = candidate_grid()
    manifest = {"search_version": SEARCH_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": sha256_file(source), "script_sha256": sha256_file(__file__),
                "protocol": experiment_protocol(), "factor_sets": groups,
                "stage_dates": stage_dates, "candidates": grid,
                "candidate_count_per_factor_set": len(grid),
                "versions": {"python": sys.version, "xgboost": xgb.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__}}
    # Written before the first candidate is fitted; this is the complete budget.
    manifest_path = run_dir / "search_manifest.json"
    write_json(manifest_path, manifest)
    selections, all_scores, training_records = {}, [], []
    for name, features in groups.items():
        print(f"\nDEVELOPMENT SEARCH | {DATASET_NAMES[name]} | {len(grid)} candidates", flush=True)
        # Keep initial history and Development only. Validation/Test targets
        # never reach candidate fitting, prediction or candidate scoring.
        dev_panel = panels[name][panels[name][DATE] < stage_dates[name]["validation"]].copy()
        cache = prepare_folds(dev_panel, features, schedules[name], ["development"])
        for fold in cache:
            training_records.append({"feature_set": name, **{k: v for k, v in fold.items()
                                     if k.startswith("train_") or k == "fit_date"}})
        records, candidate_predictions = [], {}
        for candidate in grid:
            pred = predict_candidate(cache, candidate)
            if set(pred.stage) != {"development"}:
                raise AssertionError("Non-Development predictions reached the search")
            metric = stage_metrics(pred)
            records.append({"feature_set": name, "Dataset": DATASET_NAMES[name],
                            **candidate, **metric, "selection_weeks": pred[DATE].nunique(),
                            "selection_rows": len(pred)})
            # Keep every Development prediction for independent audit.
            saved = pred.copy()
            saved.insert(0, "candidate_id", candidate["candidate_id"])
            candidate_predictions[candidate["candidate_id"]] = saved
            print(f"  {candidate['candidate_id']:02d}/{len(grid)} {candidate['label_mode']} "
                  f"Rank IC={metric['Rank IC']:.5f} Rank ICIR={metric['Rank ICIR']:.4f} "
                  f"Top5 Excess={metric['Top5 Excess']:.4%}", flush=True)
        selected_id, scores = choose_candidate(pd.DataFrame(records))
        scores.to_csv(run_dir / f"development_candidates_{name}.csv", index=False, encoding="utf-8-sig")
        # Store float32 scores as exact float64 values so CSV roundtrips do
        # not perturb Pearson correlations during independent verification.
        saved = pd.concat(candidate_predictions.values(), ignore_index=True)
        saved["y_pred"] = saved["y_pred"].astype(np.float64)
        saved.to_csv(run_dir / f"development_predictions_{name}.csv.gz", index=False,
                     compression="gzip", float_format="%.17g")
        if selected_id is None:
            selections[name] = {"status": "no_candidate_passed", "candidate": None}
            print("  No candidate passed the unchanged Development gate.", flush=True)
        else:
            chosen = next(c for c in grid if c["candidate_id"] == selected_id)
            metrics = scores.loc[scores.candidate_id == selected_id, METRIC_COLUMNS].iloc[0].to_dict()
            selections[name] = {"status": "selected", "candidate": chosen,
                                "development_metrics": metrics,
                                "selection_weeks": int(scores.selection_weeks.iloc[0]),
                                "selection_rows": int(scores.selection_rows.iloc[0])}
            print(f"  Frozen candidate {selected_id}: {chosen}", flush=True)
        all_scores.append(scores)
    pd.concat(all_scores, ignore_index=True).to_csv(run_dir / "development_candidates_all.csv",
                                                   index=False, encoding="utf-8-sig")
    write_json(run_dir / "development_training_windows.json", training_records)
    frozen = {"search_version": SEARCH_VERSION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": sha256_file(manifest_path),
              "input_sha256": sha256_file(source), "script_sha256": sha256_file(__file__),
              "selections": selections,
              "candidate_scores_sha256": sha256_file(run_dir / "development_candidates_all.csv")}
    write_json(run_dir / "frozen_selection.json", frozen)
    digest = sha256_file(run_dir / "frozen_selection.json")
    (run_dir / "frozen_selection.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"\nAll four factor sets frozen BEFORE later-stage evaluation. SHA256: {digest}", flush=True)
    return frozen


def read_frozen(source, run_dir, groups):
    frozen_path = run_dir / "frozen_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected = (run_dir / "frozen_selection.sha256").read_text().strip()
    if sha256_file(frozen_path) != expected:
        raise ValueError("Frozen selection hash changed")
    for key, path in [("input_sha256", source), ("script_sha256", __file__),
                      ("manifest_sha256", run_dir / "search_manifest.json"),
                      ("candidate_scores_sha256", run_dir / "development_candidates_all.csv")]:
        if frozen[key] != sha256_file(path):
            raise ValueError(f"Frozen search provenance changed: {key}")
    manifest = json.loads((run_dir / "search_manifest.json").read_text(encoding="utf-8"))
    if manifest["candidates"] != candidate_grid() or manifest["protocol"] != experiment_protocol():
        raise ValueError("The fixed candidate search or protocol changed")
    if manifest["factor_sets"] != groups or set(frozen["selections"]) != set(groups):
        raise ValueError("Frozen factor sets do not match current data")
    return frozen


def compare_results(baseline_path, optimized, frozen, destination):
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    rows = []
    for name, dataset in DATASET_NAMES.items():
        for stage in STAGES:
            old = baseline[(baseline.Dataset == dataset) & (baseline.Stage == stage)]
            new = optimized[(optimized.Dataset == dataset) & (optimized.Stage == stage)]
            if len(old) > 1 or len(new) > 1:
                raise ValueError("Duplicate dataset/stage results")
            row = {"Dataset": dataset, "Stage": stage,
                   "Baseline Status": "selected" if len(old) else "no_candidate_passed",
                   "Optimized Status": frozen["selections"][name]["status"]}
            for metric in METRIC_COLUMNS:
                before = float(old.iloc[0][metric]) if len(old) else np.nan
                after = float(new.iloc[0][metric]) if len(new) else np.nan
                row[f"Baseline {metric}"] = before
                row[f"Optimized {metric}"] = after
                row[f"Delta {metric}"] = after - before
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(destination, index=False, encoding="utf-8-sig")
    return comparison


def evaluate(source, baseline_path, run_dir, groups, panels, schedules, predictions_dir=None):
    frozen = read_frozen(source, run_dir, groups)
    output = run_dir / "xgboostranker_optimized_results.csv"
    if output.exists() or (run_dir / "evaluation_started.json").exists():
        raise FileExistsError("Evaluation was already started; preserve its frozen results")
    write_json(run_dir / "evaluation_started.json",
               {"started_utc": datetime.now(timezone.utc).isoformat(),
                "frozen_selection_sha256": sha256_file(run_dir / "frozen_selection.json"),
                "baseline_results_sha256": sha256_file(baseline_path)})
    rows, windows = [], []
    for name, selection in frozen["selections"].items():
        if selection["status"] != "selected":
            continue
        print(f"\nFROZEN EVALUATION | {DATASET_NAMES[name]} | candidate "
              f"{selection['candidate']['candidate_id']}", flush=True)
        cache = prepare_folds(panels[name], groups[name], schedules[name], STAGES)
        pred = predict_candidate(cache, selection["candidate"])
        for fold in cache:
            windows.append({"feature_set": name, **{k: v for k, v in fold.items()
                            if k.startswith("train_") or k == "fit_date"}})
        for stage in STAGES:
            part = pred[pred.stage == stage]
            metrics = stage_metrics(part)
            if stage == "development":
                np.testing.assert_allclose([metrics[k] for k in METRIC_COLUMNS],
                    [selection["development_metrics"][k] for k in METRIC_COLUMNS],
                    rtol=0, atol=1e-12, equal_nan=True)
            rows.append({"Model Name": "XGBoostRanker Optimized Rolling",
                         "Dataset": DATASET_NAMES[name], "Stage": stage, **metrics})
            print(f"  {stage}: Rank IC={metrics['Rank IC']:.4%}, "
                  f"Top5={metrics['Annualized Return']:.4%}, "
                  f"Top5 Excess={metrics['Top5 Excess']:.4%}", flush=True)
        pred["y_pred"] = pred["y_pred"].astype(np.float64)
        pred.to_csv(run_dir / f"selected_predictions_{name}.csv.gz", index=False,
                    compression="gzip", float_format="%.17g")
        if predictions_dir is not None:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            n_factors = 11 if name == "7_macro" else int(name)
            pred.rename(columns={"y_pred": "prediction", "y_true": "target"})[[DATE, TICKER, "target", "prediction"]].to_csv(
                predictions_dir / f"RANKER-R6-F{n_factors}.csv", index=False, float_format="%.17g")
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(run_dir / "evaluation_training_windows.json", windows)
    compare_results(baseline_path, result, frozen, run_dir / "comparison_vs_baseline.csv")
    print(f"\nResults: {output}\nComparison: {run_dir / 'comparison_vs_baseline.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline-results", type=Path, default=SCRIPT_DIR / "results" / "baseline_results.csv")
    parser.add_argument("--run-dir", type=Path, default=SCRIPT_DIR / "outputs" / "run")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS,
                        help="Per-stock predictions of each frozen configuration, read by 06_evaluation_backtest")
    parser.add_argument("--phase", choices=["tune", "evaluate", "all"], default="all")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if xgb is None or int(xgb.__version__.split(".")[0]) < 2:
        parser.error('Install numpy pandas scipy scikit-learn "xgboost>=2.0"')
    groups, panels, schedules, stage_dates = load_experiment(args.input)
    print(f"Protocol: 6 calendar years, 26-week refits, unchanged factors/metrics/gates; "
          f"{len(candidate_grid())} fixed candidates per factor set", flush=True)
    if args.check_only:
        for name, panel in panels.items():
            print(name, len(groups[name]), "factors", len(panel), "rows",
                  "stages", {s: int(panel.loc[panel.stage == s, DATE].nunique()) for s in STAGES},
                  "refit dates", [str(f['fit_date'].date()) for f in schedules[name]])
        return
    if args.phase in ["all", "evaluate"] and not args.baseline_results.is_file():
        parser.error(f"Baseline results not found: {args.baseline_results}")
    if args.phase in ["all", "tune"]:
        tune(args.input, args.run_dir, groups, panels, schedules, stage_dates)
    if args.phase in ["all", "evaluate"]:
        evaluate(args.input, args.baseline_results, args.run_dir, groups, panels, schedules, args.predictions_dir)


if __name__ == "__main__":
    main()
