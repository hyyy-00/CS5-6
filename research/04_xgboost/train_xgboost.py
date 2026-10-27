"""Self-contained XGBoost comparison for weekly factor research.

This single file runs the 5-factor, 7-factor, 7+4-macro-factor and 45-factor
experiments using a six-year rolling history and 26-week refits.
"""
from __future__ import annotations

import argparse
import csv
import itertools
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


def candidates():
    for depth, child, trees in itertools.product([2, 3], [5, 20], [50, 150, 300]):
        yield {"max_depth": depth, "min_child_weight": child, "n_estimators": trees}


def params(candidate):
    return {
        "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
        "n_jobs": 4, "random_state": 42, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0,
        "reg_lambda": 10, **candidate,
    }


def prepared(train, predict, features):
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    x_pred = predict[features].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train, x_pred = x_train.fillna(med), x_pred.fillna(med)
    lo, hi = x_train.quantile(0.01), x_train.quantile(0.99)
    return x_train.clip(lo, hi, axis=1).to_numpy(), x_pred.clip(lo, hi, axis=1).to_numpy()


def fit_predict(train, predict, features, candidate):
    if xgb is None:
        raise SystemExit(
            "Missing XGBoost. Install with: python -m pip install "
            "scikit-learn xgboost"
        )
    x_train, x_pred = prepared(train, predict, features)
    model = xgb.XGBRegressor(**params(candidate))
    model.fit(x_train, train[MODEL_TARGET].to_numpy(float))
    out = predict[[DATE, TICKER]].copy()
    out["y_true"] = predict[MODEL_TARGET].to_numpy(float)
    out["y_pred"] = model.predict(x_pred)
    return out


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


def run_rolling(panel, features, candidate, folds, rolling_years):
    parts = []
    for fold in folds:
        fit_date = fold["fit_date"]
        train = panel[(panel[DATE] >= fit_date - pd.DateOffset(years=rolling_years)) &
                      (panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date)]
        pred = panel[(panel[DATE] >= fit_date) & (panel[DATE] <= fold["end"])]
        out = fit_predict(train, pred, features, candidate)
        # A 26-week refit block can cross a stage boundary. Preserve the
        # row-level stage instead of assigning the whole block the stage of
        # its first week.
        out["stage"] = pred["stage"].to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def run_fixed(panel, features, candidate, stage_dates):
    parts = []
    splits = [("development", stage_dates["development"], stage_dates["validation"],
               stage_dates["development"]),
              ("validation", stage_dates["validation"], stage_dates["final_test"],
               stage_dates["validation"]),
              ("final_test", stage_dates["final_test"], None, stage_dates["final_test"])]
    for stage, start, end, cutoff in splits:
        train = panel[(panel[DATE] < cutoff) & (panel["label_end_date"] < cutoff)]
        pred = panel[panel[DATE] >= start]
        if end is not None:
            pred = pred[pred[DATE] < end]
        out = fit_predict(train, pred, features, candidate)
        out["stage"] = stage
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


RESULT_ROWS = []


def display_result(model_name, dataset, predictions):
    rows = []
    for stage in ["development", "validation", "final_test"]:
        part = predictions[predictions.stage == stage]
        if part.empty:
            continue
        row = {"Model Name": model_name, "Dataset": dataset, "Stage": stage}
        row.update(stage_metrics(part))
        rows.append(row)
    RESULT_ROWS.extend(rows)
    table = pd.DataFrame(rows)
    for col in ["IC", "Rank IC", "Annualized Return", "Top5 Excess"]:
        table[col] = table[col].map(pct)
    for col in ["ICIR", "Rank ICIR", "Information Ratio"]:
        table[col] = table[col].map(ratio)
    print("\nQlib-style and Top-5 results:")
    print(table[["Model Name", "Dataset", "Stage", "IC", "ICIR", "Rank IC",
                 "Rank ICIR", "Annualized Return", "Information Ratio", "Top5 Excess"]]
          .to_string(index=False))


def run_one(mode, dataset, features, panel, folds, rolling_years, stage_dates, predictions_path=None):
    print("\n" + "=" * 78)
    print(f"{mode.upper()} | {dataset} | {len(features)} factors")
    print("=" * 78)
    development_scores = []
    dev_predictions = {}
    candidates_list = list(candidates())
    for number, candidate in enumerate(candidates_list, 1):
        pred = (run_rolling(panel, features, candidate, folds, rolling_years)
                if mode == "rolling" else run_fixed(panel, features, candidate, stage_dates))
        dev = pred[pred.stage == "development"]
        metrics = stage_metrics(dev)
        development_scores.append({"candidate": number, **candidate, **metrics})
        dev_predictions[number] = pred
        print(f"Tuning {number}/12 complete", flush=True)
    scores = pd.DataFrame(development_scores)
    scores["passes_gate"] = ((scores["Rank IC"] > 0) &
                              (scores["Rank ICIR"] >= MIN_RANK_ICIR) &
                              (scores["Positive Rank IC"] >= MIN_POSITIVE_RANK_IC))
    eligible = scores[scores.passes_gate & scores["Top5 Excess"].notna()]
    if eligible.empty:
        print("No candidate passed the Development gate; no Validation or Final Test run.")
        return
    chosen = eligible.sort_values(["Top5 Excess", "Rank IC", "Rank ICIR"],
                                  ascending=[False, False, False]).iloc[0]
    candidate = candidates_list[int(chosen.candidate) - 1]
    print(f"Selected hyperparameters: candidate={int(chosen.candidate)}, {candidate}")
    print("Development gate: Rank IC > 0, Rank ICIR >= 0.05, positive Rank IC >= 50%; "
          "then highest Top-5 excess annualized return.")
    selected_pred = (run_rolling(panel, features, candidate, folds, rolling_years)
                     if mode == "rolling" else run_fixed(panel, features, candidate, stage_dates))
    display_result("XGBoost Rolling" if mode == "rolling" else "XGBoost Fixed",
                   dataset, selected_pred)
    if mode == "rolling" and predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        selected_pred.rename(columns={"y_pred": "prediction", "y_true": "target"})[[DATE, TICKER, "target", "prediction"]].to_csv(
            predictions_path, index=False, float_format="%.17g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mode", choices=["rolling", "fixed", "both"], default="rolling")
    parser.add_argument("--feature-set", choices=["all", "5", "7", "7_macro", "45"], default="all")
    parser.add_argument("--rolling-years", type=int, default=6,
                        help="Training history used by rolling mode (default: 6).")
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR / "outputs" / "xgboost_5_7_11_45_6y26w_results.csv")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS,
                        help="Per-stock rolling predictions of the selected candidate, read by 06_evaluation_backtest")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.rolling_years < 1:
        parser.error("--rolling-years must be at least 1")
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"CSV not found: {source}")
    groups = factor_sets(source)
    selected = list(groups) if args.feature_set == "all" else [args.feature_set]
    panel_cache = {}
    stage_dates_cache = {}
    for name in selected:
        panel = load_panel(source, groups[name])
        stage_dates = resolve_stage_dates(panel)
        panel = add_stages(panel, stage_dates)
        panel_cache[name] = panel
        stage_dates_cache[name] = stage_dates
    print(f"Input: {source}")
    print("Target: target (already forward-aligned log return)")
    print("Same data, six-year rolling history, 26-week refits and 12-candidate grid for every factor set.")
    print(f"Rolling training history: {args.rolling_years} years")
    first_dates = stage_dates_cache[selected[0]]
    print("Signal-aligned stages: "
          f"Development starts {first_dates['development'].date()}, "
          f"Validation starts {first_dates['validation'].date()}, "
          f"Final Test starts {first_dates['final_test'].date()}")
    if args.check_only:
        for name in selected:
            panel = panel_cache[name]
            print(f"{name}: {len(groups[name])} factors, {len(panel):,} rows, "
                  f"{panel[TICKER].nunique()} stocks")
        return
    for name in selected:
        panel = panel_cache[name]
        folds = rolling_folds(panel, args.rolling_years)
        modes = ["rolling", "fixed"] if args.mode == "both" else [args.mode]
        for mode in modes:
            run_one(mode, {"5": "5 factors", "7": "7 factors",
                           "7_macro": "7 factors + 4 macro factors",
                           "45": "45 factors"}[name], groups[name], panel, folds,
                     args.rolling_years, stage_dates_cache[name],
                     args.predictions_dir / f"XGB-R6-F{11 if name == '7_macro' else int(name)}.csv")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RESULT_ROWS).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Results CSV: {output}")


if __name__ == "__main__":
    main()
