"""Random Forest comparison under the common six-year, 26-week protocol.
Four factor sets are evaluated: 5, 7, 7 plus 4 macro, and 45 factors.
Parameters are selected in Development only, then frozen for Validation and
Final Test while model weights are refitted in chronological 26-week blocks.
"""
from __future__ import annotations

import argparse
import ast
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(p for p in SCRIPT_DIR.parents if (p / "data").is_dir())
DEFAULT_INPUT = ROOT / "data" / "test_45_factors_weekly.csv"
PREDICTIONS = ROOT / "research" / "predictions"
DATE, TICKER, TARGET = "date", "ticker", "target"
MODEL_TARGET = "_model_target"
TRAINING_YEARS = 6
REFIT_WEEKS = 26
DEV_START = pd.Timestamp("2022-09-01")
VALID_START = pd.Timestamp("2024-09-01")
TEST_START = pd.Timestamp("2025-09-01")
TOP_K = 5
MIN_RANK_ICIR = 0.05
MIN_POSITIVE_RANK_IC = 0.50

FIVE = [
    "revenue_growth_yoy", "operating_profitability", "cashflow_to_assets",
    "debt_to_assets", "asset_growth_yoy",
]
SEVEN = [
    "revenue_growth_yoy", "ret_120d", "cashflow_to_assets", "ret_1d",
    "ret_5d", "close_location_value", "body_range_ratio",
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
    header = pd.read_csv(source, nrows=0, encoding="utf-8-sig").columns.tolist()
    all_factors = [c for c in header if c not in NON_FACTORS]
    if len(all_factors) != 45:
        raise ValueError(f"Expected 45 factors, found {len(all_factors)}")
    required = set(FIVE + SEVEN + MACRO)
    missing = sorted(required - set(all_factors))
    if missing:
        raise ValueError(f"CSV does not contain required factors: {missing}")
    return {"5": FIVE, "7": SEVEN, "7_macro": SEVEN + MACRO, "45": all_factors}


def candidates():
    # Preserve the original RF grid and candidate numbering.
    for depth, leaf, features in itertools.product([4, 6, 8], [10, 20, 30], ["sqrt", 0.5]):
        yield {"n_estimators": 100, "max_depth": depth,
               "min_samples_leaf": leaf, "max_features": features}


def model_params(candidate):
    return {"random_state": 5703, "n_jobs": -1, **candidate}


def load_panel(source: Path, features: list[str]) -> pd.DataFrame:
    data = pd.read_csv(source, parse_dates=[DATE], encoding="utf-8-sig")
    required = {DATE, TICKER, TARGET, "adj_close", *features}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data[[DATE, TICKER]].isna().any().any():
        raise ValueError("Missing date/ticker values")
    if data.duplicated([DATE, TICKER]).any():
        raise ValueError("Duplicate date/ticker pairs")
    data = data.sort_values([TICKER, DATE]).reset_index(drop=True)
    next_date = data.groupby(TICKER, sort=False)[DATE].shift(-1)
    data["label_end_date"] = next_date
    data[MODEL_TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    gap = (next_date - data[DATE]).dt.days
    data["label_end_date"] = data["label_end_date"].where(gap.between(5, 14))
    for col in features + [TARGET, "adj_close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    usable = np.isfinite(data[MODEL_TARGET]) & data["label_end_date"].notna()
    usable &= data[features].notna().any(axis=1)
    panel = data.loc[usable].copy().reset_index(drop=True)
    if panel.empty:
        raise ValueError("No usable rows remain after target/date checks")
    return panel


def resolve_stage_dates(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    observed = pd.Series(panel[DATE].dropna().sort_values().unique())
    resolved = {}
    for name, anchor in [("development", DEV_START),
                         ("validation", VALID_START),
                         ("final_test", TEST_START)]:
        candidates_after = observed[observed >= anchor]
        if candidates_after.empty:
            raise ValueError(f"No observed weekly signal date on/after {anchor.date()} for {name}")
        resolved[name] = pd.Timestamp(candidates_after.iloc[0])
    if not (resolved["development"] < resolved["validation"] < resolved["final_test"]):
        raise ValueError(f"Stage dates are not strictly increasing: {resolved}")
    return resolved


def add_stages(panel: pd.DataFrame, stage_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    out = panel.copy()
    out["stage"] = np.select(
        [out[DATE] < stage_dates["development"],
         out[DATE] < stage_dates["validation"],
         out[DATE] < stage_dates["final_test"]],
        ["initial_history", "development", "validation"], default="final_test")
    return out


def rolling_folds(panel: pd.DataFrame, years: int, refit_weeks: int):
    first_fit = panel[DATE].min() + pd.DateOffset(years=years)
    eligible = panel.loc[panel[DATE] >= first_fit].copy()
    if eligible.empty:
        raise ValueError("No dates remain after the six-year initial training window")
    eligible["block"] = ((eligible[DATE] - first_fit).dt.days // (7 * refit_weeks)).astype(int)
    folds = []
    for _, part in eligible.groupby("block", sort=True):
        fit_date = part[DATE].min()
        train = panel[(panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date) &
                      (panel[DATE] >= fit_date - pd.DateOffset(years=years))]
        if not train.empty:
            folds.append({"fit_date": fit_date, "end": part[DATE].max(),
                          "stage": part["stage"].iloc[0]})
    return folds


def prepared(train, predict, features):
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    x_pred = predict[features].replace([np.inf, -np.inf], np.nan)
    med = x_train.median().fillna(0.0)
    x_train, x_pred = x_train.fillna(med), x_pred.fillna(med)
    lo, hi = x_train.quantile(0.01), x_train.quantile(0.99)
    return x_train.clip(lo, hi, axis=1).to_numpy(), x_pred.clip(lo, hi, axis=1).to_numpy()


def fit_predict(train, predict, features, candidate):
    x_train, x_pred = prepared(train, predict, features)
    model = RandomForestRegressor(**model_params(candidate))
    model.fit(x_train, train[MODEL_TARGET].to_numpy(float))
    out = predict[[DATE, TICKER, "label_end_date"]].copy()
    out["y_true"] = predict[MODEL_TARGET].to_numpy(float)
    out["y_pred"] = model.predict(x_pred)
    return out


def run_rolling(panel, features, candidate, folds, years):
    parts = []
    for fold in folds:
        fit_date = fold["fit_date"]
        train = panel[(panel[DATE] < fit_date) & (panel["label_end_date"] < fit_date) &
                      (panel[DATE] >= fit_date - pd.DateOffset(years=years))]
        pred = panel[(panel[DATE] >= fit_date) & (panel[DATE] <= fold["end"])]
        if pred.empty:
            continue
        out = fit_predict(train, pred, features, candidate)
        out["stage"] = pred["stage"].to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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
    return {"IC": ic.mean(), "ICIR": ir(ic), "Rank IC": ric.mean(),
            "Rank ICIR": ir(ric), "Annualized Return": annualize(top),
            "Information Ratio": ir(excess) * np.sqrt(52), "Top5 Excess": annualize(excess),
            "Positive Rank IC": (ric > 0).mean() if len(ric) else np.nan}


def select_candidate(panel, features, folds, years):
    # Only Development blocks are fitted during tuning. Trim a boundary-spanning
    # block so no Validation/Test predictions enter candidate evaluation.
    lock_date = resolve_stage_dates(panel)["validation"]
    dev_folds = []
    for fold in folds:
        dates = panel.loc[(panel[DATE] >= fold["fit_date"]) &
                          (panel[DATE] <= fold["end"]) &
                          (panel.stage == "development"), DATE]
        if not dates.empty:
            dev_folds.append({**fold, "end": dates.max()})
    scores = []
    candidates_list = list(candidates())
    for number, candidate in enumerate(candidates_list, 1):
        pred = run_rolling(panel, features, candidate, dev_folds, years)
        if pred.empty:
            raise ValueError("No Development predictions after six-year initial history")
        # Labels that are not yet available at selection time cannot tune parameters.
        dev = pred[(pred.stage == "development") & (pred.label_end_date < lock_date)]
        metrics = stage_metrics(dev)
        scores.append({"candidate": number, **candidate, **metrics,
                       "selection_weeks": dev[DATE].nunique()})
        print(f"      candidate{number}: {candidate}", flush=True)
        print("        " + " | ".join(f"{key}: {ratio(value)}" for key, value in metrics.items()), flush=True)
    scores = pd.DataFrame(scores)
    scores["passes_gate"] = ((scores["Rank IC"] > 0) &
                              (scores["Rank ICIR"] >= MIN_RANK_ICIR) &
                              (scores["Positive Rank IC"] >= MIN_POSITIVE_RANK_IC))
    eligible = scores[scores.passes_gate & np.isfinite(scores["Top5 Excess"])]
    if eligible.empty:
        return None, scores
    chosen = eligible.sort_values(["Top5 Excess", "Rank IC", "Rank ICIR", "candidate"],
                                  ascending=[False, False, False, True]).iloc[0]
    return candidates_list[int(chosen.candidate) - 1], scores


def print_summary(table):
    """Aligned console summary in narrow tables (display only; the CSV keeps full precision)."""
    if table.empty:
        return
    factor = table["Dataset"]
    hp = table.drop_duplicates("Dataset")[["Dataset", "status", "selected_params"]] if "selected_params" in table \
        else table.drop_duplicates("Dataset")[["Dataset", "status"]].assign(selected_params=np.nan)
    rows = []
    for item in hp.itertuples(index=False):
        p = ast.literal_eval(item.selected_params) if isinstance(item.selected_params, str) else {}
        rows.append({"Factor set": item.Dataset, "Status": item.status,
                     "n_estimators": p.get("n_estimators", "-"), "max_depth": p.get("max_depth", "-"),
                     "min_samples_leaf": p.get("min_samples_leaf", "-"), "max_features": p.get("max_features", "-")})
    print("\nSELECTED HYPERPARAMETERS (chosen in Development, frozen afterwards)")
    print(pd.DataFrame(rows).to_string(index=False))
    stage = table.get("stage", pd.Series("-", index=table.index))
    as_pct = lambda v: "n.a." if v is None or not np.isfinite(v) else f"{v:.2%}"
    first = pd.DataFrame({"Factor set": factor, "Stage": stage})
    second = first.copy()
    for col in ["IC", "Rank IC", "ICIR", "Rank ICIR"]:
        first[col] = table[col].map(ratio) if col in table else "n.a."
    first["Pos. Rank IC"] = table["Positive Rank IC"].map(as_pct) if "Positive Rank IC" in table else "n.a."
    for col, name in [("Annualized Return", "Ann. Return"), ("Top5 Excess", "Top5 Excess"),
                      ("Information Ratio", "Info Ratio")]:
        second[name] = table[col].map(ratio if col == "Information Ratio" else as_pct) if col in table else "n.a."
    print("\nSTAGE RESULTS: PREDICTION METRICS")
    print(first.to_string(index=False))
    print("\nSTAGE RESULTS: TOP-5 PORTFOLIO METRICS (before costs)")
    print(second.to_string(index=False))
    print("(Full-precision decimals are saved in the results CSV.)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR / "outputs" / "random_forest_5_7_11_45_6y26w_results.csv")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS,
                        help="Per-stock predictions of the selected hyperparameters, read by 06_evaluation_backtest")
    parser.add_argument("--feature-set", choices=["all", "5", "7", "7_macro", "45"], default="all")
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"CSV not found: {source}")
    groups = factor_sets(source)
    selected_groups = list(groups) if args.feature_set == "all" else [args.feature_set]
    print(f"Input: {source}")
    print("Experiment: Standard Random Forest, 6-year rolling history, refit every 26 weeks")
    print("Target: target (already forward-aligned weekly log return)")
    results = []
    all_scores = []
    for name in selected_groups:
        features = groups[name]
        panel = load_panel(source, features)
        stage_dates = resolve_stage_dates(panel)
        panel = add_stages(panel, stage_dates)
        folds = rolling_folds(panel, TRAINING_YEARS, REFIT_WEEKS)
        label = {"5": "5 factors", "7": "7 factors",
                 "7_macro": "7 factors + 4 macro factors", "45": "45 factors"}[name]
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"Signal-aligned stages: Development {stage_dates['development'].date()}, "
              f"Validation {stage_dates['validation'].date()}, Final Test {stage_dates['final_test'].date()}")
        print(f"Stocks: {panel[TICKER].nunique()} | Rows: {len(panel):,} | Rolling folds: {len(folds)}")
        selected, scores = select_candidate(panel, list(features), folds, TRAINING_YEARS)
        scores["dataset"] = label
        all_scores.append(scores.assign(combo=label))
        if selected is None:
            print("      No candidate passed the development gate; skipped.")
            results.append({"Dataset": label, "stage": "development", "status": "no_candidate_passed"})
            continue
        print(f"      selected hyperparameters: {selected}")
        pred = run_rolling(panel, list(features), selected, folds, TRAINING_YEARS)
        args.predictions_dir.mkdir(parents=True, exist_ok=True)
        n_factors = 11 if name == "7_macro" else int(name)
        pred.rename(columns={"y_pred": "prediction", "y_true": "target"})[[DATE, TICKER, "target", "prediction"]].to_csv(
            args.predictions_dir / f"RF-R6-F{n_factors}.csv", index=False, float_format="%.17g")
        for stage in ["development", "validation", "final_test"]:
            part = pred[pred.stage == stage]
            if part.empty:
                continue
            row = {"Dataset": label, "n_factors": len(features), "stage": stage,
                   "status": "selected", "selected_params": str(selected), **stage_metrics(part)}
            results.append(row)
        print("      stage results:")
        for stage in ["development", "validation", "final_test"]:
            part = pred[pred.stage == stage]
            if not part.empty:
                m = stage_metrics(part)
                print(f"        {stage} ({part[DATE].nunique()} weeks):")
                for key, value in m.items():
                    print(f"          {key}: {ratio(value)}")
    table = pd.DataFrame(results)
    print_summary(table)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    scores_output = output.with_name(output.stem + "_development_candidates.csv")
    pd.concat(all_scores, ignore_index=True).to_csv(scores_output, index=False, encoding="utf-8-sig")
    print(f"Results: {output}\nDevelopment candidates: {scores_output}")
    print("Completed. Hyperparameters were selected in Development only.")



if __name__ == "__main__":
    main()
