"""Reproduce the selected Ensemble backtest from frozen historical predictions.

This entry point does not train or reload any model. It reads the delivered
historical Ensemble scores and sends them to the existing integer-share engine.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from execution_status import recorded_attempt


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "external_validation/ensemble_ranker/predictions/stock_rankings.csv"
PRICES = ROOT / "external_validation/shared_inputs/inputs/test40_with_prices.csv"
ENGINE_PATH = ROOT / "backtesting/external40/integer_backtest.py"
OUTPUT = ROOT / "reports/legacy_ensemble_backtest"

EXPECTED_PREDICTION_SHA256 = "d8e219d6749031117add13e13d8fbea15abecec87e190a0c72e92ec3d340d007"
EXPECTED_PRICE_SHA256 = "3ff2cbfd862ec7207157dc67675f5c4e28c5f7e9061ff8fa7fcfa24a134fdc39"
EXPECTED = {
    ("FullHistory", "Top5_NoLoss13w"): {
        "annualized_return": 0.1043902627962429,
        "sharpe": 0.5434211006497096,
        "max_drawdown": -0.24878123919912354,
        "final_capital": 149330.60888418998,
        "total_fees": 14520.836735869885,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engine():
    spec = importlib.util.spec_from_file_location("legacy_integer_backtest", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the integer-share backtest engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _execute(frozen=None, output=None, provenance=None):
    if not __debug__:
        raise ValueError('Run without -O; the historical trading engine uses assertions')
    destination = Path(output) if output is not None else OUTPUT
    if sha256(PREDICTIONS) != EXPECTED_PREDICTION_SHA256:
        raise ValueError("Frozen prediction file hash does not match the audited historical file")
    if sha256(PRICES) != EXPECTED_PRICE_SHA256:
        raise ValueError("Price input hash does not match the historical backtest input")

    if frozen is None:
        frozen = pd.read_csv(
            PREDICTIONS,
            usecols=["date", "ticker", "ensemble_ranking_score", "ensemble_predicted_rank"],
            parse_dates=["date"],
            float_precision="round_trip",
        )
    if len(frozen) != 13_517 or frozen.date.nunique() != 366 or frozen.ticker.nunique() != 40:
        raise ValueError("Unexpected frozen prediction coverage")
    if frozen.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker prediction keys")
    if not np.isfinite(frozen.ensemble_ranking_score).all():
        raise ValueError("Non-finite Ensemble score")

    prediction = frozen[frozen.date >= pd.Timestamp("2022-08-26")].copy()
    prediction = prediction.sort_values(
        ["date", "ensemble_ranking_score", "ticker"],
        ascending=[True, False, True],
        kind="stable",
    )
    replay_rank = prediction.groupby("date").cumcount().add(1).to_numpy()
    if not np.array_equal(replay_rank, prediction.ensemble_predicted_rank.to_numpy()):
        raise ValueError("Stored Ensemble ranks do not match the frozen scores")
    prediction = prediction[["date", "ticker", "ensemble_ranking_score"]].rename(
        columns={"ensemble_ranking_score": "prediction"}
    )
    if len(prediction) != 8_354 or prediction.date.nunique() != 210:
        raise ValueError("Unexpected Common210 prediction coverage")

    prices = pd.read_csv(PRICES, parse_dates=["date"], float_precision="round_trip")
    aligned = prediction.merge(prices[['date', 'ticker']], on=['date', 'ticker'], validate='one_to_one')
    if len(aligned) != len(prediction):
        raise ValueError('Prediction rows are missing execution prices')
    engine = load_engine()
    _, _, groups, calendar, context = engine.prepare(
        raw_frame=prices, prediction_frame=prediction
    )
    windows = engine.windows(calendar)
    if len(calendar) != 210:
        raise ValueError("Unexpected historical backtest calendar")

    summaries = []
    weekly_outputs = []
    maximum_error = 0.0
    accounting_checks = []
    from pipeline.trading import independent_case
    from types import SimpleNamespace
    selected_cases = [("FullHistory", "Top5_NoLoss13w")]
    for window, strategy in selected_cases:
        weekly, orders, holdings = engine.execute(
            windows[window], groups, context, strategy, 100_000.0, 10, detail=True
        )
        metrics = engine.performance(weekly, 100_000.0)
        accounting_checks.append(independent_case(
            SimpleNamespace(capital=100_000.0, cost_bps=10, strategy=strategy, **metrics),
            weekly, orders, holdings, prices, prediction,
        ))
        summaries.append(
            {
                "model": "Ensemble_frozen_predictions",
                "scope": "Common210",
                "window": window,
                "strategy": strategy,
                "capital": 100_000.0,
                "cost_bps": 10,
                **metrics,
            }
        )
        weekly_outputs.append(
            weekly.assign(
                model="Ensemble_frozen_predictions",
                scope="Common210",
                window=window,
                strategy=strategy,
                capital=100_000.0,
                cost_bps=10,
            )
        )
        for field, expected in EXPECTED[(window, strategy)].items():
            error = abs(float(metrics[field]) - expected)
            maximum_error = max(maximum_error, error)
            if error > 1e-8:
                raise AssertionError(
                    f"Historical result mismatch for {window}/{strategy}/{field}: "
                    f"actual={metrics[field]!r}, expected={expected!r}"
                )

    destination.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination / "summary.csv", index=False, float_format="%.17g")
    weekly = pd.concat(weekly_outputs, ignore_index=True)
    weekly[[
        "model", "scope", "window", "strategy", "capital", "cost_bps",
        "date", "start_capital", "end_capital", "net_return", "cash",
        "cash_ratio", "fees", "holdings", "shares",
    ]].to_csv(destination / "weekly_capital.csv", index=False, float_format="%.17g")

    verification = {
        "status": "passed",
        "models_trained": 0,
        "prediction_source": str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
        "prediction_sha256": EXPECTED_PREDICTION_SHA256,
        "price_sha256": EXPECTED_PRICE_SHA256,
        "native_prediction_rows": 13_517,
        "native_prediction_weeks": 366,
        "common_rows": 8_354,
        "common_weeks": 210,
        "cases_verified": len(summary),
        "maximum_absolute_metric_error": maximum_error,
        "future_label_columns_used_for_trading": False,
        "independent_accounting_checks": sum(r['checks'] for r in accounting_checks),
        "maximum_accounting_error": max(r['maximum_absolute_error'] for r in accounting_checks),
        "provenance": provenance or {'prediction_mode': 'frozen prediction table'},
    }
    lines = [
        "# Historical Ensemble investment simulation",
        "",
        "Prediction source: " + (provenance or {}).get('prediction_mode', 'delivered frozen prediction table') + ". No models are trained in this simulation.",
        "",
        "Only the Ensemble Top5_NoLoss13w strategy is retained. It excludes stocks with a negative observable trailing 13-week return before selecting the five highest Ensemble ranks.",
        "",
        "Initial capital is 100,000, one-way transaction cost is 10bps, and the execution engine uses integer shares.",
        "",
        "| Window | Strategy | Weeks | Annualised return | Sharpe | Maximum drawdown | Final capital |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.window} | {row.strategy} | {row.weeks} | "
            f"{row.annualized_return:.2%} | {row.sharpe:.3f} | "
            f"{row.max_drawdown:.2%} | {row.final_capital:.2f} |"
        )
    lines += [
        "",
        "Only date, ticker and ensemble_ranking_score are passed to the trading engine. Future realised-return and evaluation-label columns in the source file are not used for selection or execution.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary[[
        "window", "strategy", "weeks", "annualized_return", "sharpe",
        "max_drawdown", "final_capital",
    ]].to_string(index=False))
    print("Verification passed; maximum absolute metric error:", maximum_error)
    return verification


def main(frozen=None, output=None, provenance=None) -> int:
    destination = Path(output) if output is not None else OUTPUT
    with recorded_attempt(destination, 'verification.json',
                          metadata={'mode': 'historical_ensemble_backtest'},
                          outputs=['summary.csv', 'weekly_capital.csv', 'README.md']) as status:
        status.update(_execute(frozen=frozen, output=destination, provenance=provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
