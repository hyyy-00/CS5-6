# 06 Model evaluation and multi-strategy backtest

Evaluates the per-stock predictions of the four model families (Ridge, Random Forest, XGBoost, XGBRanker; 5/7/11/45 factors each) under one set of rules: predictive power (IC / RankIC / ICIR / RMSE) plus a weekly backtest of 15 portfolio strategies × 14 windows × one-way 0/10/25bps, followed by an independent verification. Corresponding report: `reports/model_evaluation/model_evaluation_backtest_report_6y26w.pdf`.

## Order of execution (from the project root)

```bash
# 1) Train the four models; per-stock predictions are written to research/predictions/
python research/02_ridge/train_ridge.py
python research/03_random_forest/train_random_forest.py
python research/04_xgboost/train_xgboost.py
python research/05_xgboost_ranker/train_xgboost_ranker.py
# 2) Evaluation and backtest
python research/06_evaluation_backtest/evaluate.py          # add --no-charts to skip the ~500 SVG charts
# 3) Independent verification (weekly holdings, drift, costs, all summary metrics, phase 0, consensus scores, prediction metrics)
python research/06_evaluation_backtest/verify.py
```

## Files

| File | Purpose |
|---|---|
| `strategies.py` | Rules of the 15 strategies, weight drift, turnover, costs; window slicing |
| `metrics.py` | IC/RankIC/ICIR, MAE/RMSE; net value, annualised return, maximum drawdown, volatility, Sharpe, win rate |
| `evaluate.py` | Main program: load predictions → verify labels → build the same-factor four-model consensus → backtest → report/dashboard |
| `verify.py` | Independent recomputation that does not call `evaluate.py` |
| `reference_code/` | Original code of the report chapter, for reference only (the original `run.py`, `native_replay.py` and other files it depends on were not preserved, so it cannot run on its own) |

`results/` preserves a Linux run dated 2026-10-23: `predictions/` holds the four models' per-stock predictions, alongside `report.md`, summaries and a historical `verification.json`. To evaluate those frozen predictions without fitting the four models again, run from the project root:

```bash
python research/06_evaluation_backtest/evaluate.py --predictions-dir research/06_evaluation_backtest/results/predictions --output runs/research_evaluation_from_frozen --no-charts
python research/06_evaluation_backtest/verify.py --output runs/research_evaluation_from_frozen
```

Use a new output directory each time and pass the same `--output` to both commands. This performs evaluation, backtesting and independent recomputation; it is not a no-compute package check and does not reproduce different original-platform predictions. The retained research verification record documents that research run; the selected external strategy has its own complete replay and numerical validation record in `docs/VALIDATION.md` from the project root.

Complete output of every run (`outputs/`, about 126 MB, created when it runs): `report.md`, `strategy_dashboard.html` (interactive dashboard), `strategy_comparison.csv` (all backtest metrics), `model_evaluation.csv` (prediction metrics), `weekly_backtest.csv`, `positions.csv`, `phase_sensitivity.csv`, `coverage.csv`, `source_metric_reconciliation.csv` (reconciliation with each training script's delivered table), `charts/*.svg`, `verification.json`.

## Method

- Stock return = next global week's adjusted close / current week's adjusted close − 1; the gross portfolio return is the weighted sum, and cash earns 0.
- Traded weight = sum of absolute differences between target weights and last week's weights after drift; net return = (1 − traded weight × one-way cost) × (1 + gross return) − 1.
- Every window starts from cash with net value 1; path-dependent strategies (buffer, 2/4-week rebalancing) are replayed in each window. The end is marked to market without forced liquidation.
- 14 windows: FullHistory, the last 13/26/52/104/156 weeks, each calendar year, Development6y, Validation, FinalTest.
- Consensus: equal-weight average of the four families' weekly percentiles for the same factor set (scores rounded to 12 decimals), using only stocks predicted by all four; Ridge with 45 factors failed the gate, so there is no 45-factor consensus.
- Annualised return = net value^(52 / weeks) − 1; Sharpe = mean weekly return × 52 / annualised volatility; ICIR is not annualised.

## Relationship to the original report

`evaluate.py` is a cleaned-up rewrite of the report-chapter code: strategies, windows, costs, label checks and consensus rules follow `reference_code/full_history_backtest.py`, `portfolio_strategies.py` and `evaluate_latest.py` line by line; the missing engine `run.py` is rebuilt from the formulas in `reference_code/verify_latest.py` and `verify_prediction_metrics.py`. Rerun with the Ridge predictions, the tables on pages 6–7 of the report match exactly; the 11,340 main backtests and 972 phase tests match the report's counts. XGBoost and XGBRanker use random subsampling, so a different platform gives different predictions (see the [frozen-model reproducibility rationale](../../README.md#why-the-trained-models-are-frozen)).

> Research limits: same-close signals and same-close entry are research assumptions; slippage, market impact and capacity are not modelled. The full history includes the tuning period, and the Final Test has been viewed in earlier rounds. Each model's "best Final Test strategy" is picked after the fact and carries multiple-comparison bias.
