# XGBoostRanker (optimised version)

The completed optimised Ranker: label schemes and hyperparameters are searched within Development only, frozen, and then evaluated on Validation / Final Test.

## Files

- `train_xgboost_ranker.py` (original name `train_xgboostranker_optimized.py`): full training, Development selection, frozen configuration and later evaluation.
- `results/xgboostranker_optimized_results.csv`: delivered results, four factor sets × three stages × eight metrics, 12 rows.
- `results/baseline_results.csv`: results of the Ranker before optimisation, used by the `evaluate` phase to build the before/after comparison; keep it when running the full workflow.
- Input data: `data/test_45_factors_weekly.csv` in the project root.

## Running (from the project root)

```bash
python research/05_xgboost_ranker/train_xgboost_ranker.py --check-only
python research/05_xgboost_ranker/train_xgboost_ranker.py --phase all                 # output to outputs/run by default
# or in two steps: select and freeze in Development first, then evaluate
python research/05_xgboost_ranker/train_xgboost_ranker.py --phase tune     --run-dir research/05_xgboost_ranker/outputs/run2
python research/05_xgboost_ranker/train_xgboost_ranker.py --phase evaluate --run-dir research/05_xgboost_ranker/outputs/run2
```

`--run-dir` must be a folder that does not exist yet; `evaluate` checks the frozen records, input and script hashes of the same `tune` run. Per-stock predictions of the selected configurations are also written to `research/predictions/RANKER-R6-F{n}.csv`.

Environment of the delivered results: Python 3.12.10, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1. Different library versions may produce numerical differences.

## Fixed experiment rules

- Train on the latest 6 calendar years, refit every 26 weeks; predict every week and hold an equal-weight Top-5 portfolio.
- Factor sets are 5, 7, 11 (7 internal factors plus 4 macro factors) and 45 factors.
- Each fit uses only realised labels with `label_end_date < fit_date`. The imputation medians and 1% / 99% winsorisation thresholds are computed from the current training window only.
- Development is 2022-09-02 to 2024-08-30, 105 weeks; Validation is 2024-09-06 to 2025-08-29, 52 weeks; Final Test is 2025-09-05 to 2026-08-21, 51 weeks and 3,304 predictions.
- Each factor set has 36 predefined label/hyperparameter candidates (144 in total), selected using Development only; all four configurations are frozen before later evaluation.
- Gates are Rank IC > 0, Rank ICIR >= 0.05 and positive Rank IC weeks >= 50%; passing candidates are ranked by Development Top5 Excess, Rank IC and Rank ICIR in descending order, with the lower candidate number winning exact ties.
- What is frozen is the label scheme and the hyperparameters; the model is still refitted on a rolling 26-week schedule using the historical labels realised at that time.

## Eight evaluation metrics

IC, ICIR, Rank IC, Rank ICIR, Annualized Return, Information Ratio, Top5 Excess and Positive Rank IC, consistent with the original XGBoost.

Training labels are within-week grades of stock returns, and predictions are ranking scores. Evaluation and portfolio returns always use realised returns; grade labels or prediction scores are never treated as returns. ICIR and Rank ICIR are not annualised; the Information Ratio is annualised by the square root of 52; Top5 Excess is the geometric annualisation of the weekly excess returns, not the difference of two annualised returns. Transaction costs are not deducted.

## Delivered Final Test results

| Factor set | IC | ICIR | Rank IC | Rank ICIR | Annualised return | Information ratio | Top5 Excess | Positive Rank IC weeks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0485 | 0.3314 | 0.0503 | 0.3610 | 23.80% | 1.1827 | 16.10% | 66.67% |
| 7 | 0.0322 | 0.2133 | 0.0302 | 0.1875 | 9.93% | 0.3142 | 4.13% | 56.86% |
| 11 | 0.0069 | 0.0467 | 0.0117 | 0.0796 | 8.14% | 0.1883 | 1.84% | 49.02% |
| 45 | 0.0039 | 0.0243 | 0.0244 | 0.1524 | -7.27% | -0.5735 | -12.39% | 49.02% |

The CSV keeps unrounded values; returns and shares are stored as decimals.

> **Reproducibility note**: the model uses `subsample=0.8`. When rerun on Linux on 2026-10-23, Development selected different candidates (candidate 13 for 5, 7 and 45 factors; candidate 18 for 11 factors) and the stage metrics differ clearly from the table above (e.g. 5-factor Final Test annualised −17.3%). Exact independent refitting of the original models has not been established across environments; see the [frozen-model reproducibility rationale](../../README.md#why-the-trained-models-are-frozen).

## Comparability with the other three models

Input data, factors, rolling window, refit schedule and the eight evaluation metrics are the same. Under the previously agreed protocol, Ranker and XGBoost select on 105 Development signal weeks, while Ridge and Random Forest use 104 weeks. The Ranker's candidate search budget is also larger, so result differences cannot be attributed entirely to the algorithm.

The original Final Test had been viewed before. The optimised version did not use Validation or Final Test metrics to select candidates, but this round is an iterative experiment and cannot be called a never-observed independent test.
