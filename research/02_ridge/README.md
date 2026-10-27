# Ridge: common-protocol experiment

Version date: 2026-10-13.

## Protocol

- Model: Ridge regression with an intercept; the intercept is not penalised and the objective is `sum((y - prediction)^2) + alpha * sum(coef^2)`.
- Label: `target`, the forward-aligned next-week log return; it is not converted to the 0–4 relevance levels used by the Ranker.
- Universe: 66 stocks; some weeks have missing stocks, so the number of valid stocks per week is not always 66.
- Factor sets: 5 factors, 7 factors, 7 internal factors plus 4 macro factors (11 factors), and 45 factors. "Four sets" means four input combinations, not an enumeration of three-factor subsets.
- Training history: a trailing window of 6 **calendar years**, not a fixed 312 observed weeks.
- Refit frequency: every 26 calendar weeks (182 days) forms a refit block, refitted on the first actual signal date of the block. It is not 26 months, and the count does not restart at each stage.
- Stages: Development, Validation and Final Test are split by signal date; a refit block that crosses a stage boundary keeps each prediction's own stage.
- Selection: Development only. Candidates must pass Mean Rank IC > 0, Rank ICIR >= 0.05 and positive Rank IC weeks >= 50%; the candidate with the highest Development Top-5 excess annualised return is then selected.
- Validation and Final Test: use the frozen alpha without retuning; the model is still refitted chronologically every 26 weeks, using only realised labels with `label_end_date < fit_date`.
- This version runs rolling mode only; no fixed window, three-factor enumeration or other extra experiments.

### Alignment with the teammate's version and explicit differences

The input CSV is byte-identical to the teammate's four-file XGBoost package, SHA-256:

`26e234363e83f2eaefe0ba787b6909be0999f6d4405fbf6ef2e2c7cb8b918d97`

The four factor sets, valid samples, 6-calendar-year window, 26-week refit dates, signal-date stage split and the eight summary metric formulas have all been checked. The model-specific alpha grid and standardisation do not need to imitate the tree models.

**One deliberate safety difference remains at the selection boundary:** the teammate's script scores all Development signal weeks; this Ridge keeps the original check and additionally requires `label_end_date < first Validation signal date` during selection. The 62 labels of 2024-08-30 are only realised on 2024-09-06, so they do not take part in alpha selection. Selection actually uses 104 weeks and 6,410 records; stage reporting still shows the full Development period by signal date, 105 weeks and 6,472 records. The Development metrics in the summary table are therefore not a direct copy of the selection scores. If the team needs exactly the same selection sample, the teammate's code should add the same label-boundary check; this point cannot be claimed to be identical at present.

## Files

- `train_ridge.py`: main training script (original name `train_ridge_5_7_11_45_6y26w.py`); depends only on NumPy and pandas.
- `results/ridge_5_7_11_45_6y26w_results.csv`: the delivered rolling results.
- Input data: `data/test_45_factors_weekly.csv` in the project root (identical for all four models).

The results CSV has 10 rows: three stages each for the 5, 7 and 11 factor sets that passed the gate (9 rows), plus one `no_candidate_passed` row for 45 factors. Empty metrics mean "not admitted / no result for that stage", **not a zero score or a program omission**.

## Running (from the project root)

```bash
python research/02_ridge/train_ridge.py --check-only     # only check data and dates
python research/02_ridge/train_ridge.py                  # full training
```

- Results are written to `research/02_ridge/outputs/ridge_5_7_11_45_6y26w_results.csv`; to protect existing results the script refuses to overwrite an existing output file, so delete it or pass a new `--output` before rerunning.
- Per-stock predictions of the selected configurations are written to `research/predictions/RIDGE-R6-F{5,7,11}.csv` for `06_evaluation_backtest`.
- Add `--audit-dir NEW_FOLDER` for audit files such as selection records and the refit schedule.
- Each factor set evaluates six alphas: `0.01, 0.1, 1, 10, 100, 1000`, 24 candidates in total. Ties are broken by Top5 Excess, Rank IC and Rank ICIR in descending order, then by alpha in ascending order.
- The console ends with three tables: the selected hyperparameter (alpha), the stage prediction metrics, and the stage Top-5 portfolio metrics.

## Factors and preprocessing

- 5 factors: `revenue_growth_yoy`, `operating_profitability`, `cashflow_to_assets`, `debt_to_assets`, `asset_growth_yoy`.
- 7 factors: `revenue_growth_yoy`, `ret_120d`, `cashflow_to_assets`, `ret_1d`, `ret_5d`, `close_location_value`, `body_range_ratio`.
- 11 factors: the 7 factors above plus `macro_vix_level`, `macro_vix_change_5obs`, `macro_yield_spread_10y_2y`, `macro_dgs10_change_5obs`.
- 45 factors: all 45 columns of the input table after excluding `date, ticker, open, high, low, close, adj_close, volume, target, target2`, in their original order.

At every refit, the missing-value medians, 1% / 99% winsorisation bounds, means and population standard deviations are computed from the current training sample only and the same transform is applied to that block's predictions. A zero standard deviation is replaced by 1; an entirely missing column is filled with 0. **Ridge standardises its inputs; the teammate's XGBoost does not, as the algorithm requires — this is not an inconsistency in the window rules.**

## Actual dates and samples

Raw input: 2016-09-02 to 2026-08-28, 522 signal dates, 27,651 records. Checking `target` against the same stock's adjusted price in the next market-wide observed week gives a maximum error of `5.21e-16`. The final week's 66 records have no next-week price and another 16 lack the same stock's next-week price, so 82 records are excluded; 27,569 verifiable labels remain, including training history.

| Stage | Actual signal range | Weeks | Records |
|---|---|---:|---:|
| Development (reporting) | 2022-09-02 to 2024-08-30 | 105 | 6,472 |
| Development (selection) | 2022-09-02 to 2024-08-23 | 104 | 6,410 |
| Validation | 2024-09-06 to 2025-08-29 | 52 | 3,230 |
| Final Test | 2025-09-05 to 2026-08-21 | 51 | 3,304 |

The last evaluable signal date is 2026-08-21, whose label is realised on 2026-08-28. The eight refit dates are 2022-09-02, 2023-03-03, 2023-09-01, 2024-03-01, 2024-08-30, 2025-02-28, 2025-08-29 and 2026-02-27.

## Delivered results

Final Test only; the CSV has all three stages.

| Factor set | Frozen alpha | Rank IC | Rank ICIR | Top-5 annualised return | Top-5 excess annualised return |
|---|---:|---:|---:|---:|---:|
| 5 factors | 1000 | 2.7338% | 0.2077 | -14.5591% | -20.0171% |
| 7 factors | 100 | 4.1675% | 0.2305 | -7.8620% | -13.4696% |
| 11 factors | 1000 | 3.7280% | 0.2085 | 0.9379% | -4.9331% |
| 45 factors | not admitted | — | — | — | — |

The Development Rank ICIR of the 45-factor candidates ranges from 0.0045 to 0.0185, all below the 0.05 gate. Its non-admission must not be read as a training failure.

## Metric formulas and interpretation

- IC: weekly Pearson correlation between predictions and realised log returns, averaged over valid weeks.
- Rank IC: weekly Spearman correlation (average ranks for ties), averaged.
- ICIR / Rank ICIR: mean of the weekly series divided by its sample standard deviation (`ddof=1`), **not multiplied by sqrt(52)**.
- Top-5: each week the 5 stocks with the highest predictions, equal-weighting their simple returns `expm1(target)`.
- Benchmark: equal-weight simple return of stocks with verifiable labels in the same week; not SPY or an external index.
- Annualized Return: `exp(mean(log(1 + weekly_top5_return)) * 52) - 1`.
- Top5 Excess: first compute weekly Top-5 return minus the same week's benchmark, then annualise that excess series with the same formula; **not the difference of two annualised returns**.
- Information Ratio: mean weekly excess return / sample standard deviation, multiplied by `sqrt(52)`.
- Positive Rank IC: share of valid Rank IC weeks that are above zero.

Correlations, returns and week shares are stored as raw decimals in the CSV; e.g. `0.041675` means a Rank IC of about 4.17%, not 0.041675%. The metric columns match the teammate's; RMSE is not added to this table. Ridge is a continuous return regression, so regression errors could be computed, but Ranker scores cannot use the same RMSE definition.

Results are historical comparisons before costs, not executable trading returns. The 11-factor set has a slightly positive Top-5 annualised return but a negative excess return; a positive Rank IC does not guarantee profit. The four factor sets are predefined inputs, and their screening was not re-shown to be independent of the historical test period; the data were used in earlier experiments, so the Final Test cannot be called never-observed new data.

## Second verification

The teammate's input data were checked byte for byte, as were the factors, valid samples and calendar refit rules. The 24 predictions from the eight refits of the three admitted models were recomputed with an independent augmented least-squares implementation; the maximum prediction error is `1.29e-16`. All nine stage-metric rows were recomputed independently with a maximum absolute error of `3.56e-15`. The selection gates, frozen alphas and non-admitted status were all verified.
