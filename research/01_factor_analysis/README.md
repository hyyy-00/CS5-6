# 01 Single-factor screening

Full-sample Rank IC screening of the 45 candidate factors of the 66 training stocks, producing the factor sets shared by the four models.

## Running (from the project root)

```bash
python research/01_factor_analysis/factor_analysis.py            # reads data/test_45_factors_weekly.csv by default
```

Outputs are written to `research/01_factor_analysis/outputs/`:

| File | Contents |
|---|---|
| `full_sample_factor_screening.csv` | Coverage, Mean Rank IC, Rank ICIR, annual sign consistency, screening status and reason for every factor |
| `weekly_rank_ic.csv` / `annual_rank_ic.csv` | Weekly and annual Rank IC |
| `factor_correlations.csv` / `high_correlation_pairs.csv` | Mean weekly Spearman correlation matrix and factor pairs with \|ρ\| ≥ 0.85 |
| `seven_factor_results.csv` / `five_factor_comparison.csv` | Seven-factor screening results and the design comparison of the financial factors |
| `factor_sets.json` / `model_features.json` | The 5 / 7 / 11 / 45-factor lists used by the models and the screening configuration |

`results/Factor_Analysis_Results_EN.xlsx` is the delivered results workbook; `Factor_Analysis_Guide_EN.docx` explains the method.

## Screening rules

- Weekly cross-sectional Spearman Rank IC (average ranks), with at least 20 stocks per week and at least 5 distinct factor values; macro factors are identical across stocks within a week, so their Rank IC is undefined and they are kept separately.
- Selected into the 7 factors: |Mean Rank IC| ≥ 0.01, |Rank ICIR| ≥ 0.08 (not annualised), sign agreement in ≥ 60% of eligible years, coverage ≥ 80%, at least 104 valid weeks.
- De-duplication: greedy selection in descending |Mean Rank IC|; a later factor with mean weekly |ρ| ≥ 0.85 to an already selected one becomes a correlated alternative; the 4 `cs_rank_*` columns identical to raw return/volatility columns are dropped.
- Resulting factor sets:
  - **7 factors**: `revenue_growth_yoy, ret_120d, cashflow_to_assets, ret_1d, ret_5d, close_location_value, body_range_ratio`
  - **11 factors**: the 7 factors + 4 macro factors (VIX level, 5-observation VIX change, 10y−2y spread, 5-observation 10y change)
  - **5 factors** (specified by financial dimension in advance, not chosen by IC gates): revenue growth, operating profitability, cash flow / assets, debt / assets, asset growth
  - **45 factors**: all candidate columns

> **Limitation**: screening uses returns from the whole period (2016-09-02 to 2026-08-21), so later splitting the same history into Development / Validation / Final Test does not create an independent test. The final week, 2026-08-28, has no next-week price and is excluded.
