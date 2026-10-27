# Random Forest Standard Experiment

This version aligns Random Forest with the common experiment protocol used for the current Standard XGBoost comparison.

## Configuration

- Model: `RandomForestRegressor` predicting the continuous forward-aligned weekly `target` log return.
- Factor sets: 5 factors, 7 factors, 7 factors plus 4 macro factors (11 total), and all 45 factors.
- Training history: trailing six calendar years.
- Refit schedule: every 26 calendar weeks.
- Development: all RF candidates are evaluated using Development predictions only. Candidates must satisfy Mean Rank IC > 0, Rank ICIR >= 0.05 and positive Rank IC weeks >= 50%; the passing candidate with the highest Development Top-5 excess annualised return is selected.
- Validation and Final Test: the selected parameters are frozen. Models may refit chronologically every 26 weeks, but no later stage is used to retune parameters.
- Stage labels: each prediction row is assigned from its own signal date, including when a 26-week refit block crosses a stage boundary.
- Preprocessing: median imputation and 1%/99% winsorisation are fitted on each training window only.

## Files

- `train_random_forest.py`: training and evaluation script (original name `random_forest_standard_6y_26w.py`).
- `results/random_forest_5_7_11_45_6y26w_results.csv`: delivered stage results (selected parameters per factor set).
- `results/random_forest_5_7_11_45_6y26w_results_development_candidates.csv`: every candidate's Development score and gate status.
- Input: `data/test_45_factors_weekly.csv` at the repository root.

## Run (from the repository root)

```bash
python research/03_random_forest/train_random_forest.py --feature-set all
python research/03_random_forest/train_random_forest.py --feature-set 5     # one factor set
```

Factor set names are `5`, `7`, `7_macro` and `45`. Results go to `research/03_random_forest/outputs/`;
per-stock predictions of the selected parameters go to `research/predictions/RF-R6-F{n}.csv` for `06_evaluation_backtest`.

Candidate grid (18 entries, unchanged): `n_estimators=100`, `max_depth ∈ {4,6,8}`, `min_samples_leaf ∈ {10,20,30}`,
`max_features ∈ {"sqrt", 0.5}`, `random_state=5703`, `n_jobs=-1`. Development selection excludes labels that are only
realised at the Validation boundary (104 selection weeks).
