# Standard XGBoost: common-protocol experiment

## Protocol

- Model: `XGBRegressor` with a squared-error regression objective.
- Label: `target`, the forward-aligned next-week log return; it is not converted to the 0–4 relevance levels used by the Ranker.
- Universe: 66 stocks.
- Factor sets: 5 factors, 7 factors, 7 internal factors plus 4 macro factors (11 factors), and 45 factors.
- Training history: 6-year trailing window.
- Refit frequency: every 26 weeks.
- Stages: Development, Validation and Final Test are split by signal date; the actual signal start dates are resolved from the data to avoid misassigning stages by label end date.
- Selection: Development only. Candidates must pass Mean Rank IC > 0, Rank ICIR >= 0.05 and positive Rank IC weeks >= 50%; the candidate with the highest Development Top-5 excess annualised return is then selected.
- Validation and Final Test: use the frozen hyperparameters without retuning; the model can still be refitted chronologically every 26 weeks, using only labels realised before the prediction date.

## Files

- `train_xgboost.py`: main training script (original name `train_xgboost_5_7_11_45_6y26w.py`).
- `results/xgboost_5_7_11_45_6y26w_results.csv`: the delivered 6-year, 26-week rolling results.
- Input data: `data/test_45_factors_weekly.csv` in the project root.

## Running (from the project root)

```bash
python research/04_xgboost/train_xgboost.py --check-only
python research/04_xgboost/train_xgboost.py                      # default --mode rolling --feature-set all
python research/04_xgboost/train_xgboost.py --mode fixed         # optional: fixed-window comparison
```

The script uses 12 candidate hyperparameter sets by default (`max_depth ∈ {2,3}`, `min_child_weight ∈ {5,20}`, `n_estimators ∈ {50,150,300}`). Results go to `research/04_xgboost/outputs/`, and per-stock predictions of the selected rolling configuration go to `research/predictions/XGB-R6-F{n}.csv`.

> **Reproducibility note**: the model uses `subsample=0.8` and `colsample_bytree=0.8`. Page 3 of the report records that rerunning the delivered script differs clearly from the delivered CSV (5-factor Final Test gross annualised −20.80% vs −13.58%); the package did not keep the frozen hyperparameters or per-stock predictions, so the cause cannot be confirmed. "XGB" in `06_evaluation_backtest` always means the predictions rerun with this script.

## Interpretation

Standard XGBoost outputs continuous return predictions, so RMSE can be reported; when comparing with XGBRanker, use the common ranking and strategy metrics such as Rank IC, Rank ICIR, Top-5 annualised return and Top-5 excess return. XGBRanker outputs ranking scores that cannot be treated as return predictions for RMSE.
