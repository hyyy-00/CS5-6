# Ensemble external-simulation files

This directory contains the focused saved-model evidence for the three-component Ensemble. The external simulation exposes only `Top5_NoLoss13w` on Common210 FullHistory.

Retained result files:

- `02_ensemble_evaluation_backtest_report.md`: focused report and interpretation limits.
- `03_selected_strategy_metrics.csv`: the single retained strategy value.
- `45_rolling_models_replay_check.csv`: saved-snapshot replay audit.
- `model_configuration.json`, `factor_definitions.csv`: Ensemble definition.
- `prediction_price_join_audit.json`, `five_factor_value_rank_check.json`, `original_rankic_recomputation.csv`: supporting audits.

Separate Ranker comparisons, other strategy tables, charts and final-year annualised-return presentation were removed from this external deliverable. Original research material remains separate. V1 demonstrates component training only and has no external prediction or backtest outputs. A verified complete replay is retained under FINAL_OUTPUTS/01_saved_model_replay/verified_replay/ from the project root.
