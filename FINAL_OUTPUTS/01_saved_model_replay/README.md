# Saved Ensemble replay

This folder is the review copy of the focused external simulation. It retains only the three-component Ensemble `Top5_NoLoss13w` result on Common210 FullHistory.

Run `python safecapital.py` from the project root. The command reloads the 45 frozen model snapshots, verifies every historical prediction, and executes one backtest case. It does not fit a model.

The separate Ranker comparison, other strategy outputs and the final-year annualised-return display are excluded. See `selected_strategies.csv` for the retained value and `02_ensemble_evaluation_backtest_report.md` for scope and limitations.

`verified_replay/` contains a complete passed replay: regenerated per-stock predictions, weekly capital, strategy summary, prediction verification, independent account verification and the run manifest. The prediction table precedes the 13-week filter and is not a holdings list. No individual transaction or holding export is required for this methods-and-results presentation.
