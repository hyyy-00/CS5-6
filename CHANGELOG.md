# Changelog

## 2026-10-26

- Restricted external strategy delivery to the frozen Ensemble Top5_NoLoss13w Common210 FullHistory case.
- Removed standalone external Ranker fitting, model artifacts, predictions and comparisons.
- Converted V1 into a training-only Ensemble demonstration using the supplied processed weekly training data. Removed external prediction, backtest, reporting and resume interfaces from this workflow.
- Removed retrained-Ensemble external results and separate final-year annualised-return displays from the external result package.
- Retained original four-model research methodology and evidence.
- Documented strategy selection, 13-week filtering, equal target allocations, integer shares, fees and cash rules.
- Retained a verified complete frozen replay with predictions, weekly capital and numerical verification records. Individual order and holding exports are outside the presentation scope.
- Defined reproduction as starting from the supplied processed weekly data; raw acquisition and reconstruction are outside delivery scope.
- Added regression coverage for the training-only boundary and preserved failure/status, exact inference and independent accounting checks. See docs/VALIDATION.md for validation outcomes.
- Kept delivery text in English and refreshed file-integrity manifests.
