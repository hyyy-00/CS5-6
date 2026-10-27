# Selected historical Ensemble models

The root entry point loads the 45 unchanged model JSON files from `saved_models/`, organised as three components times fifteen historical update dates. No call to model fitting is made. The external backtest exposes only Common210 FullHistory `Top5_NoLoss13w`. The saved model hashes are checked against `saved_models_manifest.json`, whose hashes were checked against the original handoff archive manifest during packaging.

`inputs/test40.csv` is the unchanged original external input. Its factor columns are used for historical inference. `predictions/stock_rankings.csv` provides the reference calendar and expected scores for exact replay validation. Future-return labels are excluded from model input.

`model_configuration.json` records component features, parameters and the equal-weight percentile formula. The manifest also preserves original training-window dates and label-observability metadata; inference does not require rerunning training.

Run from the project root with `python safecapital.py`. The selected models cover 2019-08-30 through 2026-08-28 and must be used only within their recorded historical intervals. They are not a live market-data service.

The `code/ranker_core.py` file supports the Ensemble Training Demonstration (V1). It is explicitly selected by `--retrain-v1`, uses the supplied corrected weekly training data and produces fitted component files only. It does not run an external backtest.

The original model training functions and original 66-stock data are also supplied under `training_reference/` and `inputs/train66.csv`. The optional original training wrapper can be inspected in `training_reference/train_models.py`; the documented V1 demonstration is in `docs/V1_RETRAINING.md` from the project root. Experimental models are saved to a new run folder; default inference continues to load the same delivered files from `saved_models/`. This freezes inference artifacts for reproducibility and does not claim that refitting is invariant across environments.
