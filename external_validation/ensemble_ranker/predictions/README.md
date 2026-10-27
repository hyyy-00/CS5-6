# Historical prediction reference

`stock_rankings.csv` contains 13,517 stock-week rows across 366 weeks for the 40 external stocks. It is the unchanged reference from the model handoff, SHA-256 `d8e219d6749031117add13e13d8fbea15abecec87e190a0c72e92ec3d340d007`.

The default root command `python safecapital.py` regenerates scores from the original saved model files and external factor input, then compares them to this reference before backtesting the regenerated scores. It does not fit any models.

The older spelling `python safecapital.py --legacy-ensemble` is now a compatibility alias for the same frozen-model inference workflow. Realised-return and evaluation-label columns are excluded from prediction and trading.
