# Selected external account engine

`external40/integer_backtest.py` supplies portfolio preparation, integer-share execution and performance calculations for the frozen Ensemble replay. Use `python safecapital.py` from the project root to run the selected strategy.

`pipeline/trading.py:independent_case` independently reconstructs selection and accounting from the in-memory trades and holdings. The removed standalone comparison, recovery helpers and V1 external backtest have no entry points in this delivery.

See the root README for the 13-week filter, Top5 allocation and retained outputs. The original four-model research engine is separate under `research/06_evaluation_backtest/`.
