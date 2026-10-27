# Historical Ensemble investment simulation

Prediction source: loaded historical saved models; new inference, no training. No models are trained in this simulation.

Only the Ensemble Top5_NoLoss13w strategy is retained. It excludes stocks with a negative observable trailing 13-week return before selecting the five highest Ensemble ranks.

Initial capital is 100,000, one-way transaction cost is 10bps, and the execution engine uses integer shares.

| Window | Strategy | Weeks | Annualised return | Sharpe | Maximum drawdown | Final capital |
|---|---|---:|---:|---:|---:|---:|
| FullHistory | Top5_NoLoss13w | 210 | 10.44% | 0.543 | -24.88% | 149330.61 |

Only date, ticker and ensemble_ranking_score are passed to the trading engine. Future realised-return and evaluation-label columns in the source file are not used for selection or execution.
