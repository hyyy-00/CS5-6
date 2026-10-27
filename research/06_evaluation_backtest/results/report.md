# Model evaluation and multi-strategy backtest (auto-generated)

Series: 18 (15 original models, 3 consensus); 15 strategies, 14 windows, one-way 0/10/25bps: 11,340 main backtests and 972 rebalance-phase tests.

## Predictive power: Final Test

|Model|IC|RankIC|RankICIR|Positive RankIC weeks|RMSE|
|---|---:|---:|---:|---:|---:|
|XGB-R6-F7|0.026|0.054|0.356|68.63%|0.059|
|CONSENSUS-R6-F7|0.038|0.047|0.292|60.78%|—|
|XGB-R6-F45|0.037|0.047|0.297|64.71%|0.059|
|RF-R6-F7|0.045|0.045|0.307|64.71%|0.059|
|CONSENSUS-R6-F11|0.033|0.044|0.278|64.71%|—|
|RIDGE-R6-F7|0.030|0.042|0.230|64.71%|0.059|
|RIDGE-R6-F11|0.028|0.037|0.208|66.67%|0.059|
|XGB-R6-F11|0.028|0.036|0.254|56.86%|0.060|
|RANKER-R6-F11|0.008|0.035|0.228|60.78%|—|
|RIDGE-R6-F5|0.033|0.027|0.208|58.82%|0.059|
|RF-R6-F11|0.040|0.027|0.193|52.94%|0.059|
|CONSENSUS-R6-F5|0.019|0.024|0.146|47.06%|—|
|RF-R6-F5|0.010|0.021|0.126|49.02%|0.060|
|RANKER-R6-F7|-0.003|0.019|0.128|54.90%|—|
|XGB-R6-F5|-0.004|0.019|0.106|50.98%|0.060|
|RANKER-R6-F5|0.003|0.010|0.064|50.98%|—|
|RF-R6-F45|0.014|0.008|0.046|47.06%|0.059|
|RANKER-R6-F45|-0.011|0.002|0.013|54.90%|—|

## Equal-weight Top5 (one-way 10bps)

|Model|Full-history net ann.|Validation net ann.|Final Test net ann.|Final Test max drawdown|
|---|---:|---:|---:|---:|
|CONSENSUS-R6-F11|3.24%|6.25%|-12.22%|-23.89%|
|CONSENSUS-R6-F5|3.97%|14.38%|-7.75%|-28.59%|
|CONSENSUS-R6-F7|-2.77%|5.02%|-6.16%|-29.00%|
|RANKER-R6-F11|-8.64%|-13.63%|-26.34%|-38.37%|
|RANKER-R6-F45|-4.49%|-4.73%|-8.53%|-29.01%|
|RANKER-R6-F5|5.17%|-2.86%|-18.04%|-20.63%|
|RANKER-R6-F7|1.92%|7.74%|-14.40%|-29.84%|
|RF-R6-F11|0.21%|-10.23%|-3.92%|-22.64%|
|RF-R6-F45|-0.35%|8.79%|-15.50%|-21.33%|
|RF-R6-F5|5.07%|6.75%|-12.52%|-32.12%|
|RF-R6-F7|-6.93%|-10.26%|-27.96%|-32.94%|
|RIDGE-R6-F11|-0.70%|-7.45%|-5.65%|-26.26%|
|RIDGE-R6-F5|-4.08%|18.11%|-15.10%|-20.55%|
|RIDGE-R6-F7|-3.65%|-11.43%|-13.82%|-21.71%|
|XGB-R6-F11|-4.84%|-10.84%|-36.34%|-41.87%|
|XGB-R6-F45|-5.01%|-15.83%|-13.19%|-29.73%|
|XGB-R6-F5|1.10%|25.43%|-23.15%|-38.67%|
|XGB-R6-F7|-7.28%|9.29%|-21.04%|-32.70%|

## Best Final Test strategy per model (descriptive only; selection bias)

|Model|Strategy|Net ann.|Max drawdown|Sharpe|
|---|---|---:|---:|---:|
|RF-R6-F11|Top5_Rebalance4w|38.70%|-18.46%|1.536|
|CONSENSUS-R6-F7|Top5_Rebalance2w|32.07%|-15.87%|1.174|
|RIDGE-R6-F5|Top5_NoLoss13w|29.59%|-13.71%|1.158|
|RF-R6-F5|Top5_NoLoss13w|26.48%|-18.49%|0.995|
|XGB-R6-F5|Top10_NoLoss13w|23.17%|-16.70%|1.182|
|CONSENSUS-R6-F5|Top5_NoLoss4w|21.47%|-18.33%|0.979|
|RIDGE-R6-F7|Top5_NoLoss13w|19.42%|-17.01%|0.846|
|RIDGE-R6-F11|Top5_NoLoss13w|18.34%|-18.19%|0.818|
|RF-R6-F45|Top5_Rebalance2w|15.71%|-13.16%|0.784|
|RF-R6-F7|Top10_NoLoss13w|14.93%|-22.50%|0.725|
|XGB-R6-F7|Top5_Rebalance4w|14.93%|-19.93%|0.608|
|RANKER-R6-F7|Top5_Rebalance4w|13.38%|-21.91%|0.548|
|RANKER-R6-F5|Top10_NoLoss13w|11.64%|-13.94%|0.712|
|RANKER-R6-F11|Top10_NoLoss13w|8.53%|-20.78%|0.481|
|CONSENSUS-R6-F11|Top5_NoLoss13w|5.55%|-26.12%|0.340|
|XGB-R6-F45|Top10_NoLoss13w|4.73%|-23.00%|0.321|
|RANKER-R6-F45|Top10_PositiveScore|0.71%|-16.43%|0.130|
|XGB-R6-F11|Top5_Rebalance4w|-6.68%|-26.96%|-0.139|

## Reconciliation with the results delivered by the training scripts

|Model family|Max RankIC difference|Max IC difference|
|---|---:|---:|
|RANKER|4.90e-02|5.92e-02|
|RF|2.65e-03|2.09e-03|
|RIDGE|9.02e-17|6.73e-16|
|XGB|3.00e-02|5.55e-02|

## Interpretation limits

- Same-close signals and same-close entry are research assumptions; slippage, market impact, liquidity and capacity are not modelled. Cash earns 0 and the benchmark is gross of costs.
- The full history includes the Development tuning period; the Final Test has been viewed in earlier rounds and is not a new unseen test.
- The best Final Test strategy is picked after comparing 15 strategies on the same data, so it carries multiple-comparison bias.
- XGBoost and Random Forest use subsampling / parallel random numbers; other platforms or library versions can give different predictions.
