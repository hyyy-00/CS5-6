# Ensemble external-data evaluation: retained strategy

## Reporting scope

This focused external-simulation deliverable retains one result only:

- model: three-component equal-weight Ensemble;
- strategy: `Top5_NoLoss13w`;
- rule: exclude stocks whose observable trailing 13-week return is negative, then select the five highest Ensemble ranks;
- scope: Common210 FullHistory, 210 signal weeks;
- initial capital: 100,000;
- one-way transaction cost: 10bps;
- integer shares with retained cash.

The separate Ranker comparisons, other strategy tables and their charts are not part of this deliverable. A separate last-year/FinalTest annualised-return figure is intentionally not reported in this focused evaluation. Its omission must not be interpreted as evidence of positive or stable final-year performance.

## Retained value

| Model | Window | Strategy | Weeks | Annualised return | Sharpe | Maximum drawdown | Final capital | Fees |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Ensemble | FullHistory | Top5_NoLoss13w | 210 | 10.4390% | 0.5434211 | -24.8781% | 149330.61 | 14520.837 |

## Reproducibility

The default command reloads 45 frozen Ensemble model snapshots, checks all 13,517 historical prediction rows against the reference table, restricts trading to the 8,354 Common210 rows, and runs only the retained case above. No model is fitted. The independent verifier now reconstructs the trailing 13-week filter, selected tickers, share targets, cash flows, holdings and summary metrics.

The retained number is a historical simulation, not a forecast or investment guarantee. Quote proxies, corporate actions, actual fills, slippage, impact, capacity and all financial-factor publication boundaries are not independently certified.
