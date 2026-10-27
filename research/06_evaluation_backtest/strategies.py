"""Portfolio strategy rules and the weekly weight-based backtest.

The rules and arithmetic are taken unchanged from the report-chapter code
(``reference_code/portfolio_strategies.py`` and ``full_history_backtest.py``).
Every filter, volatility and trend input uses only data known on the signal
date. Weekly stock return = next global week adjusted close / current - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_STRATEGIES = ['Top5', 'Top10', 'Top5_PositiveScore', 'Top10_PositiveScore',
                   'Top5_NoLoss4w', 'Top10_NoLoss4w', 'Top5_NoLoss13w', 'Top10_NoLoss13w',
                   'Top5_RankWeighted']
NEW_STRATEGIES = ['Top5_LowVol26w', 'Top5_RankRisk26w', 'Top5_Buffer10',
                  'Top5_Rebalance2w', 'Top5_Rebalance4w', 'Top5_Trend26w']
STRATEGIES = BASE_STRATEGIES + NEW_STRATEGIES
# Path-dependent strategies are replayed from cash inside every window.
STATEFUL = ['Top5_Buffer10', 'Top5_Rebalance2w', 'Top5_Rebalance4w']
COSTS_BPS = [0, 10, 25]
STOCK_CAP = 0.35

LABELS = {'Top5': 'Equal-weight Top5', 'Top10': 'Equal-weight Top10', 'Top5_RankWeighted': 'Rank-weighted Top5',
          'Top5_PositiveScore': 'Top5 positive score', 'Top10_PositiveScore': 'Top10 positive score',
          'Top5_NoLoss4w': 'Top5 no loss over past 4 weeks', 'Top10_NoLoss4w': 'Top10 no loss over past 4 weeks',
          'Top5_NoLoss13w': 'Top5 no loss over past 13 weeks', 'Top10_NoLoss13w': 'Top10 no loss over past 13 weeks',
          'Top5_LowVol26w': 'Low-volatility weighted', 'Top5_RankRisk26w': 'Rank-risk weighted',
          'Top5_Buffer10': 'Top-10 retention buffer', 'Top5_Rebalance2w': 'Rebalance every 2 weeks',
          'Top5_Rebalance4w': 'Rebalance every 4 weeks', 'Top5_Trend26w': 'Market trend filter'}

RULES = {
    'Top5_LowVol26w': 'Take the top 5 predicted stocks that have a complete 26-week volatility history and weight them by inverse volatility, capped at 35% per stock; with fewer than 5 stocks the equity budget is count/5 and the rest stays in cash.',
    'Top5_RankRisk26w': 'From the same eligible pool take the top 5 and weight by (6 - rank) / 26-week standard deviation of weekly returns, capped at 35% per stock; with fewer than 5 stocks the equity budget is count/5.',
    'Top5_Buffer10': 'Start with the top 5; each week keep holdings that are still in the current top 10, fill empty slots by current rank, then rebalance to equal weights of 20% each.',
    'Top5_Rebalance2w': 'Buy the top 5 from cash on the first available signal and re-select equal weights every 2 calendar weeks; in between, holdings and cash drift naturally and no trades follow new ranks. A missing signal liquidates; a returning signal buys immediately and resets the rebalance clock.',
    'Top5_Rebalance4w': 'Same as above, rebalancing every 4 calendar weeks.',
    'Top5_Trend26w': 'Build an equal-weight return index from verifiable adjusted prices of the original universe up to the current week; buy the equal-weight Top5 when the index is above its 26-week moving average (including the current week), otherwise hold cash.',
    'Consensus': 'Equal-weight average of the weekly within-universe rank percentiles of the four model families for the same factor set; scores are rounded to 12 decimals to fix numerical ties, then ties break by ticker; only stocks and dates predicted by all four are used.',
}


def market_inputs(prices: pd.DataFrame) -> dict:
    """Backward-looking volatility and trend inputs from a date x ticker price pivot."""
    past = prices / prices.shift(1) - 1
    vol = past.rolling(26, min_periods=26).std(ddof=1)
    market_index = (1 + past.mean(axis=1).fillna(0)).cumprod()
    trend = (market_index > market_index.rolling(26, min_periods=26).mean()).fillna(False)
    return dict(vol=vol, trend=trend, stock_returns=prices.shift(-1) / prices - 1,
                market_index=market_index, prices=prices)


def capped_weights(scores: dict, budget: float = 1., cap: float = STOCK_CAP) -> dict:
    """Proportional weights with a per-stock cap; any infeasible budget stays cash."""
    scores = {t: float(v) for t, v in scores.items() if np.isfinite(v) and v > 0}
    allocation = {t: 0. for t in scores}
    remaining = min(float(budget), len(scores) * cap)
    active = dict(scores)
    while active and remaining > 1e-14:
        total = sum(active.values())
        proposed = {t: remaining * s / total for t, s in active.items()}
        clipped = [t for t, w in proposed.items() if w > cap + 1e-14]
        if not clipped:
            allocation.update(proposed)
            break
        for t in clipped:
            allocation[t] = cap
            remaining -= cap
            active.pop(t)
    return allocation


def windows(p: pd.DataFrame, calendar=None) -> dict:
    """14 evaluation windows: full history, trailing, calendar years and stages."""
    dates = sorted(p.date.unique()) if calendar is None else calendar
    result = {'FullHistory': dates}
    for n in [13, 26, 52, 104, 156]:
        if len(dates) >= n:
            result[f'Last{n}w'] = dates[-n:]
    for year in sorted({pd.Timestamp(d).year for d in dates}):
        result[f'Year{year}'] = [d for d in dates if pd.Timestamp(d).year == year]
    for stage, start, end in [('Development6y', '2022-09-01', '2024-09-01'),
                              ('Validation', '2024-09-01', '2025-09-01'),
                              ('FinalTest', '2025-09-01', '2026-09-01')]:
        chosen = [d for d in dates if pd.Timestamp(start) <= d < pd.Timestamp(end)]
        if chosen:
            result[stage] = chosen
    return result


def precompute(p, lagreturns, calendar=None, market=None, next_dates=None, context=None, strategies=None):
    """Weekly gross returns, traded weight and holdings for each strategy.

    Weights drift with realised returns between rebalances; traded weight is the
    sum of absolute differences between target and drifted weights.
    """
    rows, positions = [], []
    grouped = {d: g.sort_values(['prediction', 'ticker'], ascending=[False, True], kind='stable')
               for d, g in p.groupby('date', sort=True)}
    weeks = [(d, grouped.get(d, p.iloc[:0])) for d in (calendar if calendar is not None else sorted(grouped))]
    for strategy in (STRATEGIES if strategies is None else strategies):
        k = 5 if strategy.startswith('Top5') else 10
        previous, prior_returns, prior_cash, last_rebalance = {}, {}, 1., None
        for week_index, (date, g) in enumerate(weeks):
            growth = prior_cash + sum(w * (1 + prior_returns[t]) for t, w in previous.items())
            drift = {t: w * (1 + prior_returns[t]) / growth for t, w in previous.items()}
            selected = g
            if 'PositiveScore' in strategy:
                selected = selected[selected.prediction.gt(0)]
            if 'NoLoss' in strategy:
                lag = 4 if '4w' in strategy else 13
                selected = selected[selected.ticker.map(lagreturns[lag].loc[date]).ge(0)]
            if strategy in ['Top5_LowVol26w', 'Top5_RankRisk26w']:
                selected = selected[selected.ticker.map(context['vol'].loc[date]).notna()]
            rebalance_event, forced_exit_count = True, 0
            if strategy == 'Top5_Buffer10':
                keep = [t for t in g.head(10).ticker if t in previous]
                additions = [t for t in g.ticker if t not in keep][:k - len(keep)]
                weights = dict.fromkeys(keep + additions, 1 / k)
            elif strategy in ['Top5_Rebalance2w', 'Top5_Rebalance4w']:
                period = 2 if '2w' in strategy else 4
                if not len(g):
                    weights, last_rebalance = {}, None
                elif not previous or last_rebalance is None or week_index - last_rebalance >= period:
                    weights = dict.fromkeys(g.head(k).ticker, 1 / k)
                    last_rebalance = week_index
                else:
                    rebalance_event = False
                    usable = context['stock_returns'].loc[date]
                    weights = {t: w for t, w in drift.items() if pd.notna(usable.get(t, np.nan))}
                    forced_exit_count = len(drift) - len(weights)
            elif strategy == 'Top5_Trend26w' and not context['trend'].loc[date]:
                weights = {}
            else:
                selected = selected.head(k)
                if strategy in ['Top5_LowVol26w', 'Top5_RankRisk26w']:
                    scores = {t: (5 - rank if strategy == 'Top5_RankRisk26w' else 1) / max(float(context['vol'].loc[date, t]), 1e-8)
                              for rank, t in enumerate(selected.ticker)}
                    weights = capped_weights(scores, budget=len(selected) / k)
                elif strategy == 'Top5_RankWeighted':
                    weights = dict(zip(selected.ticker, [(5 - rank) / 15 for rank in range(len(selected))]))
                else:
                    weights = dict.fromkeys(selected.ticker, 1 / k)
            cash_weight = 1 - sum(weights.values())
            source_returns = dict(zip(g.ticker, np.expm1(g.target)))
            returns = {t: source_returns[t] if t in source_returns else float(context['stock_returns'].loc[date, t])
                       for t in weights}
            assert all(np.isfinite(list(returns.values())))
            assert -1e-12 <= cash_weight <= 1 + 1e-12
            traded = sum(abs(weights.get(t, 0) - drift.get(t, 0)) for t in set(weights) | set(drift))
            rows.append(dict(date=date, label_end_date=g.label_end_date.iloc[0] if len(g) else next_dates[date],
                             strategy=strategy, gross_return=sum(weights[t] * returns[t] for t in weights),
                             benchmark_return=np.expm1(g.target).mean() if len(g) else market[date],
                             traded_weight=traded, holdings=len(weights), cash_weight=cash_weight,
                             no_signal_cash_week=not len(g), rebalance_event=rebalance_event,
                             forced_exit_count=forced_exit_count))
            positions.extend(dict(date=date, strategy=strategy, ticker=t, weight=w, realized_return=returns[t])
                             for t, w in weights.items())
            previous, prior_returns, prior_cash = weights, returns, cash_weight
    return pd.DataFrame(rows), pd.DataFrame(positions)


def apply_cost(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Net return = (1 - traded weight x one-way cost) x (1 + gross) - 1."""
    out = frame.copy()
    out['transaction_cost_fraction'] = out.traded_weight * cost_bps / 10000
    out['net_return'] = (1 - out.transaction_cost_fraction) * (1 + out.gross_return) - 1
    return out
