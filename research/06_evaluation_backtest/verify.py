"""Independent check of evaluate.py outputs, re-derived from prices.

Rebuilds every model/window/strategy position, drift, traded weight and cost,
all summary metrics at every cost level, phase-zero equivalence, consensus
scores and the prediction statistics. Writes outputs/verification.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
checks, max_error = 0, 0.


def eq(a, b, label, tol=1e-9):
    global checks, max_error
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape, (label, x.shape, y.shape)
    assert np.array_equal(np.isnan(x), np.isnan(y)), label
    mask = np.isfinite(x) & np.isfinite(y)
    e = float(np.max(np.abs(x[mask] - y[mask]))) if mask.any() else 0.
    assert e < tol, (label, e)
    max_error = max(max_error, e)
    checks += x.size


def cap(scores, budget):
    """Capped proportional weights solved by bisection (independent of the water-fill)."""
    s = np.asarray(scores, float)
    budget = min(budget, .35 * len(s))
    if not len(s):
        return s
    lo, hi = 0., budget / max(s.min(), 1e-30) + 1
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(.35, mid * s).sum() < budget:
            lo = mid
        else:
            hi = mid
    return np.minimum(.35, hi * s)


def verify_portfolios(out, raw):
    prices = raw.pivot(index='date', columns='ticker', values='adj_close').sort_index()
    fwd, past = prices.shift(-1) / prices - 1, prices / prices.shift(1) - 1
    vol = past.rolling(26, min_periods=26).std(ddof=1)
    momentums = {n: prices / prices.shift(n) - 1 for n in [4, 13]}
    market_index = (1 + past.mean(axis=1).fillna(0)).cumprod()
    trend = (market_index > market_index.rolling(26, min_periods=26).mean()).fillna(False)
    preds = {p.stem: pd.read_csv(p, parse_dates=['date']) for p in (out / 'predictions').glob('*.csv')}
    for name, p in preds.items():
        eq(np.expm1(p.target), [fwd.loc[d, t] for d, t in zip(p.date, p.ticker)], name + ' direct price returns')
    weekly = pd.read_csv(out / 'weekly_backtest.csv', parse_dates=['date', 'label_end_date'])
    fullpos = pd.read_csv(out / 'positions.csv', parse_dates=['date'])
    winpos = pd.read_csv(out / 'stateful_window_positions.csv', parse_dates=['date'])
    fullgroups = {key: g for key, g in fullpos.groupby(['experiment', 'strategy'])}
    wingroups = {key: g for key, g in winpos.groupby(['experiment', 'window', 'strategy'])}
    predgroups = {name: {d: g.sort_values(['prediction', 'ticker'], ascending=[False, True], kind='stable')
                         for d, g in p.groupby('date')} for name, p in preds.items()}
    global checks
    for (name, window, strategy), g in weekly.groupby(['experiment', 'window', 'strategy']):
        g, pg = g.sort_values('date'), predgroups[name]
        if window != 'FullHistory' and strategy in ['Top5_Buffer10', 'Top5_Rebalance2w', 'Top5_Rebalance4w']:
            positions = wingroups.get((name, window, strategy), winpos.iloc[:0])
        else:
            positions = fullgroups.get((name, strategy), fullpos.iloc[:0])
        holdings = {d: dict(zip(z.ticker, z.weight)) for d, z in positions.groupby('date')}
        previous, cash, prev_r, last = {}, 1., {}, None
        for i, row in enumerate(g.itertuples()):
            d = row.date
            weights = holdings.get(d, {})
            growth = cash + sum(w * (1 + prev_r[t]) for t, w in previous.items())
            drift = {t: w * (1 + prev_r[t]) / growth for t, w in previous.items()}
            pool = pg.get(d, preds[name].iloc[:0])
            selected = pool
            k = 5 if strategy.startswith('Top5') else 10
            if 'PositiveScore' in strategy:
                selected = selected[selected.prediction > 0]
            if 'NoLoss' in strategy:
                momentum = momentums[4 if '4w' in strategy else 13].loc[d]
                selected = selected[selected.ticker.map(momentum) >= 0]
            if strategy in ['Top5_LowVol26w', 'Top5_RankRisk26w']:
                selected = selected[selected.ticker.map(vol.loc[d]).notna()]
            if strategy == 'Top5_Buffer10':
                kept = [t for t in pool.head(10).ticker if t in previous]
                target = dict.fromkeys(kept + [t for t in pool.ticker if t not in kept][:5 - len(kept)], .2)
            elif strategy in ['Top5_Rebalance2w', 'Top5_Rebalance4w']:
                period = 2 if '2w' in strategy else 4
                if not len(pool):
                    target, last = {}, None
                elif not previous or last is None or i - last >= period:
                    target, last = dict.fromkeys(pool.head(5).ticker, .2), i
                else:
                    target = {t: w for t, w in drift.items() if pd.notna(fwd.loc[d].get(t, np.nan))}
            elif strategy == 'Top5_Trend26w' and not trend.loc[d]:
                target = {}
            else:
                names = list(selected.head(k).ticker)
                if strategy in ['Top5_LowVol26w', 'Top5_RankRisk26w']:
                    scores = [(5 - j if strategy == 'Top5_RankRisk26w' else 1) / max(vol.loc[d, t], 1e-8) for j, t in enumerate(names)]
                    target = dict(zip(names, cap(scores, len(names) / 5)))
                elif strategy == 'Top5_RankWeighted':
                    target = {t: (5 - j) / 15 for j, t in enumerate(names)}
                else:
                    target = dict.fromkeys(names, 1 / k)
            assert set(weights) == set(target), (name, window, strategy, d)
            checks += 1
            eq([weights[t] for t in sorted(weights)], [target[t] for t in sorted(target)], 'weights')
            realized = {t: float(fwd.loc[d, t]) for t in weights}
            gross = sum(weights[t] * realized[t] for t in weights)
            # First week: drift is empty, so traded weight = sum of weights (start from cash).
            trade = sum(abs(weights.get(t, 0) - drift.get(t, 0)) for t in set(weights) | set(drift))
            eq([row.gross_return, row.traded_weight, row.cash_weight, row.transaction_cost_fraction, row.net_return],
               [gross, trade, 1 - sum(weights.values()), trade * .001, (1 - trade * .001) * (1 + gross) - 1], 'weekly arithmetic')
            previous, prev_r, cash = weights, realized, 1 - sum(weights.values())
    stats = pd.read_csv(out / 'strategy_comparison.csv')
    groups = {key: g.sort_values('date') for key, g in weekly.groupby(['experiment', 'window', 'strategy'])}
    for row in stats.itertuples():
        g = groups[row.experiment, row.window, row.strategy]
        r = (1 - g.traded_weight.to_numpy() * row.cost_bps / 10000) * (1 + g.gross_return.to_numpy()) - 1
        nav = np.r_[1., np.cumprod(1 + r)]
        v = np.std(r, ddof=1) * np.sqrt(52)
        eq([row.total_return, row.annualized_return, row.max_drawdown, row.volatility, row.sharpe, row.win_rate],
           [nav[-1] - 1, nav[-1] ** (52 / len(r)) - 1, (nav / np.maximum.accumulate(nav) - 1).min(), v,
            np.mean(r) * 52 / v if v > 0 else np.nan, np.mean(r > 0)], 'summary metrics')
    phases = pd.read_csv(out / 'phase_sensitivity.csv')
    zero = phases[phases.initial_delay_weeks.eq(0)].merge(stats, on=['experiment', 'window', 'strategy', 'cost_bps'],
                                                          suffixes=('_phase', '_main'), validate='one_to_one')
    for col in ['total_return', 'annualized_return', 'max_drawdown', 'sharpe']:
        eq(zero[col + '_phase'], zero[col + '_main'], 'phase zero ' + col)
    for name in [n for n in preds if n.startswith('CONSENSUS')]:
        factors = name.split('-F')[-1]
        joint, cols = preds[name][['date', 'ticker', 'prediction']], []
        for family in ['RF', 'RIDGE', 'XGB', 'RANKER']:
            x = preds[f'{family}-R6-F{factors}'].copy()
            x[family] = x.groupby('date').prediction.rank(pct=True, method='average')
            joint = joint.merge(x[['date', 'ticker', family]], on=['date', 'ticker'], validate='one_to_one')
            cols.append(family)
        eq(joint.prediction, joint[cols].mean(axis=1).round(12), 'consensus scores')
    return len(stats)


def verify_prediction_metrics(out):
    records = pd.read_csv(out / 'model_evaluation.csv', parse_dates=['first', 'last'])
    predictions = {p.stem: pd.read_csv(p, parse_dates=['date']) for p in (out / 'predictions').glob('*.csv')}
    for r in records.itertuples():
        p = predictions[r.experiment]
        p = p[(p.date >= r.first) & (p.date <= r.last)]
        ic, ric = [], []
        for _, g in p.groupby('date'):
            a, b = g.prediction.to_numpy(), g.target.to_numpy()
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                ic.append(np.corrcoef(a, b)[0, 1])
                ric.append(np.corrcoef(g.prediction.rank(method='average'), g.target.rank(method='average'))[0, 1])
            else:
                ic.append(np.nan)
                ric.append(np.nan)
        ic, ric = np.asarray(ic), np.asarray(ric)
        good = ric[np.isfinite(ric)]
        values = [len(p), len(ic), np.nanmean(ic), np.nanmean(ric), np.nanmean(ic) / np.nanstd(ic, ddof=1),
                  np.nanmean(ric) / np.nanstd(ric, ddof=1), np.mean(good > 0)]
        expected = [r.rows, r.weeks, r.ic, r.rank_ic, r.icir, r.rank_icir, r.positive_rank_ic]
        if not r.experiment.startswith(('RANKER', 'CONSENSUS')):
            values += [np.abs(p.prediction - p.target).mean(), np.sqrt(np.mean((p.prediction - p.target) ** 2)),
                       np.sqrt(np.mean(p.target ** 2))]
            expected += [r.mae, r.rmse, r.zero_prediction_rmse]
        eq(values, expected, f'prediction metrics {r.experiment} {r.window}', tol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'test_45_factors_weekly.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.data, parse_dates=['date'])
    cases = verify_portfolios(args.output, raw)
    verify_prediction_metrics(args.output)
    result = dict(status='passed', checks=checks, max_abs_difference=max_error, summary_cases=cases,
                  scope='direct price returns; every model/window/strategy position, drift and cost; all summary '
                        'metrics at all fees; phase-zero equivalence; consensus scores; IC/RankIC/ICIR/MAE/RMSE')
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    manifest_path = args.output / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(status='completed_and_verified', independent_checks=checks)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
