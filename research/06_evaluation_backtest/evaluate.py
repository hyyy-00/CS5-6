"""Model evaluation and multi-strategy backtest for the four model families.

Inputs : research/predictions/{RF,RIDGE,XGB,RANKER}-R6-F{5,7,11,45}.csv
         (written by the training scripts in research/02-05)
         data/test_45_factors_weekly.csv (prices and labels)
Outputs: research/06_evaluation_backtest/outputs/

For every prediction series (plus a same-factor four-family consensus) this
runs 15 strategies x 14 windows x 3 one-way costs, an IC/RankIC evaluation,
and a rebalance-phase sensitivity test. Afterwards run verify.py.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import performance, prediction_metrics
from strategies import (COSTS_BPS, LABELS, RULES, STATEFUL, STRATEGIES, apply_cost,
                        market_inputs, precompute, windows)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'test_45_factors_weekly.csv'
PREDICTIONS = ROOT / 'research' / 'predictions'
FAMILIES = ['RF', 'RIDGE', 'XGB', 'RANKER']
FACTOR_SETS = [5, 7, 11, 45]
DELIVERED = {  # results tables shipped with each training script
    'RF': ROOT / 'research/03_random_forest/results/random_forest_5_7_11_45_6y26w_results.csv',
    'RIDGE': ROOT / 'research/02_ridge/results/ridge_5_7_11_45_6y26w_results.csv',
    'XGB': ROOT / 'research/04_xgboost/results/xgboost_5_7_11_45_6y26w_results.csv',
    'RANKER': ROOT / 'research/05_xgboost_ranker/results/xgboostranker_optimized_results.csv',
}
DATASETS = {5: '5 factors', 7: '7 factors', 11: '7 factors + 4 macro factors', 45: '45 factors'}
STAGE_WINDOWS = {'development': 'Development6y', 'validation': 'Validation', 'final_test': 'FinalTest'}
COLORS = ['#2563eb', '#f97316', '#16a34a', '#dc2626', '#9333ea', '#0891b2', '#be185d', '#a16207',
          '#475569', '#7c3aed', '#0f766e', '#c026d3', '#0284c7', '#be123c', '#65a30d', '#111827']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


# ----------------------------------------------------------------------------- inputs
def load_records(directory: Path):
    records = []
    for family in FAMILIES:
        for n in FACTOR_SETS:
            path = directory / f'{family}-R6-F{n}.csv'
            if path.is_file():
                records.append((path.stem, pd.read_csv(path, parse_dates=['date']), 'model_prediction', str(path)))
    if not records:
        raise SystemExit(f'No prediction files in {directory}. Run the training scripts in research/02-05 first.')
    return records


def prepare(records, raw, out_dir, write_audits=True):
    """Join predictions to verified next-week labels; drop rows without a realised label."""
    dates = sorted(raw.date.unique())
    next_dates = dict(zip(dates[:-1], dates[1:]))
    indexed = raw.set_index(['date', 'ticker']).adj_close
    label = raw[['date', 'ticker', 'target', 'adj_close']].rename(columns={'target': 'data_target'})
    label['label_end_date'] = label.date.map(next_dates)
    label['next_price'] = indexed.reindex(pd.MultiIndex.from_arrays([label.label_end_date, label.ticker])).to_numpy()
    ready, coverage, excluded, gaps = {}, [], [], []
    for name, p, origin, source in records:
        assert not p.duplicated(['date', 'ticker']).any(), name
        p = p.drop(columns=['label_end_date'], errors='ignore').merge(
            label, on=['date', 'ticker'], how='left', validate='one_to_one', indicator=True)
        assert p._merge.eq('both').all(), name
        assert np.isfinite(p[['prediction', 'target']].to_numpy()).all(), name
        assert np.max(abs(p.target - p.data_target)) < 1e-10, name
        known = p.label_end_date.notna() & np.isfinite(p.next_price)
        error = np.max(abs(np.log(p.loc[known, 'next_price'] / p.loc[known, 'adj_close']) - p.loc[known, 'target']))
        assert error < 1e-10, (name, error)
        bad = p.loc[~known, ['date', 'ticker']].copy()
        bad['experiment'] = name
        bad['reason'] = np.where(p.loc[~known, 'label_end_date'].isna(), 'terminal_next_week_unavailable',
                                 'next_global_week_stock_price_missing')
        excluded.append(bad)
        first, last, observed = p.date.min(), p.date.max(), set(p.date)
        missing = [d for d in dates if first <= d <= last and d not in observed]
        gaps.extend(dict(experiment=name, date=d, reason='source_has_no_prediction_on_this_week') for d in missing)
        coverage.append(dict(experiment=name, origin=origin, source=source, input_rows=len(p),
                             input_weeks=p.date.nunique(), input_first=p.date.min(), input_last=p.date.max(),
                             usable_rows=int(known.sum()), excluded_rows=int((~known).sum()),
                             signal_first=p.loc[known, 'date'].min(), signal_last=p.loc[known, 'date'].max(),
                             realization_last=p.loc[known, 'label_end_date'].max(),
                             weeks=p.loc[known, 'date'].nunique(), missing_source_weeks=len(missing),
                             label_max_error=error))
        q = p.loc[known].drop(columns=['_merge', 'data_target', 'adj_close', 'next_price'])
        ready[name] = q
        to_csv(q, out_dir / 'predictions' / f'{name}.csv')
    if write_audits:
        to_csv(pd.concat(excluded, ignore_index=True), out_dir / 'excluded_rows.csv')
        to_csv(pd.DataFrame(gaps, columns=['experiment', 'date', 'reason']), out_dir / 'missing_source_weeks.csv')
    return ready, pd.DataFrame(coverage)


def consensus(ready, raw, out_dir):
    """Same-factor four-family equal-weight percentile average (fixed in advance)."""
    derived, coverage, components = {}, [], []
    for n in FACTOR_SETS:
        names = [f'{f}-R6-F{n}' for f in FAMILIES]
        if not all(k in ready for k in names):
            continue
        joint = None
        for name in names:
            p = ready[name][['date', 'ticker', 'target', 'prediction']].copy()
            p[name] = p.groupby('date').prediction.rank(pct=True, method='average')
            p = p[['date', 'ticker', 'target', name]]
            if joint is None:
                joint = p
            else:
                joint = joint.merge(p, on=['date', 'ticker'], suffixes=('', '_check'), validate='one_to_one')
                assert np.allclose(joint.target, joint.target_check, atol=1e-12, rtol=0)
                joint = joint.drop(columns='target_check')
        # Rounding freezes floating-point ties before ranking; ties then break by ticker.
        joint['prediction'] = joint[names].mean(axis=1).round(12)
        name = f'CONSENSUS-R6-F{n}'
        made, cov = prepare([(name, joint[['date', 'ticker', 'target', 'prediction']], 'derived_consensus',
                              'equal-weight percentile average of ' + ','.join(names))], raw, out_dir, write_audits=False)
        derived.update(made)
        coverage.append(cov)
        components.append(joint.assign(experiment=name))
    if components:
        to_csv(pd.concat(components, ignore_index=True), out_dir / 'consensus_components.csv')
    return derived, coverage


def reconciliation(metrics_table, out_dir):
    """Compare re-computed stage metrics with the tables delivered by each training script."""
    rows = []
    for family, path in DELIVERED.items():
        if not path.is_file():
            continue
        supplied = pd.read_csv(path, encoding='utf-8-sig').rename(columns={'stage': 'Stage'})
        for n, dataset in DATASETS.items():
            name = f'{family}-R6-F{n}'
            for stage, window in STAGE_WINDOWS.items():
                s = supplied[(supplied.Dataset == dataset) & (supplied.Stage == stage)]
                m = metrics_table[(metrics_table.experiment == name) & (metrics_table.window == window)]
                if len(s) != 1 or len(m) != 1 or pd.isna(s['Rank IC'].iloc[0]):
                    continue
                rows.append(dict(family=family, Dataset=dataset, Stage=stage,
                                 rank_ic_supplied=float(s['Rank IC'].iloc[0]), rank_ic_replayed=float(m.rank_ic.iloc[0]),
                                 ic_supplied=float(s['IC'].iloc[0]), ic_replayed=float(m.ic.iloc[0])))
    table = pd.DataFrame(rows)
    if len(table):
        table['rank_ic_difference'] = table.rank_ic_replayed - table.rank_ic_supplied
        table['ic_difference'] = table.ic_replayed - table.ic_supplied
    to_csv(table, out_dir / 'source_metric_reconciliation.csv')
    return table


# ----------------------------------------------------------------------------- charts
def svg_chart(path, title, series, ylabel='NAV (initial = 1)', baseline=1.):
    """Dependency-free SVG line chart. ``series`` = [(label, dates, values), ...]."""
    width, height, left, right, top, bottom = 1450, 620, 70, 330, 50, 60
    all_dates = [pd.Timestamp(d) for _, ds, _ in series for d in ds]
    values = np.concatenate([np.asarray(v, float) for _, _, v in series] + [[baseline]])
    t0, t1 = min(all_dates), max(all_dates)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (hi - lo) * 0.05 or 0.05
    lo, hi = lo - pad, hi + pad
    span = max((t1 - t0).days, 1)
    x = lambda d: left + (pd.Timestamp(d) - t0).days / span * (width - left - right)
    y = lambda v: top + (hi - v) / (hi - lo) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="system-ui,sans-serif" font-size="12"><rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="28" font-size="16" font-weight="600">{html.escape(title)}</text>']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>'
                     f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#475569">{v:.2f}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
    for year in range(t0.year, t1.year + 1):
        d = pd.Timestamp(year=year, month=1, day=1)
        if t0 <= d <= t1:
            parts.append(f'<text x="{x(d):.1f}" y="{height - bottom + 20}" text-anchor="middle" fill="#475569">{year}</text>')
    parts.append(f'<text x="18" y="{top + (height - top - bottom) / 2}" transform="rotate(-90 18 {top + (height - top - bottom) / 2})" '
                 f'text-anchor="middle" fill="#475569">{html.escape(ylabel)}</text>')
    for i, (label, ds, vs) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        points = ' '.join(f'{x(d):.1f},{y(v):.1f}' for d, v in zip(ds, vs) if np.isfinite(v))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>')
        ly = top + 14 + i * 18
        parts.append(f'<line x1="{width - right + 16}" x2="{width - right + 40}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
                     f'<text x="{width - right + 46}" y="{ly + 4}">{html.escape(str(label))}</text>')
    parts.append('</svg>')
    Path(path).write_text(''.join(parts), encoding='utf-8')


def dashboard(summary, records, out_dir):
    data = summary[summary.cost_bps.eq(10)].copy()
    payload = json.dumps(dict(rows=json.loads(data.to_json(orient='records', date_format='iso')), charts=records),
                         ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model evaluation and backtest</title><style>body{font:15px system-ui;margin:30px;color:#172033;background:#f8fafc}select{padding:8px;margin:5px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;background:white;margin-top:16px}p{line-height:1.7}</style><h1>Model evaluation and backtest</h1><p>Six-year rolling training, 26-week refits, 15 strategies. This page shows one-way 10bps; the CSV files include 0/10/25bps. Every window starts from cash. A positive Ranker score does not imply a positive expected return; the benchmark is the gross equal-weight return of the universe. The full history includes the tuning period, and the Final Test has been viewed in earlier experiments.</p><label>Model <select id="model"></select></label><label>Window <select id="window"></select></label><table><thead><tr><th>Strategy</th><th>Net annualised return</th><th>Max drawdown</th><th>Sharpe</th><th>Mean traded weight</th><th>Mean cash weight</th></tr></thead><tbody id="table"></tbody></table><img id="nav"><img id="dd"><script>const DATA=PAYLOAD;const m=document.getElementById('model'),w=document.getElementById('window');const pct=x=>(100*x).toFixed(2)+'%';for(const x of [...new Set(DATA.rows.map(r=>r.experiment))])m.add(new Option(x,x));function windows(){w.innerHTML='';for(const x of DATA.charts.filter(r=>r.experiment===m.value))w.add(new Option(x.window,x.window));draw()}function draw(){let rows=DATA.rows.filter(r=>r.experiment===m.value&&r.window===w.value).sort((a,b)=>b.annualized_return-a.annualized_return);document.getElementById('table').innerHTML=rows.map(r=>`<tr><td>${r.strategy}</td><td>${pct(r.annualized_return)}</td><td>${pct(r.max_drawdown)}</td><td>${r.sharpe===null?'—':r.sharpe.toFixed(3)}</td><td>${pct(r.mean_traded_weight)}</td><td>${pct(r.mean_cash_weight)}</td></tr>`).join('');const c=DATA.charts.find(r=>r.experiment===m.value&&r.window===w.value);document.getElementById('nav').src=c.nav;document.getElementById('dd').src=c.drawdown}m.onchange=windows;w.onchange=draw;windows();</script></html>'''.replace('PAYLOAD', payload)
    (out_dir / 'strategy_dashboard.html').write_text(page, encoding='utf-8')


# ----------------------------------------------------------------------------- report
def pct(v):
    return '—' if pd.isna(v) else f'{v:.2%}'


def markdown_report(summary, metrics_table, phases, recon, out_dir):
    s = summary[summary.cost_bps.eq(10)]
    originals = sorted(e for e in s.experiment.unique() if not e.startswith('CONSENSUS'))
    lines = ['# Model evaluation and multi-strategy backtest (auto-generated)', '',
             f'Series: {s.experiment.nunique()} ({len(originals)} original models, {s.experiment.nunique() - len(originals)} consensus); '
             f'15 strategies, {s.window.nunique()} windows, one-way 0/10/25bps: {len(summary):,} main backtests and {len(phases):,} rebalance-phase tests.', '',
             '## Predictive power: Final Test', '', '|Model|IC|RankIC|RankICIR|Positive RankIC weeks|RMSE|', '|---|---:|---:|---:|---:|---:|']
    ft = metrics_table[metrics_table.window.eq('FinalTest')].sort_values('rank_ic', ascending=False)
    for r in ft.itertuples():
        lines.append(f'|{r.experiment}|{r.ic:.3f}|{r.rank_ic:.3f}|{r.rank_icir:.3f}|{r.positive_rank_ic:.2%}|'
                     f'{"—" if pd.isna(r.rmse) else f"{r.rmse:.3f}"}|')
    lines += ['', '## Equal-weight Top5 (one-way 10bps)', '', '|Model|Full-history net ann.|Validation net ann.|Final Test net ann.|Final Test max drawdown|', '|---|---:|---:|---:|---:|']
    top5 = s[s.strategy.eq('Top5')].pivot_table(index='experiment', columns='window', values='annualized_return')
    mdd = s[s.strategy.eq('Top5') & s.window.eq('FinalTest')].set_index('experiment').max_drawdown
    for e in top5.index:
        lines.append(f'|{e}|{pct(top5.loc[e].get("FullHistory"))}|{pct(top5.loc[e].get("Validation"))}|'
                     f'{pct(top5.loc[e].get("FinalTest"))}|{pct(mdd.get(e))}|')
    lines += ['', '## Best Final Test strategy per model (descriptive only; selection bias)', '', '|Model|Strategy|Net ann.|Max drawdown|Sharpe|', '|---|---|---:|---:|---:|']
    best = s[s.window.eq('FinalTest')].sort_values('annualized_return', ascending=False).groupby('experiment').head(1)
    for r in best.itertuples():
        lines.append(f'|{r.experiment}|{r.strategy}|{pct(r.annualized_return)}|{pct(r.max_drawdown)}|{r.sharpe:.3f}|')
    if len(recon):
        lines += ['', '## Reconciliation with the results delivered by the training scripts', '', '|Model family|Max RankIC difference|Max IC difference|', '|---|---:|---:|']
        for fam, g in recon.groupby('family'):
            lines.append(f'|{fam}|{g.rank_ic_difference.abs().max():.2e}|{g.ic_difference.abs().max():.2e}|')
    lines += ['', '## Interpretation limits', '',
              '- Same-close signals and same-close entry are research assumptions; slippage, market impact, liquidity and capacity are not modelled. Cash earns 0 and the benchmark is gross of costs.',
              '- The full history includes the Development tuning period; the Final Test has been viewed in earlier rounds and is not a new unseen test.',
              '- The best Final Test strategy is picked after comparing 15 strategies on the same data, so it carries multiple-comparison bias.',
              '- XGBoost and Random Forest use subsampling / parallel random numbers; other platforms or library versions can give different predictions.']
    (out_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions-dir', type=Path, default=PREDICTIONS)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs')
    parser.add_argument('--no-charts', action='store_true', help='Skip the per-model/window SVG charts')
    args = parser.parse_args()
    out = args.output
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    (out / 'charts').mkdir(exist_ok=True)

    raw = pd.read_csv(args.data, parse_dates=['date'])
    ready, coverage = prepare(load_records(args.predictions_dir), raw, out)
    derived, derived_cov = consensus(ready, raw, out)
    ready.update(derived)
    coverage = pd.concat([coverage, *derived_cov], ignore_index=True)
    status = [dict(experiment=f'{f}-R6-F{n}', available=f'{f}-R6-F{n}' in ready,
                   status='replayed' if f'{f}-R6-F{n}' in ready else 'no_candidate_passed_or_not_run')
              for f in FAMILIES for n in FACTOR_SETS]
    to_csv(pd.DataFrame(status), out / 'official_model_status.csv')

    prices = raw.pivot(index='date', columns='ticker', values='adj_close').sort_index()
    context = market_inputs(prices)
    dates_all = list(prices.index)
    next_dates = dict(zip(dates_all[:-1], dates_all[1:]))
    market = context['stock_returns'].mean(axis=1).to_dict()
    lag = {n: prices / prices.shift(n) - 1 for n in [4, 13]}

    metric_rows, summaries, weeklies, positions, window_positions, phase_rows, charts = [], [], [], [], [], [], []
    for name, p in ready.items():
        calendar = [d for d in dates_all if p.date.min() <= d <= p.date.max()]
        coverage.loc[coverage.experiment.eq(name), 'backtest_calendar_weeks'] = len(calendar)
        coverage.loc[coverage.experiment.eq(name), 'no_signal_cash_weeks'] = len(calendar) - p.date.nunique()
        full, pos = precompute(p, lag, calendar, market, next_dates, context)
        positions.append(pos.assign(experiment=name))
        for window, dates in windows(p, calendar).items():
            q = p[p.date.isin(dates)]
            m = prediction_metrics(q)
            if name.startswith(('RANKER', 'CONSENSUS')):  # ranking scores are not return forecasts
                m['mae'] = m['rmse'] = m['zero_prediction_rmse'] = np.nan
            metric_rows.append(dict(experiment=name, window=window, first=q.date.min(), last=q.date.max(), **m))
            w = full[full.date.isin(dates)].copy()
            if window != 'FullHistory':
                replay, wp = precompute(q, lag, dates, market, next_dates, context, STATEFUL)
                w = pd.concat([w[~w.strategy.isin(STATEFUL)], replay], ignore_index=True)
                window_positions.append(wp.assign(experiment=name, window=window))
            first = w.date.eq(w.date.min())  # every window starts from cash
            w.loc[first, 'traded_weight'] = 1 - w.loc[first, 'cash_weight']
            for cost in COSTS_BPS:
                z = apply_cost(w, cost)
                for strategy, g in z.groupby('strategy'):
                    summaries.append(dict(experiment=name, window=window, cost_bps=cost, strategy=strategy, **performance(g)))
                if cost == 10:
                    weeklies.append(z.assign(experiment=name, window=window, cost_bps=cost))
                    if not args.no_charts:
                        nav, dd = [], []
                        for strategy, g in z.groupby('strategy', sort=False):
                            g = g.sort_values('date')
                            ds = [g.date.iloc[0]] + list(g.label_end_date)
                            v = np.r_[1., np.cumprod(1 + g.net_return)]
                            nav.append((strategy, ds, v))
                            dd.append((strategy, ds, v / np.maximum.accumulate(v) - 1))
                        g = z[z.strategy.eq('Top5')].sort_values('date')
                        ds = [g.date.iloc[0]] + list(g.label_end_date)
                        nav.append(('EqualWeight gross', ds, np.r_[1., np.cumprod(1 + g.benchmark_return)]))
                        stem = f'{name}_{window}'
                        svg_chart(out / 'charts' / f'{stem}.svg', f'{name} | {window} | 10bps', nav)
                        svg_chart(out / 'charts' / f'{stem}_drawdown.svg', f'{name} | {window} | drawdown', dd, 'Drawdown', 0)
                        charts.append(dict(experiment=name, window=window, nav=f'charts/{stem}.svg',
                                           drawdown=f'charts/{stem}_drawdown.svg'))
            if window in ['FullHistory', 'Validation', 'FinalTest']:
                # Rebalance phase test: delay the first purchase by 0..period-1 weeks (cash meanwhile).
                benchmarks = {d: float(np.expm1(g.target).mean()) for d, g in q.groupby('date')}
                for period in [2, 4]:
                    strategy = f'Top5_Rebalance{period}w'
                    for phase in range(period):
                        ds = dates[phase:]
                        z, _ = precompute(q[q.date.isin(ds)], lag, ds, market, next_dates, context, [strategy])
                        if phase:
                            pad = pd.DataFrame([dict(date=d, label_end_date=next_dates[d], strategy=strategy, gross_return=0.,
                                                     benchmark_return=benchmarks.get(d, market[d]), traded_weight=0., holdings=0,
                                                     cash_weight=1., no_signal_cash_week=False, rebalance_event=False,
                                                     forced_exit_count=0) for d in dates[:phase]])
                            z = pd.concat([pad, z], ignore_index=True)
                        for cost in COSTS_BPS:
                            phase_rows.append(dict(experiment=name, window=window, strategy=strategy, cost_bps=cost,
                                                   initial_delay_weeks=phase, **performance(apply_cost(z, cost))))
        print('Evaluated', name, len(calendar), 'weeks', flush=True)

    summary = pd.DataFrame(summaries)
    metrics_table = pd.DataFrame(metric_rows)
    phases = pd.DataFrame(phase_rows)
    to_csv(coverage, out / 'coverage.csv')
    to_csv(metrics_table, out / 'model_evaluation.csv')
    to_csv(summary, out / 'strategy_comparison.csv')
    to_csv(pd.concat(weeklies, ignore_index=True), out / 'weekly_backtest.csv')
    to_csv(pd.concat(positions, ignore_index=True), out / 'positions.csv')
    to_csv(pd.concat(window_positions, ignore_index=True), out / 'stateful_window_positions.csv')
    to_csv(phases, out / 'phase_sensitivity.csv')
    to_csv(summary[summary.window.eq('FullHistory') & summary.cost_bps.eq(10)], out / 'full_history_comparison.csv')
    to_csv(pd.DataFrame(dict(date=prices.index, market_index=context['market_index'], risk_on=context['trend'])),
           out / 'market_trend.csv')
    (out / 'strategy_rules.json').write_text(json.dumps(dict(
        strategies=STRATEGIES, labels=LABELS, rules=RULES, cost_bps=COSTS_BPS, windows_start_cash=True,
        consensus='same-factor four-family percentile average; missing family means no consensus',
        positive_score='literal prediction > 0; Ranker scores do not imply positive expected return'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    recon = reconciliation(metrics_table, out)
    markdown_report(summary, metrics_table, phases, recon, out)
    if charts:
        dashboard(summary, charts, out)
    (out / 'run_manifest.json').write_text(json.dumps(dict(
        status='backtest_completed_pending_verification', model_series=len(ready), strategy_count=len(STRATEGIES),
        summary_rows=len(summary), phase_cases=len(phases), data_sha256=digest(args.data),
        raw_first=str(raw.date.min().date()), raw_last=str(raw.date.max().date()),
        signal_first=str(coverage.signal_first.min().date()), signal_last=str(coverage.signal_last.max().date()),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__), indent=2), encoding='utf-8')
    print(f'Done: {len(summary):,} backtests, {len(phases):,} phase cases -> {out}')


if __name__ == '__main__':
    main()
