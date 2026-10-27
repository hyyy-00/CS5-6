"""Ridge: four feature sets, six calendar years, 26-calendar-week refits.

Standalone numpy/pandas program. Never selects parameters on holdout metrics.
Development selection additionally purges labels not realized before Validation.
Optional audit files are separate from the four-file delivery package.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'test_45_factors_weekly.csv'
PREDICTIONS = ROOT / 'research' / 'predictions'
ALPHAS = [0.01, 0.1, 1., 10., 100., 1000.]
REFIT_WEEKS = 26
ANCHORS = {'development': '2022-09-01', 'validation': '2024-09-01',
           'final_test': '2025-09-01'}
FIVE = ['revenue_growth_yoy', 'operating_profitability', 'cashflow_to_assets',
        'debt_to_assets', 'asset_growth_yoy']
SEVEN = ['revenue_growth_yoy', 'ret_120d', 'cashflow_to_assets', 'ret_1d',
         'ret_5d', 'close_location_value', 'body_range_ratio']
MACRO = ['macro_vix_level', 'macro_vix_change_5obs',
         'macro_yield_spread_10y_2y', 'macro_dgs10_change_5obs']
NON_FACTORS = {'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close',
               'volume', 'target', 'target2'}
NAMES = {'5': '5 factors', '7': '7 factors',
         '7_macro': '7 factors + 4 macro factors', '45': '45 factors'}
METRICS = ['IC', 'ICIR', 'Rank IC', 'Rank ICIR', 'Annualized Return',
           'Information Ratio', 'Top5 Excess', 'Positive Rank IC']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(source):
    raw = pd.read_csv(source, parse_dates=['date']).sort_values(
        ['ticker', 'date']).reset_index(drop=True)
    require({'date', 'ticker', 'target', 'adj_close'} <= set(raw), 'Missing columns')
    require(not raw[['date', 'ticker']].isna().any().any(), 'Missing date/ticker')
    require(not raw.duplicated(['date', 'ticker']).any(), 'Duplicate date/ticker')
    require(raw.ticker.nunique() == 66, 'Expected the agreed 66-stock universe')
    all_factors = [c for c in raw if c not in NON_FACTORS]
    require(len(all_factors) == 45, 'Expected exactly 45 factor columns')
    groups = {'5': FIVE, '7': SEVEN, '7_macro': SEVEN + MACRO, '45': all_factors}
    require(set(FIVE + SEVEN + MACRO) <= set(all_factors), 'Missing selected factors')
    for col in all_factors + ['target', 'adj_close']:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    raw['label_end_date'] = raw.date.map(dict(zip(dates[:-1], dates[1:])))
    next_prices = raw.set_index(['date', 'ticker']).adj_close.reindex(
        pd.MultiIndex.from_arrays([raw.label_end_date, raw.ticker])).to_numpy()
    valid = (raw.label_end_date.notna() & np.isfinite(next_prices) &
             np.isfinite(raw.target) & raw.adj_close.gt(0) & (next_prices > 0))
    error = float(np.max(np.abs(np.log(next_prices[valid] /
                  raw.loc[valid, 'adj_close']) - raw.loc[valid, 'target'])))
    require(error < 1e-10, 'Target differs from verified next-global-week log return')
    boundaries = {}
    for stage, anchor in ANCHORS.items():
        candidates = dates[dates >= pd.Timestamp(anchor)]
        require(len(candidates) > 0, f'No signal dates for {stage}')
        boundaries[stage] = candidates[0]
    raw['stage'] = np.select(
        [raw.date < boundaries['development'], raw.date < boundaries['validation'],
         raw.date < boundaries['final_test']],
        ['initial_history', 'development', 'validation'], default='final_test')
    excluded = raw.loc[~valid, ['date', 'ticker', 'target', 'label_end_date']].copy()
    panel = raw.loc[valid].copy()
    panel[all_factors] = panel[all_factors].replace([np.inf, -np.inf], np.nan)
    return raw, panel, groups, boundaries, excluded, error


def make_folds(panel, years):
    anchor = panel.date.min() + pd.DateOffset(years=years)
    eligible = panel.loc[panel.date >= anchor].copy()
    eligible['block'] = ((eligible.date - anchor).dt.days // (7 * REFIT_WEEKS)).astype(int)
    folds = []
    for block, part in eligible.groupby('block', sort=True):
        fit = part.date.min()
        folds.append({'fold_id': int(block), 'fit_date': fit,
                      'window_start': fit - pd.DateOffset(years=years),
                      'end': part.date.max()})
    require(bool(folds), 'Insufficient history for this window')
    return folds


def train_rows(panel, fold):
    train = panel.loc[(panel.date >= fold['window_start']) &
                      (panel.date < fold['fit_date']) &
                      (panel.label_end_date < fold['fit_date'])]
    require(not train.empty, 'Empty training sample')
    require(train.label_end_date.max() < fold['fit_date'], 'Training label leakage')
    return train


def prepared(train, predict, features):
    x = train[features].replace([np.inf, -np.inf], np.nan)
    med = x.median().fillna(0.)
    x = x.fillna(med)
    lo, hi = x.quantile(.01), x.quantile(.99)
    x = x.clip(lo, hi, axis=1)
    mean, scale = x.mean(), x.std(ddof=0)
    scale = scale.mask(scale.eq(0), 1.)
    xp = predict[features].replace([np.inf, -np.inf], np.nan).fillna(med)
    xp = xp.clip(lo, hi, axis=1)
    x, xp = ((x-mean)/scale).to_numpy(float), ((xp-mean)/scale).to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(xp).all(), 'Nonfinite processed data')
    return x, xp


def fit_ridge(x, y, alpha):
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x-xm, y-ym
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return coef, ym-float(xm @ coef)


def predict_fold(train, pred, features, alphas, fold):
    x, xp = prepared(train, pred, features)
    outputs = {}
    for alpha in alphas:
        coef, intercept = fit_ridge(x, train.target.to_numpy(float), alpha)
        out = pred[['date', 'ticker', 'label_end_date', 'stage', 'target']].copy()
        out = out.rename(columns={'target': 'actual'})
        out['prediction'] = xp @ coef + intercept
        out['alpha'], out['fold_id'], out['fit_date'] = alpha, fold['fold_id'], fold['fit_date']
        require(np.isfinite(out.prediction).all(), 'Invalid predictions')
        outputs[alpha] = out
    return outputs


def annualize(series):
    series = pd.Series(series).dropna().astype(float)
    return float(np.expm1(np.log1p(series).mean()*52)) if len(series) and (series > -1).all() else np.nan


def ratio(series):
    series = pd.Series(series).dropna()
    return float(series.mean()/series.std(ddof=1)) if len(series)>1 and series.std(ddof=1)>0 else np.nan


def weekly_metrics(pred):
    rows = []
    for date, g in pred.groupby('date', sort=True):
        g = g.sort_values('ticker').dropna(subset=['prediction', 'actual'])
        varied = len(g)>=2 and g.prediction.nunique()>1 and g.actual.nunique()>1
        ic = g.prediction.corr(g.actual) if varied else np.nan
        # Spearman = Pearson on average ranks, without a scipy dependency.
        ric = g.prediction.rank().corr(g.actual.rank()) if varied else np.nan
        ret = bench = np.nan
        if len(g) >= 5:
            ret = float(np.expm1(g.nlargest(5, 'prediction').actual).mean())
            bench = float(np.expm1(g.actual).mean())
        rows.append({'date': date, 'ic': ic, 'rank_ic': ric, 'top5_return': ret,
                     'benchmark_return': bench, 'excess_return': ret-bench, 'rows': len(g)})
    return pd.DataFrame(rows)


def metrics(pred):
    require(not pred.empty, 'No predictions to evaluate')
    w = weekly_metrics(pred)
    ric = w.rank_ic.dropna()
    return {'IC': float(w.ic.mean()), 'ICIR': ratio(w.ic),
            'Rank IC': float(ric.mean()), 'Rank ICIR': ratio(ric),
            'Annualized Return': annualize(w.top5_return),
            'Information Ratio': ratio(w.excess_return)*np.sqrt(52),
            'Top5 Excess': annualize(w.excess_return),
            'Positive Rank IC': float(ric.gt(0).mean()) if len(ric) else np.nan,
            'Weeks': int(len(w)), 'Rows': len(pred),
            'Signal Start': str(pred.date.min().date()),
            'Signal End': str(pred.date.max().date())}


def passes_gate(m):
    return (m['Rank IC'] > 0 and m['Rank ICIR'] >= .05 and
            m['Positive Rank IC'] >= .5 and np.isfinite(m['Top5 Excess']))


def run_group(panel, features, name, folds, boundaries):
    # Candidate predictions are generated only for label-realized Development rows.
    dev = panel.loc[(panel.stage == 'development') &
                    (panel.label_end_date < boundaries['validation'])]
    pieces = {a: [] for a in ALPHAS}
    audit_folds = []
    for fold in folds:
        train = train_rows(panel, fold)
        audit_folds.append({**fold, 'train_rows': len(train),
                            'max_train_label_end': train.label_end_date.max()})
        pred = dev.loc[(dev.date >= fold['fit_date']) & (dev.date <= fold['end'])]
        if pred.empty:
            continue
        out = predict_fold(train, pred, features, ALPHAS, fold)
        for a in ALPHAS:
            pieces[a].append(out[a])
    scores = []
    for a in ALPHAS:
        require(bool(pieces[a]), 'No Development predictions')
        m = metrics(pd.concat(pieces[a], ignore_index=True))
        scores.append({'feature_set': name, 'alpha': a, **m, 'passes_gate': passes_gate(m)})
        print(f'{name}: alpha={a:g}, RankIC={m["Rank IC"]:.6f}, '
              f'RankICIR={m["Rank ICIR"]:.6f}, excess={m["Top5 Excess"]:.6f}, '
              f'pass={passes_gate(m)}', flush=True)
    scores = pd.DataFrame(scores)
    eligible = scores.loc[scores.passes_gate].sort_values(
        ['Top5 Excess', 'Rank IC', 'Rank ICIR', 'alpha'],
        ascending=[False, False, False, True], kind='stable')
    if eligible.empty:
        row = {'Model Name': 'Ridge Rolling', 'Dataset': NAMES[name],
               'Stage': 'not_admitted', 'Alpha': np.nan, 'Status': 'no_candidate_passed'}
        row.update({metric: np.nan for metric in METRICS})
        return [row], scores, pd.DataFrame(), audit_folds
    chosen = eligible.iloc[0]
    alpha = float(chosen.alpha)
    print(f'{name}: frozen alpha={alpha:g}; refitting chronologically', flush=True)
    parts = []
    for fold in folds:
        train = train_rows(panel, fold)
        pred = panel.loc[(panel.date >= fold['fit_date']) & (panel.date <= fold['end'])]
        parts.append(predict_fold(train, pred, features, [alpha], fold)[alpha])
    all_pred = pd.concat(parts, ignore_index=True)
    all_pred['feature_set'] = name
    rows = []
    for stage in ANCHORS:
        part = all_pred.loc[all_pred.stage == stage]
        require(not part.empty, f'No {stage} predictions')
        row = {'Model Name': 'Ridge Rolling', 'Dataset': NAMES[name], 'Stage': stage,
               **metrics(part), 'Alpha': alpha, 'Status': 'selected',
               'Selection Weeks': int(chosen['Weeks']),
               'Selection Rows': int(chosen['Rows'])}
        rows.append(row)
    return rows, scores, all_pred, audit_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=HERE/'outputs'/'ridge_5_7_11_45_6y26w_results.csv')
    parser.add_argument('--mode', choices=['rolling'], default='rolling')
    parser.add_argument('--feature-set', choices=['all', *NAMES], default='all')
    parser.add_argument('--rolling-years', type=int, default=6)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-dir', type=Path, help='Optional new directory for reproducibility evidence')
    parser.add_argument('--predictions-dir', type=Path, default=PREDICTIONS,
                        help='Per-stock predictions of the selected alpha, read by 06_evaluation_backtest')
    args = parser.parse_args()
    require(args.rolling_years == 6, 'This aligned release uses six calendar years')
    raw, panel, groups, boundaries, excluded, error = load_data(args.input)
    chosen_groups = groups if args.feature_set == 'all' else {args.feature_set: groups[args.feature_set]}
    folds = make_folds(panel, args.rolling_years)
    print(f'Input: {len(raw)} rows, {raw.ticker.nunique()} stocks; '
          f'{len(panel)} verifiable labels; {len(excluded)} exclusions; label error={error:.3g}')
    print('Stages:', {k: str(v.date()) for k, v in boundaries.items()})
    print('Refit dates:', [str(f['fit_date'].date()) for f in folds])
    if args.check_only:
        return
    require(not args.output.exists(), f'Output already exists: {args.output}; specify a new --output')
    if args.audit_dir:
        require(not args.audit_dir.exists(), 'Audit directory already exists; specify a new directory')
    rows, score_tables, pred_tables, fold_records = [], [], [], []
    for name, features in chosen_groups.items():
        require(panel[features].notna().any(axis=1).all(), f'{name}: all factors missing in a row')
        r, scores, pred, fs = run_group(panel, features, name, folds, boundaries)
        rows.extend(r)
        score_tables.append(scores)
        if not pred.empty:
            pred_tables.append(pred)
        fold_records.extend({'feature_set': name, **f} for f in fs)
    table = pd.DataFrame(rows)
    base = ['Model Name', 'Dataset', 'Stage', *METRICS]
    table = table[base + [c for c in table if c not in base]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding='utf-8-sig', float_format='%.17g')
    if args.audit_dir:
        args.audit_dir.mkdir(parents=True)
        pd.concat(score_tables, ignore_index=True).to_csv(args.audit_dir/'development_candidates.csv', index=False)
        if pred_tables:
            pd.concat(pred_tables, ignore_index=True).to_csv(args.audit_dir/'predictions.csv', index=False,
                                                           float_format='%.17g')
        pd.DataFrame(fold_records).to_csv(args.audit_dir/'refit_schedule.csv', index=False)
        excluded.to_csv(args.audit_dir/'excluded_labels.csv', index=False)
        manifest = {'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'features': chosen_groups, 'alphas': ALPHAS, 'calendar_years': 6,
                    'refit_calendar_weeks': REFIT_WEEKS, 'raw_rows': len(raw),
                    'usable_rows': len(panel), 'excluded_rows': len(excluded),
                    'target_max_abs_error': error,
                    'stage_dates': {k: str(v.date()) for k, v in boundaries.items()},
                    'selection_label_cutoff_exclusive': str(boundaries['validation'].date()),
                    'selection_boundary_purge': True, 'python': platform.python_version(),
                    'numpy': np.__version__, 'pandas': pd.__version__}
        (args.audit_dir/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if pred_tables:
        args.predictions_dir.mkdir(parents=True, exist_ok=True)
        for pred in pred_tables:
            name = pred.feature_set.iloc[0]
            n = 11 if name == '7_macro' else int(name)
            out = pred[['date', 'ticker', 'actual', 'prediction']].rename(columns={'actual': 'target'})
            out.to_csv(args.predictions_dir / f'RIDGE-R6-F{n}.csv', index=False, float_format='%.17g')
    hp = table.drop_duplicates('Dataset')[['Dataset', 'Status', 'Alpha']].rename(columns={'Dataset': 'Factor set'})
    hp['Alpha'] = hp['Alpha'].map(lambda v: '-' if pd.isna(v) else f'{v:g}')
    print('\nSELECTED HYPERPARAMETER (alpha chosen in Development, frozen afterwards)')
    print(hp.to_string(index=False))
    base = pd.DataFrame({'Factor set': table['Dataset'], 'Stage': table['Stage']})
    first, second = base.copy(), base.copy()
    for col in ['IC', 'Rank IC', 'ICIR', 'Rank ICIR']:
        first[col] = table[col].map(lambda v: 'n.a.' if pd.isna(v) else f'{v:.4f}')
    first['Pos. Rank IC'] = table['Positive Rank IC'].map(lambda v: 'n.a.' if pd.isna(v) else f'{v:.2%}')
    for col, name in [('Annualized Return', 'Ann. Return'), ('Top5 Excess', 'Top5 Excess'),
                      ('Information Ratio', 'Info Ratio')]:
        fmt = (lambda v: 'n.a.' if pd.isna(v) else f'{v:.4f}') if col == 'Information Ratio' \
            else (lambda v: 'n.a.' if pd.isna(v) else f'{v:.2%}')
        second[name] = table[col].map(fmt)
    print('\nSTAGE RESULTS: PREDICTION METRICS')
    print(first.to_string(index=False))
    print('\nSTAGE RESULTS: TOP-5 PORTFOLIO METRICS (before costs)')
    print(second.to_string(index=False))
    print('(Full-precision decimals are saved in the results CSV.)')
    print('Results:', args.output)


if __name__ == '__main__':
    main()
