"""Historical inference with the delivered Ensemble models, without fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker
from execution_status import recorded_attempt


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'external_validation/ensemble_ranker'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_input(relative, manifest, package=PACKAGE):
    path = package / relative
    if digest(path) != manifest['files'][relative]:
        raise ValueError(f'Saved-model input changed: {relative}')
    return path


def replay(package=PACKAGE):
    """Infer using factor columns only; compare to reference after inference."""
    manifest = json.loads((package / 'saved_models_manifest.json').read_text('utf-8-sig'))
    config = json.loads(checked_input('model_configuration.json', manifest, package).read_text('utf-8-sig'))
    if len(config['components']) != 3 or config['ensemble']['weights'] != [1/3]*3:
        raise ValueError('Unexpected Ensemble configuration')
    windows = manifest['training_windows']
    expected_ids = {f'{c["component_id"]}_refit_{r:04d}' for c in config['components'] for r in range(156,522,26)}
    if len(windows) != 45 or {w['model_id'] for w in windows} != expected_ids:
        raise ValueError('Incomplete historical model manifest')
    for window in windows:
        if int(window['training_end_week']) >= int(window['refit_week']) or int(window['maximum_training_label_end_week']) > int(window['refit_week']):
            raise ValueError('Model metadata contains a future training boundary')
    features = list(dict.fromkeys(f for c in config['components'] for f in c['features']))
    reference = pd.read_csv(
        checked_input('predictions/stock_rankings.csv', manifest, package),
        parse_dates=['date'], float_precision='round_trip',
    )
    if reference.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate reference keys')
    calendar = reference[['week_index', 'date']].drop_duplicates().sort_values('week_index')
    if list(calendar.week_index) != list(range(156, 522)) or calendar.date.duplicated().any():
        raise ValueError('Historical calendar does not contain weeks 156 through 521')
    test = pd.read_csv(
        checked_input('inputs/test40.csv', manifest, package),
        usecols=['date', 'ticker', *features], parse_dates=['date'], float_precision='round_trip',
    )
    if test.duplicated(['date', 'ticker']).any() or not np.isfinite(test[features].to_numpy()).all():
        raise ValueError('Duplicate keys or invalid factors in external input')
    test = test.merge(calendar, on='date', how='inner', validate='many_to_one')
    test = test.sort_values(['week_index', 'ticker'], kind='stable').reset_index(drop=True)
    pd.testing.assert_frame_equal(
        test[['date', 'ticker']].sort_values(['date', 'ticker']).reset_index(drop=True),
        reference[['date', 'ticker']].sort_values(['date', 'ticker']).reset_index(drop=True),
    )
    if len(test) != 13517 or test.ticker.nunique() != 40:
        raise ValueError('Unexpected external prediction coverage')
    for feature in features:
        test[f'csrank__{feature}'] = test.groupby('week_index')[feature].rank(method='average', pct=True)

    result = test[['week_index', 'date', 'ticker']].copy()
    audit = []
    for i, component in enumerate(config['components'], 1):
        values = np.full(len(test), np.nan)
        ids = np.full(len(test), '', dtype=object)
        for refit in range(156, 522, 26):
            model_id = f'{component["component_id"]}_refit_{refit:04d}'
            path = checked_input(f'saved_models/{model_id}.json', manifest, package)
            mask = test.week_index.between(refit, min(refit+26, 522)-1)
            model = XGBRanker(n_jobs=1)
            model.load_model(path)
            columns = [f'csrank__{f}' for f in component['features']]
            scores = model.predict(test.loc[mask, columns].to_numpy(dtype=np.float32)).astype(float)
            values[mask.to_numpy()] = scores
            ids[mask.to_numpy()] = model_id
            audit.append({'model_id': model_id, 'rows': int(mask.sum()), 'sha256': digest(path)})
        if not np.isfinite(values).all():
            raise ValueError(f'Uncovered dates for component {i}')
        result[f'model_{i}_id'] = ids
        result[f'model_{i}_raw_score'] = values
        result[f'model_{i}_percentile'] = result.groupby('date')[f'model_{i}_raw_score'].rank(method='average', pct=True)
        result[f'model_{i}_predicted_rank'] = result.groupby('date')[f'model_{i}_raw_score'].rank(method='first', ascending=False).astype(int)
    result['ensemble_ranking_score'] = result[[f'model_{i}_percentile' for i in range(1, 4)]].mean(axis=1)
    result['ensemble_predicted_rank'] = result.groupby('date').ensemble_ranking_score.rank(method='first', ascending=False).astype(int)
    joined = result.merge(reference, on=['week_index', 'date', 'ticker'], validate='one_to_one', suffixes=('_generated', '_reference'))
    compare = ['ensemble_ranking_score', 'ensemble_predicted_rank']
    compare += [f'model_{i}_{field}' for i in range(1, 4) for field in ['id', 'raw_score', 'percentile', 'predicted_rank']]
    for column in compare:
        np.testing.assert_array_equal(joined[f'{column}_generated'], joined[f'{column}_reference'], err_msg=column)
    return result, {
        'status': 'passed', 'models_loaded': len(audit), 'models_trained': 0,
        'rows': len(result), 'weeks': int(result.date.nunique()),
        'maximum_score_error': 0.0, 'all_component_and_ensemble_predictions_identical': True,
        'future_label_columns_used_for_prediction': False,
        'manifest_sha256': digest(package / 'saved_models_manifest.json'),
        'historical_snapshots': audit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='New folder name under runs/')
    args = parser.parse_args(argv)
    run_id = args.run_id or ('saved_ensemble_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    import re
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run-id')
    if run_id.upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1,10)], *[f'LPT{i}' for i in range(1,10)]}:
        raise ValueError('Reserved run-id')
    run = ROOT / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    with recorded_attempt(run, 'manifest.json', metadata={'mode': 'frozen_model_inference'}) as status:
        status['stage'] = 'predict'
        result, audit = replay()
        result.to_csv(run / 'saved_model_predictions.csv', index=False, float_format='%.17g')
        (run / 'prediction_verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        status['stage'] = 'backtest'
        from run_legacy_ensemble_backtest import main as backtest
        backtest(frozen=result, output=run / 'backtest', provenance={
            'prediction_mode': 'loaded historical saved models; new inference, no training',
            'models_loaded': 45, 'prediction_verification': audit,
        })
        status.update(models_loaded=45, models_trained=0, stage='complete',
                      prediction_rows=len(result), prediction_weeks=int(result.date.nunique()),
                      backtest_cases=1,
                      files={p.relative_to(run).as_posix(): digest(p) for p in sorted(run.rglob('*'))
                             if p.is_file() and p != run/'manifest.json'})
    print('Saved models replayed exactly; results:', run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
