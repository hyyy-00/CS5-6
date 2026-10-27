from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from . import ROOT, VERSION
from .contracts import load_module, read_json, require, sha, write_frame, write_json
from .inputs import FEATURES, read_csv


def implementations():
    return load_module('repro_ensemble_core', ROOT / 'external_validation/ensemble_ranker/code/ranker_core.py')


def specifications():
    core = implementations()
    return [dict(name=f'Model{i}_v1', window=spec.training_window_weeks, start=156,
                 features=list(spec.features), component=i - 1, params=spec.params)
            for i, spec in enumerate(core.COMPONENTS, 1)]


def ranked(frame):
    frame = frame.copy()
    for name in FEATURES:
        frame[f'csrank__{name}'] = frame.groupby('week_index')[name].rank(method='average', pct=True)
    return frame


def training_slice(train, spec, refit, signal_date):
    history = train[train.week_index.between(refit - spec['window'], refit - 1)].copy()
    require(history.week_index.nunique() == spec['window'], f'Incomplete {spec["name"]} training window')
    require(history.date.max() < signal_date, 'Training signals reach the prediction signal')
    require(pd.to_datetime(history.label_end_date).max() <= signal_date, 'Unobserved future training label')
    return history


def train(run, options):
    destination = run / 'train'
    (destination / 'models').mkdir()
    training = ranked(read_csv(run / 'prepare/train.csv'))
    training['label_end_date'] = pd.to_datetime(training.label_end_date)
    calendar = read_csv(run / 'prepare/calendar.csv')
    dates = dict(zip(calendar.week_index, calendar.date))
    total = len(calendar)
    core = implementations()
    records = []
    for spec in specifications():
        refits = list(range(spec['start'], total, 26))
        if options['profile'] == 'smoke':
            refits = refits[-1:]
        for refit in refits:
            history = training_slice(training, spec, refit, dates[refit])
            model = core.fit_ranker(history, core.COMPONENTS[spec['component']], 1)
            feature_columns = [f'csrank__{f}' for f in spec['features']]
            params = dict(core.COMPONENTS[spec['component']].params, random_state=5703, n_jobs=1)
            identifier = f'{spec["name"]}_refit_{refit:04d}'
            path = destination / 'models' / f'{identifier}.json'
            model.save_model(path)
            # New training must also survive save/load before predictions are accepted.
            x = history[feature_columns].to_numpy(dtype=np.float32)
            restored = XGBRanker(n_jobs=1)
            restored.load_model(path)
            np.testing.assert_array_equal(model.predict(x), restored.predict(x))
            params = {key: ('NaN_default' if isinstance(value, float) and not np.isfinite(value) else value)
                      for key, value in params.items() if value is not None}
            records.append(dict(model_id=identifier, model=spec['name'], refit_week=refit,
                                prediction_end_week=min(refit + 26, total) - 1,
                                refit_date=str(dates[refit].date()), training_weeks=spec['window'],
                                training_rows=len(history), training_tickers=int(history.ticker.nunique()),
                                training_start=str(history.date.min().date()), training_end=str(history.date.max().date()),
                                maximum_label_end=str(history.label_end_date.max().date()),
                                features=spec['features'], ranked_features=feature_columns, parameters=params,
                                file=f'models/{identifier}.json', sha256=sha(path),
                                training_input_sha256=sha(run / 'prepare/train.csv'),
                                provenance='newly fitted from this run; not historical saved-model replay', version=VERSION))
            print(f'TRAIN {identifier}: {len(history)} rows, {spec["window"]} weeks', flush=True)
    write_json(destination / 'model_manifest.json', records)
    return {'new_models_fitted': len(records), 'save_load_score_check': 'passed', 'historical_models_loaded': False,
            'purpose': 'Ensemble training method demonstration; no external predictions or backtest'}

