"""Optional training experiment using the original training code and input.

New models are written to a fresh runs/ directory. This script never replaces
the frozen model files selected for the delivered historical inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import xgboost

import ranker_core as core
from prediction_helpers import training_window


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
from execution_status import recorded_attempt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='New output folder name under project runs/')
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full')
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error('--n-jobs must be positive')
    run_id = args.run_id or ('original_training_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        parser.error('Invalid run-id')
    if run_id.upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1,10)], *[f'LPT{i}' for i in range(1,10)]}:
        parser.error('Reserved run-id')

    run = ROOT/'runs'/run_id
    run.mkdir(parents=True, exist_ok=False)
    with recorded_attempt(run, 'training_manifest.json', metadata={'mode': 'optional_original_training'}) as status:
        status.update(_train(args, run))
    print('Training experiment saved:', run)
    print('Frozen inference models remain unchanged.')
    return 0


def _train(args, run):

    delivered = json.loads((PACKAGE/'saved_models_manifest.json').read_text('utf-8-sig'))
    for name in ['inputs/train66.csv', 'training_reference/ranker_core.py', 'training_reference/prediction_helpers.py']:
        if sha(PACKAGE/name) != delivered['files'][name]:
            raise ValueError(f'Original training source changed: {name}')
    frozen_before = {p.name: sha(p) for p in (PACKAGE/'saved_models').glob('*.json')}
    if len(frozen_before) != 45:
        raise ValueError('Expected 45 frozen snapshots')
    for name, value in frozen_before.items():
        if value != delivered['files'][f'saved_models/{name}']:
            raise ValueError(f'Frozen snapshot hash changed: {name}')

    data, _ = core.load_data(PACKAGE/'inputs/train66.csv')
    refits = list(range(core.FORMAL_START_WEEK, core.TOTAL_EXPECTED_WEEKS, core.REFIT_INTERVAL_WEEKS))
    if args.profile == 'smoke':
        refits = refits[-1:]
    models = run/'models'
    models.mkdir()
    records = []
    for component in core.COMPONENTS:
        for refit in refits:
            training = training_window(data, component, refit)
            model = core.fit_ranker(training, component, args.n_jobs)
            model_id = f'{component.component_id}_refit_{refit:04d}'
            target = models/f'{model_id}.json'
            model.save_model(target)
            records.append({
                'model_id': model_id, 'refit_week': refit,
                'training_rows': len(training), 'training_weeks': int(training.week_index.nunique()),
                'features': list(component.features), 'parameters': component.params,
                'random_state': core.RANDOM_SEED, 'n_jobs': args.n_jobs,
                'file': f'models/{target.name}', 'sha256': sha(target),
            })
            print('TRAIN', model_id, 'rows', len(training), flush=True)
    frozen_after = {p.name: sha(p) for p in (PACKAGE/'saved_models').glob('*.json')}
    if frozen_after != frozen_before:
        raise RuntimeError('The frozen inference snapshots were changed')
    manifest = {
        'status': 'passed', 'mode': 'optional_original_training', 'profile': args.profile,
        'new_models_fitted': len(records), 'production_model_files_changed': False,
        'production_model_source': 'external_validation/ensemble_ranker/saved_models/',
        'training_input_sha256': sha(PACKAGE/'inputs/train66.csv'),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgboost.__version__},
        'records': records,
        'note': 'New experiment weights are not automatically used by the default prediction command. Exact recreation of delivered weights is not guaranteed.',
    }
    return manifest


if __name__ == '__main__':
    raise SystemExit(main())
