from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

from . import ROOT, VERSION


class ContractError(ValueError):
    """An explicit unsafe or unavailable input/output contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), 'utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def environment():
    require(__debug__, 'Optimized Python disables legacy engine assertions; run without -O')
    import numpy
    import pandas
    import scipy
    import sklearn
    import xgboost
    import exchange_calendars
    versions = {name: module.__version__ for name, module in [
        ('numpy', numpy), ('pandas', pandas), ('scipy', scipy),
        ('scikit-learn', sklearn), ('xgboost', xgboost), ('exchange-calendars', exchange_calendars)]}
    expected = {'numpy': '1.26.4', 'pandas': '2.2.2', 'scipy': '1.15.3',
                'scikit-learn': '1.6.1', 'xgboost': '3.4.1', 'exchange-calendars': '4.13.2'}
    require(versions == expected, f'Environment differs from controlled protocol: actual={versions}; expected={expected}. See requirements-repro.txt')
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                random_seed=5703, training_threads=1, version=VERSION)


def source_identity():
    files = [ROOT / 'safecapital.py', ROOT / 'run.py', ROOT / 'execution_status.py', ROOT / 'requirements-repro.txt'] + list((ROOT / 'pipeline').glob('*.py'))
    files += [ROOT / 'backtesting/external40/integer_backtest.py',
              ROOT / 'external_validation/ensemble_ranker/code/ranker_core.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def output_manifest(directory):
    return {p.relative_to(directory).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file()}


def validate_outputs(run, stage, record):
    require(record.get('status') == 'passed', f'Required stage is not passed: {stage}')
    require(bool(record.get('outputs')), f'{stage} has no output contract')
    actual = output_manifest(run / stage)
    require(actual == record.get('outputs'), f'{stage} outputs missing, changed or unexpected; create a new run')


def write_frame(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n',
                 compression='gzip' if str(path).endswith('.gz') else None)


def data_modules():
    location = str(ROOT / 'data_processing')
    if location not in sys.path:
        sys.path.insert(0, location)
    import collect_sec_fundamentals as financial
    import build_technical_factors as technical
    return financial, technical
