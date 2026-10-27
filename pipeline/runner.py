from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import re
import sys
import traceback
from pathlib import Path

from . import ROOT, VERSION
from .contracts import ContractError, environment, output_manifest, read_json, require, sha, source_identity, validate_outputs, write_json

STAGES = ['prepare', 'train']
REQUIRED_OUTPUTS = {
    'prepare': ['train.csv', 'calendar.csv', 'data_audit.json'],
    'train': ['model_manifest.json'],
}


def validate_stage_artifacts(name, directory):
    for filename in REQUIRED_OUTPUTS[name]:
        path = directory / filename
        require(path.is_file() and path.stat().st_size > 0, f'{name}: required artifact missing/empty: {filename}')
    if name == 'train':
        records = read_json(directory / 'model_manifest.json')
        require(bool(records), 'Training produced no models')
        for item in records:
            path = (directory / item['file']).resolve()
            require(path.is_relative_to(directory.resolve()) and path.is_file(), 'Model artifact missing/unsafe')
            require(sha(path) == item['sha256'], 'Training model artifact hash mismatch')


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


def identity(options):
    from .inputs import preflight
    paths, missing = preflight()
    require(not missing, 'Required processed weekly data missing:\n' + '\n'.join(missing))
    return {'version': VERSION, 'options': options, 'environment': environment(),
            'source_files': source_identity(), 'input_files': {str(p.resolve()): sha(p) for p in paths}}


def check_dependencies(run, stages, manifest):
    for stage in stages:
        validate_outputs(run, stage, manifest['stages'].get(stage, {}))


def execute_stage(run, name, options, manifest):
    from . import inputs, learning
    handlers = {'prepare': inputs.prepare, 'train': learning.train}
    check_dependencies(run, STAGES[:STAGES.index(name)], manifest)
    require(identity(options) == manifest['identity'], 'Input/code/environment changed since run start')
    directory = run / name
    require(not directory.exists(), f'Stage directory already exists: {name}. Use a new run; no old outputs will be overwritten.')
    directory.mkdir()
    manifest['stages'][name] = {'status': 'running'}
    write_json(run / 'manifest.json', manifest)
    result = handlers[name](run, options)
    validate_stage_artifacts(name, directory)
    require(identity(options) == manifest['identity'], 'Input/code/environment changed during stage')
    manifest['stages'][name] = {'status': 'passed', 'result': result, 'outputs': output_manifest(directory)}
    write_json(run / 'manifest.json', manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ensemble Training Demonstration (V1): prepare supplied weekly data and fit the three components; no external prediction or backtest.')
    parser.add_argument('--profile', choices=['smoke', 'full'], default='full')
    parser.add_argument('--stage', choices=['check', 'prepare', 'all'], default='all')
    parser.add_argument('--run-id', help='New output directory name; existing runs are never overwritten')
    args = parser.parse_args(argv)
    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    run_id = args.run_id or ('ensemble_training_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id)), 'Invalid run-id')
    require(run_id.upper() not in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}, 'Reserved run-id')
    run = runs / run_id
    require(not run.exists(), 'Run already exists; choose a new run-id')
    run.mkdir()
    options = dict(profile=args.profile)
    manifest = {'version': VERSION, 'status': 'initializing', 'stages': {},
                'purpose': 'Ensemble training method demonstration only',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'requested_options': options}
    current = 'check'
    with (run / 'run.log').open('w', encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            manifest.update(identity=identity(options), status='partial')
            write_json(run / 'manifest.json', manifest)
            requested = [] if args.stage == 'check' else (['prepare'] if args.stage == 'prepare' else STAGES)
            for name in requested:
                current = name
                print('START', name, flush=True)
                execute_stage(run, name, options, manifest)
                print('PASS', name, flush=True)
            if requested == STAGES:
                check_dependencies(run, STAGES, manifest)
                manifest['status'] = 'passed'
            else:
                manifest['status'] = 'preflight_passed' if not requested else 'prepared'
            write_json(run / 'manifest.json', manifest)
            print('RESULT', manifest['status'], str(run / 'manifest.json'), flush=True)
            return 0
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            manifest['status'] = 'failed'
            manifest['failure'] = {'stage': current, 'error_type': type(error).__name__, 'message': str(error)}
            if current in STAGES:
                manifest['stages'][current] = {'status': 'failed'}
            write_json(run / 'manifest.json', manifest)
            traceback.print_exc()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return 2
