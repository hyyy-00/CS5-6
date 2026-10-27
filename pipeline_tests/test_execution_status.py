"""Regression tests for failed attempts after a previous successful result."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from execution_status import recorded_attempt, write_status
import saved_ensemble
import run_legacy_ensemble_backtest as legacy
import pandas as pd


def setUpModule():
    (ROOT/'runs').mkdir(parents=True, exist_ok=True)


class AttemptStatusTests(unittest.TestCase):
    def test_running_is_visible_until_success_commits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            path = Path(d)/'verification.json'
            with recorded_attempt(d, path.name) as state:
                self.assertEqual(json.loads(path.read_text('utf-8'))['status'], 'running')
                state.update(cases=4)
            self.assertEqual(json.loads(path.read_text('utf-8'))['status'], 'passed')

    def test_old_success_and_outputs_are_archived_on_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            directory = Path(d)
            write_status(directory/'verification.json', {'status': 'passed', 'cases': 4})
            (directory/'summary.csv').write_text('old result', 'utf-8')
            with self.assertRaisesRegex(ValueError, 'new failure'):
                with recorded_attempt(directory, 'verification.json', outputs=['summary.csv']):
                    raise ValueError('new failure')
            current = json.loads((directory/'verification.json').read_text('utf-8'))
            self.assertEqual(current['status'], 'failed')
            self.assertFalse(current['outputs_valid'])
            self.assertNotIn('cases', current)
            self.assertFalse((directory/'summary.csv').exists())
            old = list((directory/'previous_attempts').glob('*/verification.json'))
            self.assertEqual(len(old), 1)
            self.assertEqual(json.loads(old[0].read_text('utf-8'))['status'], 'passed')
            self.assertEqual(old[0].with_name('summary.csv').read_text('utf-8'), 'old result')

    def test_interruption_cannot_leave_passed_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            with self.assertRaises(KeyboardInterrupt):
                with recorded_attempt(d, 'manifest.json') as state:
                    state['status'] = 'passed'
                    raise KeyboardInterrupt()
            self.assertEqual(json.loads((Path(d)/'manifest.json').read_text('utf-8'))['status'], 'failed')

    def test_saved_prediction_input_failure_has_failed_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            base = Path(d)
            with patch.object(saved_ensemble, 'ROOT', base), patch.object(saved_ensemble, 'replay', side_effect=ValueError('bad model hash')):
                with self.assertRaisesRegex(ValueError, 'bad model hash'):
                    saved_ensemble.main(['--run-id', 'failed_input'])
            status = json.loads((base/'runs/failed_input/manifest.json').read_text('utf-8'))
            self.assertEqual((status['status'], status['stage']), ('failed', 'predict'))

    def test_backtest_failure_does_not_turn_prediction_pass_into_run_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            base = Path(d)
            frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-03']), 'ticker': ['A'], 'ensemble_ranking_score': [1.0]})
            with patch.object(saved_ensemble, 'ROOT', base), patch.object(saved_ensemble, 'replay', return_value=(frame, {'status': 'passed'})), patch.object(legacy, 'main', side_effect=ValueError('simulation failed')):
                with self.assertRaisesRegex(ValueError, 'simulation failed'):
                    saved_ensemble.main(['--run-id', 'failed_backtest'])
            status = json.loads((base/'runs/failed_backtest/manifest.json').read_text('utf-8'))
            self.assertEqual((status['status'], status['stage']), ('failed', 'backtest'))

    def test_existing_completed_run_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            base = Path(d)
            run = base/'runs/completed'
            run.mkdir(parents=True)
            path = run/'manifest.json'
            write_status(path, {'status': 'passed', 'previous_run': True})
            before = path.read_bytes()
            with patch.object(saved_ensemble, 'ROOT', base), patch.object(saved_ensemble, 'replay') as replay:
                with self.assertRaises(FileExistsError):
                    saved_ensemble.main(['--run-id', 'completed'])
            replay.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_legacy_input_failure_invalidates_old_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            directory = Path(d)
            write_status(directory/'verification.json', {'status': 'passed'})
            (directory/'summary.csv').write_text('old rows', 'utf-8')
            with patch.object(legacy, 'sha256', side_effect=ValueError('bad input')):
                with self.assertRaisesRegex(ValueError, 'bad input'):
                    legacy.main(output=directory)
            self.assertEqual(json.loads((directory/'verification.json').read_text('utf-8'))['status'], 'failed')
            self.assertFalse((directory/'summary.csv').exists())



if __name__ == '__main__':
    unittest.main()
