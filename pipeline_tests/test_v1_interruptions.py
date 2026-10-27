"""Test V1 interruption state with real runner I/O and no numerical work."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_safecapital_interruption_test_pipeline'


def forbidden_numerical_work(*args, **kwargs):
    raise AssertionError('This state-only test must never execute numerical work')


class V1InterruptionTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.context.enter_context(patch.dict(sys.modules))
        self.context.enter_context(patch.object(sys, 'path', list(sys.path)))
        spec = importlib.util.spec_from_file_location(
            PACKAGE, ROOT / 'pipeline/__init__.py',
            submodule_search_locations=[str(ROOT / 'pipeline')])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
        self.runner = importlib.import_module(f'{PACKAGE}.runner')
        self.context.enter_context(patch.object(self.runner, 'ROOT', self.root))
        self.context.enter_context(patch.object(self.runner, 'identity', side_effect=self.identity))

        # Import guards keep every numerical implementation unloaded. Only prepare
        # is replaced with the deliberate failure for each exercised stage.
        for name, handlers in {
            'inputs': ['prepare'], 'learning': ['train'],
        }.items():
            module = types.ModuleType(f'{PACKAGE}.{name}')
            for handler in handlers:
                setattr(module, handler, forbidden_numerical_work)
            sys.modules[module.__name__] = module
            setattr(package, name, module)
        self.inputs = package.inputs
        self.console = io.StringIO()
        self.context.enter_context(redirect_stdout(self.console))
        self.context.enter_context(redirect_stderr(self.console))

    def identity(self, options):
        return {'version': self.runner.VERSION, 'options': options,
                'environment': {}, 'source_files': {}, 'input_files': {}}

    def read_manifest(self, run_id):
        return json.loads((self.root / 'runs' / run_id / 'manifest.json').read_text('utf-8'))

    def interrupt_prepare(self, error, run_id):
        def interrupt(run, options):
            # The real execute_stage must have committed running before failure.
            current = json.loads((run / 'manifest.json').read_text('utf-8'))
            self.assertEqual(current['stages']['prepare']['status'], 'running')
            raise error

        with patch.object(self.inputs, 'prepare', side_effect=interrupt):
            return self.runner.main(['--run-id', run_id, '--stage', 'prepare'])

    def assert_failed_prepare(self, run_id, error_type, message):
        manifest = self.read_manifest(run_id)
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['stages']['prepare']['status'], 'failed')
        self.assertEqual(manifest['failure'], {
            'stage': 'prepare', 'error_type': error_type, 'message': message})

    def test_keyboard_interrupt_marks_running_prepare_failed_before_propagating(self):
        error = KeyboardInterrupt('stop preparation')
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.interrupt_prepare(error, 'keyboard_interrupt')
        self.assertIs(caught.exception, error)
        self.assert_failed_prepare('keyboard_interrupt', 'KeyboardInterrupt', 'stop preparation')

    def test_system_exit_marks_running_prepare_failed_and_preserves_exit_code(self):
        error = SystemExit(17)
        with self.assertRaises(SystemExit) as caught:
            self.interrupt_prepare(error, 'system_exit')
        self.assertIs(caught.exception, error)
        self.assertEqual(caught.exception.code, 17)
        self.assert_failed_prepare('system_exit', 'SystemExit', '17')

    def test_ordinary_exception_still_records_failure_and_returns_two(self):
        result = self.interrupt_prepare(ValueError('bad input fixture'), 'ordinary_error')
        self.assertEqual(result, 2)
        self.assert_failed_prepare('ordinary_error', 'ValueError', 'bad input fixture')

    def test_interrupted_training_preserves_prepared_artifacts(self):
        def prepare(run, options):
            for name in self.runner.REQUIRED_OUTPUTS['prepare']:
                (run / 'prepare' / name).write_text('fixture', encoding='utf-8')
            return {'fixture': True}
        learning = sys.modules[f'{PACKAGE}.learning']
        with patch.object(self.inputs, 'prepare', side_effect=prepare), patch.object(learning, 'train', side_effect=KeyboardInterrupt('training stopped')):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.main(['--run-id', 'interrupted_training'])
        after = self.read_manifest('interrupted_training')
        self.assertEqual(after['status'], 'failed')
        self.assertEqual(after['stages']['prepare']['status'], 'passed')
        self.assertEqual(after['stages']['train']['status'], 'failed')
        self.assertEqual((self.root / 'runs/interrupted_training/prepare/train.csv').read_text(), 'fixture')

    def test_existing_completed_run_is_not_overwritten(self):
        run = self.root / 'runs' / 'completed_run'
        run.mkdir(parents=True)
        path = run / 'manifest.json'
        path.write_bytes(b'{"status": "passed", "historical_fixture": true}\n')
        artifact = run / 'existing_result.txt'
        artifact.write_bytes(b'preserve completed artifact\n')
        before_manifest, before_artifact = path.read_bytes(), artifact.read_bytes()
        with self.assertRaises(self.runner.ContractError):
            self.runner.main(['--run-id', 'completed_run', '--stage', 'prepare'])
        self.assertEqual(path.read_bytes(), before_manifest)
        self.assertEqual(artifact.read_bytes(), before_artifact)


if __name__ == '__main__':
    unittest.main()
