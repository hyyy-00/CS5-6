"""Regression checks for historical inference and the default CLI routing."""
import unittest
from unittest.mock import patch

import safecapital
import saved_ensemble


class SavedEnsembleTests(unittest.TestCase):
    def test_replay_matches_every_reference_score_without_training(self):
        with patch('xgboost.XGBRanker.fit', side_effect=AssertionError('Training is forbidden')):
            prediction, audit = saved_ensemble.replay()
        self.assertEqual((len(prediction), prediction.date.nunique()), (13517, 366))
        self.assertEqual(audit['models_loaded'], 45)
        self.assertEqual(audit['models_trained'], 0)
        self.assertEqual(audit['maximum_score_error'], 0.0)

    def test_default_cli_loads_saved_models(self):
        with patch('sys.argv', ['safecapital.py']), patch('saved_ensemble.main', return_value=0) as saved:
            self.assertEqual(safecapital.main(), 0)
        saved.assert_called_once_with([])

    def test_legacy_alias_also_loads_frozen_models(self):
        with patch('sys.argv', ['safecapital.py', '--legacy-ensemble']), patch('saved_ensemble.main', return_value=0) as saved:
            self.assertEqual(safecapital.main(), 0)
        saved.assert_called_once_with([])

    def test_v1_training_requires_explicit_switch(self):
        with patch('sys.argv', ['safecapital.py', '--retrain-v1', '--stage', 'check']), patch('pipeline.runner.main', return_value=0) as train:
            self.assertEqual(safecapital.main(), 0)
        train.assert_called_once_with(['--stage', 'check'])

    def test_model_hash_tampering_is_rejected(self):
        with patch('saved_ensemble.digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'input changed'):
                saved_ensemble.replay()


if __name__ == '__main__':
    unittest.main()
