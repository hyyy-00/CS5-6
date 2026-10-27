# Original-model training reference

`ranker_core.py` and `prediction_helpers.py` are unchanged files from the original model handoff. Their source hashes and the original `../inputs/train66.csv` hash are recorded in `../saved_models_manifest.json`.

`train_models.py` is a packaging wrapper that invokes the original functions and writes new weights only into a fresh project `runs/` folder. It checks that the 45 frozen inference files remain unchanged. It does not deploy its outputs or make the default predictor consume newly trained weights.

From the project root:

```bash
python external_validation/ensemble_ranker/training_reference/train_models.py --n-jobs 4
```

The full run fits 45 snapshots; `--profile smoke` fits only the latest three. Input factors, training label construction, component hyperparameters, refit interval and seed are defined in the original code. This experiment uses the original data and is separate from the corrected-data V1 pipeline.
