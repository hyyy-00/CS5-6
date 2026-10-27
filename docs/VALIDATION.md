# Validation results

Validation date: 2026-10-26. Environment: Python 3.12.10, with the pinned runtime dependencies recorded in [validation_results.json](validation_results.json).

## Frozen Ensemble strategy replay

The complete frozen-model prediction and selected-strategy replay passed. The delivery retains its output evidence at [verified_replay](../FINAL_OUTPUTS/01_saved_model_replay/verified_replay/).

- All 45 original frozen model snapshots and their recorded source/input hashes are unchanged.
- All 13,517 prediction rows over 366 weeks matched the reference exactly, including component scores, percentiles, model IDs and ranks.
- Exactly one strategy case was executed: Top5_NoLoss13w / Common210 FullHistory (210 weeks).
- Aggregate strategy metrics matched the reference with maximum absolute error 0.
- Independent selection/account verification passed 9,586 checks; maximum accounting error was 2.84217e-14.
- The top-level run manifest, prediction verification and portfolio verification all record `passed`.

## Training demonstration

The full V1 training demonstration passed preparation and training, fitting 45 snapshots across the three Ensemble components. Every fitted model passed an exact save/load score comparison on its training matrix. The run produced no external prediction or backtest stage. These experimental weights are not included as a second strategy model; the delivered strategy continues to use the original frozen snapshots.

## Regression and package checks

- Regression suite: 80 tests, 77 passed and 3 skipped because Windows symlink privileges were unavailable; no failures.
- Python syntax: 46 source files passed.
- Runtime dependency and processed-input readiness checks passed.
- Replay output hashes were checked against the passed run manifest.
- Release file inventory, hashes and ZIP CRC are checked when the delivery is built. Run `python -B tools/check_release.py` to verify the unpacked copy.

This record separates the verified external strategy from the retained original four-model research chapter. The chapter and its research verification records describe their own documented experiment. Reproduction starts from supplied processed weekly data; raw source reconstruction is outside the delivery boundary.
