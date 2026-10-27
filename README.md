# SafeCapital

SafeCapital presents a weekly stock-ranking research method and one selected external portfolio strategy: the frozen three-component Ensemble with `Top5_NoLoss13w` on Common210 FullHistory.

**Reproduction in this project starts from the supplied processed weekly data. Raw data acquisition and reconstruction are outside the scope of this delivery.** The supplied data is the defined starting point for the research, model-training demonstration and strategy replay. No additional raw-data collection is required for these workflows.

## Two clearly separated workflows

| Workflow | Purpose | Command |
|---|---|---|
| Frozen Ensemble Replay | Load the 45 trained snapshots, reproduce predictions and verify the selected external strategy result | `python safecapital.py` |
| Ensemble Training Demonstration (V1) | Prepare supplied weekly training data, fit only the three Ensemble components, save models and verify save/load consistency | `python safecapital.py --retrain-v1 --profile full` |

V1 is the legacy identifier of the training demonstration, not the version of the whole SafeCapital project or the name of a second externally evaluated Ensemble. The training demonstration does not generate external predictions, run an external backtest or compare two Ensembles. Its new models stay in a separate run folder and never replace the default frozen models. The standalone external Ranker branches and their model/prediction/backtest artifacts have been removed.

The original four-family research stages (Ridge, Random Forest, XGBoost and XGBRanker) remain as methodology material. A research XGBRanker and an Ensemble component are distinct from the removed standalone external Ranker comparison branches.

## Why the trained models are frozen

The strategy result belongs to a specific trained Ensemble. Fixed training hyperparameters alone do not guarantee identical fitted trees. These models use random subsampling; even with a fixed seed, platform, library versions, threads and floating-point computation can affect tree fitting. Historical project comparisons observed different fitted trees and predictions across environments. This does not mean every repeated fit must differ.

Different learned trees can change scores, stock rankings, selected portfolios and strategy results. A changed file hash alone does not prove different predictions, because serialization or metadata can also differ. The package therefore verifies both artifact identity and numerical outputs.

We freeze the original 45 trained snapshots, record their SHA-256 hashes and load the date-appropriate snapshot for each historical signal. The replay regenerates all component and Ensemble predictions from the supplied factors, compares them exactly with the reference, recomputes the portfolio simulation and independently checks selection and accounting. Freezing removes retraining variability; matching data, preprocessing and trading rules are still required, and the checks must pass in the execution environment.

Retraining demonstrates how the components are fitted. Newly fitted files are separate experimental artifacts and are not used to establish the delivered strategy result.

## Installation

Use 64-bit Python 3.12 and an isolated environment. Run commands from the unpacked project root.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
```

If activation is unavailable, invoke the environment's Python executable directly. Installation requires network access; the supplied-data workflows run offline. The pinned dependencies are checked by `tools/check_environment.py`. On macOS, XGBoost may additionally need an OpenMP runtime. Numerical acceptance is determined by the run's prediction and account checks.

## Run the selected strategy

```bash
python safecapital.py
```

`python run.py` and `python safecapital.py --saved-ensemble` are equivalent. `--legacy-ensemble` remains a compatibility alias. Run one entry point for one replay. Use `--run-id NAME` to select a new output folder; existing run folders are not overwritten.

The replay loads 45 original XGBRanker snapshots (three components, 15 refits each). It ranks factor values within the external stock universe, computes component scores and weekly percentile scores, and averages the three percentiles with weights of one third. Ticker order resolves equal Ensemble scores. Each historical date uses its effective model snapshot, not a later refit. Future-return labels do not enter prediction or stock selection.

Prediction verification covers 13,517 stock-week rows over 366 signal weeks. The selected portfolio period uses Common210, covering 8,354 stock-week rows over 210 signal weeks from 2022-08-26 to 2026-08-28.

### Strategy rules

1. Calculate each stock's observable trailing 13-week price return on the signal date. The execution-price panel stores the following week's close, so the engine shifts that series back one row before constructing the historical filter.
2. Exclude negative trailing returns and stocks without sufficient valid 13-week history. Zero returns remain eligible.
3. Select up to five eligible stocks with the highest Ensemble scores.
4. Target 20% of opening account equity per selected stock. If fewer than five qualify, retain the unused allocation as cash. Integer-share rounding and transaction fees also leave residual cash.
5. Rebalance weekly using next-open price proxies, retain cash and mark holdings using next-close proxies. Initial capital is 100,000; fees are 10 basis points on each buy or sell. Share sizing includes fees and avoids negative cash. Liquidate the remaining holdings at the end of the simulation.

The external result is one FullHistory strategy case. No separate final-year annualised-return table or chart is provided, and the training demonstration has no external performance comparison.

| Model | Strategy | Weeks | Annualised return | Sharpe | Maximum drawdown | Final capital |
|---|---|---:|---:|---:|---:|---:|
| Frozen Ensemble | Top5_NoLoss13w | 210 | 10.4390% | 0.5434211 | -24.8781% | 149330.61 |

## Outputs and verification

This package is intended to present the research method and strategy result. Predictions, weekly capital and verification records are retained. Separate weekly selected-stock lists, individual filter-decision tables, order ledgers and per-stock holding ledgers are not required delivery outputs. Orders and holdings are computed in memory so that an independent verifier can check stock selection, integer-share sizing, fees, cash, holdings and aggregate metrics.

| Output in each replay folder | Contents |
|---|---|
| `saved_model_predictions.csv` | Weekly per-stock component model IDs, raw scores, percentiles and ranks, plus Ensemble score and rank; these are before the strategy filter |
| `prediction_verification.json` | Snapshot identities, prediction coverage and exact equality with the historical reference |
| `backtest/weekly_capital.csv` | Weekly capital, net return, cash, fees, number of holdings and share count |
| `backtest/summary.csv` | Aggregate metrics for the single selected strategy |
| `backtest/verification.json` | Metric agreement and independent selection/account checks |
| `backtest/README.md` | Strategy and result summary |
| `manifest.json` | Complete-run status and output hashes |

Prediction ranks are not an executed holdings list. Inspect `manifest.json`: a complete replay has `status: passed`, following both prediction and portfolio verification. Failed or interrupted attempts are recorded as failed. Runs are short, standalone executions; no checkpoint recovery or resume feature is provided. Start a fresh run after interruption.

A verified complete replay is retained under `FINAL_OUTPUTS/01_saved_model_replay/verified_replay/`, including predictions, weekly capital and verification records. Supporting configuration, factor definitions and snapshot/ranking audits are in `reports/ensemble/`. [Validation results](docs/VALIDATION.md) state the checks and their outcomes.

## Ensemble training demonstration

```bash
# Latest refit for each component: three fitted model files
python safecapital.py --retrain-v1 --profile smoke

# All 15 refits per component: 45 fitted model files
python safecapital.py --retrain-v1 --profile full
```

Choose one profile. V1 uses `data/reproducible/train66_corrected_complete.csv`: 66 stocks, 521 complete signal weeks. It validates weekly keys, factors and labels, constructs chronological windows of 156 / 52 / 52 weeks, fits the three components every 26 weeks, and saves JSON models. Each fitted model is reloaded and its predictions on the training matrix must match exactly. This is a serialization check, not an external performance evaluation.

Outputs are `prepare/train.csv`, `prepare/calendar.csv`, `prepare/data_audit.json`, `train/models/`, `train/model_manifest.json`, `manifest.json` and `run.log`. There are no external prediction, backtest or comparison stages. See [the training guide](docs/V1_RETRAINING.md).

The unchanged original training implementation and its historical 66-stock input are also retained in `external_validation/ensemble_ranker/training_reference/` and `inputs/train66.csv` for method provenance. V1's corrected 521-week training input and the original frozen model input are explicitly different data versions.

## Research method and project layout

| Location | Role |
|---|---|
| [data/](data/README.md) | Supplied processed data and its version definitions |
| [research/01_factor_analysis/](research/01_factor_analysis/README.md) | Construction definitions and screening of 45 factors |
| [research/02_ridge/](research/02_ridge/README.md) | Ridge research |
| [research/03_random_forest/](research/03_random_forest/README.md) | Random Forest research |
| [research/04_xgboost/](research/04_xgboost/README.md) | XGBoost regression research |
| [research/05_xgboost_ranker/](research/05_xgboost_ranker/README.md) | XGBRanker research |
| [research/06_evaluation_backtest/](research/06_evaluation_backtest/README.md) | Original research evaluation methodology and retained evidence |
| `external_validation/ensemble_ranker/` | Original frozen Ensemble, factors, reference predictions and training sources |
| `external_validation/shared_inputs/` | Processed external price panel for the selected replay |
| `pipeline/` | Training demonstration and independent replay verification helpers |
| [FINAL_OUTPUTS/](FINAL_OUTPUTS/README.md) | Selected strategy evidence and original research chapter |
| `data_processing/` | Historical acquisition/preparation source, retained as methodological reference outside the reproduction boundary |
| `runs/` | Fresh local runs, excluded from the delivery archive |

The original four-model research uses its supplied 522-week data. Those historical results belong to their documented research protocol, not to an additional external Ensemble comparison. Run research scripts only when a new research experiment is intended; their guides specify inputs and output locations.

## Checks

```bash
# File inventory and hashes only
python -B tools/check_release.py

# Dependency and processed-input readiness
python -B tools/check_environment.py

# Regression checks, including frozen inference and synthetic accounting
python -B -m unittest discover -s pipeline_tests -v
```

The default replay itself performs the full selected strategy verification. File-integrity checks and dependency checks have distinct purposes and do not replace numerical checks. Do not run with `python -O`, because the historical execution engine uses assertions.

## Interpretation and provenance

Frozen artifacts establish a reproducible model version; they do not establish future profitability. Full-sample factor screening and previously viewed research periods limit independent generalisation claims. Quote proxies do not certify actual fills, slippage, market impact, capacity or all corporate actions. Processed-input checks do not independently certify every financial-data publication time or historical revision.

See [NOTICE.md](NOTICE.md) for provenance and [CHANGELOG.md](CHANGELOG.md) for delivery changes.
