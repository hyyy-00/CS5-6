# Supplied data and reproduction boundary

**Reproduction in this project starts from the supplied processed weekly data. Raw data acquisition and reconstruction are outside the scope of this delivery.**

| Input | Use |
|---|---|
| `test_45_factors_weekly.csv` | Original four-model research: 66 stocks and 522 signal weeks, 2016-09-02 to 2026-08-28 |
| `reproducible/train66_corrected_complete.csv` | Ensemble training demonstration: 66 stocks and 521 complete signal weeks, 2016-09-02 to 2026-08-21 |
| `reproducible/train_universe.csv` | Training universe validation |
| `additional_40/additional_40_five_factors_targets_final.xlsx` | Processed external factors, labels and price proxies |
| `external_validation/ensemble_ranker/inputs/test40.csv` (project-relative) | Original external factors used by frozen model inference |
| `external_validation/shared_inputs/inputs/test40_with_prices.csv` (project-relative) | External price panel used by the selected strategy replay |

Original Ensemble training inputs are retained under `external_validation/ensemble_ranker/inputs/`. Original research and corrected training data are separate versions; replacing one with the other changes the experiment.

Other daily/intermediate tables and `data_processing/` source are retained as method provenance. The acquisition scripts were revised after some historical inputs were produced. They are not prerequisites for reproducing the supplied strategy result, and no raw-source rebuild is required or exposed by the training demonstration. The old `raw_inputs/` folder is only a scope note.

Factors and labels in the supplied weekly panels define the reproduction input contract. The frozen prediction path excludes future labels. Prices are quote proxies; financial publication timing, historical revisions and actual trading fills remain interpretation limits rather than raw-reconstruction deliverables.
