# Data collection and cleaning reference

Reproduction starts from the supplied processed weekly data. Raw acquisition and reconstruction are outside this delivery scope. The scripts below document the upstream method; they are not steps required to reproduce the selected strategy or run the Ensemble training demonstration.

Run every script with the **project root as the working directory**; they read and write `data/`. They need network access (Yahoo Finance, SEC EDGAR, FRED) and an extra dependency. These are data-generation commands, not package checks. They write frozen-data paths, so run new acquisition experiments in a separate project copy:

```bash
pip install -r requirements-data.txt      # yfinance
export SEC_USER_AGENT="YourProject/1.0 your.name@example.com"   # SEC requires a contact e-mail
# Windows PowerShell: $env:SEC_USER_AGENT = "YourProject/1.0 your.name@example.com"
```

| Order | Script | Input → output |
|---|---|---|
| 1 | `fetch_and_clean.py` | Yahoo Finance → `data/Original_basic_data.csv`, `data/clean_basic_data.csv`, `data/audit/` |
| 2 | `build_technical_factors.py` | `clean_basic_data.csv` → `data/test_data_weekly.csv` (35 internal factors + target), `data/factor_definitions.csv` |
| 3 | `build_external_factors.py` | Weekly panel + SEC companyfacts (6 fundamental factors aligned to filing dates) + FRED (4 macro factors) → `data/test_45_factors_weekly.csv` |
| 4 | `collect_additional_40_staples.py` | Selects 40 external stocks not in the original list → `data/additional_40/*.csv` |
| 5 | `export_additional_40_excel.py` | Adds next-week open/close prices → `additional_40_five_factors_targets_final.xlsx` |

```bash
python data_processing/fetch_and_clean.py
python data_processing/build_technical_factors.py
python data_processing/build_external_factors.py
python data_processing/collect_additional_40_staples.py
python data_processing/export_additional_40_excel.py \
  --input data/additional_40/additional_40_weekly_5_factors.csv \
  --universe data/additional_40/additional_40_universe.csv \
  --daily data/additional_40/additional_40_daily_ohlcv.csv \
  --output data/additional_40/additional_40_five_factors_targets_final.xlsx
```

(Step 5's default paths point next to the script, so pass the arguments explicitly.)

## Notes

- `collect_sec_fundamentals.py` is the SEC helper module called by the historical acquisition steps 3 and 4.
- Keep new data-acquisition experiments separate from the supplied processed inputs.
- Downloaded data changes as sources are revised, so reruns are not guaranteed to match the frozen files in `data/` byte for byte; see `data/README.md`.

- The supplied scripts are later repaired versions; historical research files came from an earlier version. Rebuilding those sources is not part of this delivery. The documented reproduction uses the supplied weekly panels.
