"""Download and clean the daily frequency historical data of US consumer staples stocks.
"""

from pathlib import Path
import hashlib
import json
import tempfile
import time
from typing import Any

import pandas as pd
import yfinance as yf

TICKERS = [
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB", "EL",
    "GIS", "SFM", "HSY", "MDLZ", "SJM", "MKC", "KHC", "STZ", "TAP", "BF-B",
    "KR", "SYY", "TSN", "HRL", "CPB", "CAG", "CHD", "CLX", "ADM", "KDP",
    "MNST", "CELH", "FIZZ", "COKE", "SAM", "MGPI", "WVVI", "BUD", "UL", "BTI",
    "DEO", "ABEV", "CCEP", "CCU", "FMX", "KOF", "NSRGY", "PETS", "BG", "DG",
    "TGT", "CENTA", "CASY", "IMKTA", "WMK", "UNFI", "USFD", "CHEF", "PFGC", "PPC",
    "FRPT", "CALM", "FLO", "POST", "NOMD", "BGS", "JJSF", "JBSS", "RBGLY", "INGR",
    "DAR", "CRESY", "HAIN", "LWAY", "SENEA", "SEB", "HEINY", "RMCF", "COTY", "ELF",
    "EPC", "ENR", "HELE", "NWL", "WDFC", "NUS", "USNA", "HLF", "MED", "TPB",
    "UVV", "ANDE", "DANOY", "CENT", "IPAR", "SPB", "ASBFY", "JVA", "REED", "AGRO",
]

START_DATE = "2015-09-01"
END_DATE = "2026-09-01"
DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "Original_basic_data.csv"
CLEAN_FILE = DATA_DIR / "clean_basic_data.csv"
AUDIT_DIR = DATA_DIR / "audit"
MAX_RETRIES = 3


def download_one_ticker(ticker: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download a stock; When a failure occurs, retry at most three times and return an auditable download record."""
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # auto_adjust=False returns both Close (raw close) and Adj Close (adjusted close).
            frame = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if frame.empty:
                last_error = "No data returned"
            else:
                # Some versions of yfinance return multiple layers of column names; When downloading a single code, only the name of the price field is retained.
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                frame = frame.reset_index()
                frame["ticker"] = ticker
                return frame, {
                    "ticker": ticker, "status": "success", "attempts": attempt,
                    "rows_downloaded": len(frame), "message": "",
                }
        except Exception as error:  # yfinance/network errors should not stop the full batch.
            last_error = str(error)

        if attempt < MAX_RETRIES:
            time.sleep(attempt)  # 1 second, then 2 seconds before retrying.

    return pd.DataFrame(), {
        "ticker": ticker, "status": "failed", "attempts": MAX_RETRIES,
        "rows_downloaded": 0, "message": last_error,
    }


def download_ticker_batch(tickers: list[str]) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """One request downloads the entire stock pool, reducing network requests and shortening execution time."""
    try:
        batch = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as error:
        # When batch requests are abnormal, they are handled by downloading and retrying them one by one.
        return [], [{"ticker": ticker, "status": "batch_error", "attempts": 0,
                     "rows_downloaded": 0, "message": str(error)} for ticker in tickers]

    frames: list[pd.DataFrame] = []
    log: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            frame = batch[ticker].dropna(how="all").reset_index()
            if frame.empty:
                raise ValueError("No data returned")
            frame["ticker"] = ticker
            frames.append(frame)
            log.append({"ticker": ticker, "status": "success", "attempts": 1,
                        "rows_downloaded": len(frame), "message": "batch download"})
        except (KeyError, ValueError) as error:
            log.append({"ticker": ticker, "status": "missing_from_batch", "attempts": 1,
                        "rows_downloaded": 0, "message": str(error)})
    return frames, log


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carry out the most basic and transparent data cleaning and generate quality reports aggregated by stocks."""
    required = ["Date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing fields in the download result: {sorted(missing_columns)}")

    data = raw[required].copy()
    data.columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Convert the content that cannot be parsed into missing values for unified processing.
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Delete records that cannot be used for the calculation of OHLCV factors or yields; Do not fill in with future values.
    original_rows_by_ticker = data.groupby("ticker").size().rename("rows_before_cleaning")
    data = data.dropna(subset=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    data = data.drop_duplicates(subset=["date", "ticker"], keep="last")
    data = data[
        (data[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1)
        & (data["volume"] >= 0)
        & (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
    ]
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Do not fill in the missing prices with future values to avoid leaking future information to subsequent models.
    # Retain the gap; The quality report will help users decide whether to exclude stocks or supplement data sources.
    data["daily_return"] = data.groupby("ticker")["adj_close"].pct_change(fill_method=None)

    # Calculate the coverage rate based on the actual trading days that have occurred in this download. This will not miscount the holidays of the US exchanges as missing.
    expected_trading_days = data["date"].drop_duplicates().sort_values()
    quality = (
        data.groupby("ticker", as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            rows=("date", "size"),
            missing_adjusted_close=("adj_close", lambda s: int(s.isna().sum())),
            zero_volume_days=("volume", lambda s: int((s == 0).sum())),
        )
    )
    quality["expected_trading_days"] = len(expected_trading_days)
    quality["coverage_ratio"] = (quality["rows"] / quality["expected_trading_days"]).round(4)
    quality = quality.merge(original_rows_by_ticker, on="ticker", how="left")
    quality["rows_removed_by_cleaning"] = quality["rows_before_cleaning"] - quality["rows"]
    quality = quality.drop(columns="rows_before_cleaning")
    quality = quality.sort_values("ticker").reset_index(drop=True)
    return data, quality


def main() -> None:
    """Download and save the original data in sequence, and then output the cleaned data and quality report."""
    DATA_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(Path(tempfile.gettempdir()) / "safecapital_yfinance_cache")
    print(f"Downloading in batches {len(TICKERS)} ...")
    downloaded, batch_log = download_ticker_batch(TICKERS)
    missing_tickers = [entry["ticker"] for entry in batch_log if entry["status"] != "success"]
    download_log = [entry for entry in batch_log if entry["status"] == "success"]

    # Only retry each stock that was missed in the batch request one by one, taking into account both speed and success rate.
    for ticker in missing_tickers:
        print(f"Trying again {ticker} ...")
        frame, log_entry = download_one_ticker(ticker)
        download_log.append(log_entry)
        if frame.empty:
            print(f"warning: {ticker} download failed: {log_entry['message']}")
        else:
            downloaded.append(frame)
        time.sleep(0.3)

    if not downloaded:
        raise RuntimeError("No data was downloaded. Please check the network, code and date range.")

    raw = pd.concat(downloaded, ignore_index=True)
    raw.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    clean, quality = clean_prices(raw)
    clean.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
    # Keep machine-readable provenance required for later factor audits.
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(download_log).to_csv(AUDIT_DIR / "price_download_log.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(AUDIT_DIR / "price_quality_report.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "tickers": TICKERS,
        "ticker_universe_sha256": hashlib.sha256(",".join(TICKERS).encode("utf-8")).hexdigest(),
        "raw_rows": len(raw),
        "clean_rows": len(clean),
    }
    (AUDIT_DIR / "price_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"The original data has been saved: {RAW_FILE}")
    print(f"Cleaning data has been saved: {CLEAN_FILE}")
    print("\nDownload record: ")
    print(pd.DataFrame(download_log).to_string(index=False))
    print("\nData Quality Report:")
    print(quality.to_string(index=False))



if __name__ == "__main__":
    main()
