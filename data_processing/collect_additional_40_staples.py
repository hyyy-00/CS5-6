"""Build a separate 40-stock US-traded consumer-equity extension panel.

This script never overwrites the original 100-stock files.  It keeps only
tickers that are absent from fetch_and_clean.TICKERS, have usable Yahoo
Finance daily OHLCV history and have all five requested point-in-time SEC
fundamental factors for at least two years of weekly observations.
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from collect_sec_fundamentals import (
    FACTS_URL,
    cik_by_ticker,
    collect_financial_factors,
    fetch_json,
    next_market_day,
    require_user_agent,
)
from fetch_and_clean import TICKERS as ORIGINAL_TICKERS


START_DATE = "2015-09-01"
END_DATE = "2026-09-01"
MIN_COMPLETE_WEEKS = 104
OUTPUT_DIR = Path("data") / "additional_40"
WORK_DIR = OUTPUT_DIR / "_work"
FIVE_FACTORS = [
    "revenue_growth_yoy",
    "operating_profitability",
    "cashflow_to_assets",
    "debt_to_assets",
    "asset_growth_yoy",
]

# Broad US-traded food, beverage, grocery, household, pet and consumer-service
# pool. The original 100-stock list already covers much of Consumer Staples;
# the wider candidate pool supplies enough non-overlapping securities while
# the code validates actual price history and SEC facts before selection.
CANDIDATES = [
    "LW", "BRID", "TR", "NATR", "NGVC", "VLGEA",
    "DLTR", "BJ", "OLLI", "GO", "CHWY", "REYN", "VITL", "AVO", "COCO", "SMPL",
    "VGR", "UTZ", "BRCC", "EWCZ", "ODD", "YSG", "BARK", "WOOF", "ZVIA", "BYND",
    "OTLY", "HFFG", "STKL", "SCL", "FIVE", "CAVA", "WING", "EAT", "TXRH", "MCD",
    "YUM", "QSR", "CMG", "SBUX", "PZZA", "JACK", "DIN", "CAKE", "SHAK", "RRGB",
    "PLAY", "NATH", "ARKR", "FWRG", "PBPB", "HIMS", "ELMD", "CROX",
    # Established US consumer retailers added to ensure a 40-stock extension
    # with sufficient history and complete US-GAAP factor coverage.
    "TJX", "ROST", "BURL", "KSS", "BBY", "WSM", "RH", "AEO", "ANF", "URBN", "GPS",
]


def clean_daily(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardise one ticker's batch-downloaded OHLCV data."""
    if frame.empty:
        return pd.DataFrame()
    frame = frame.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    frame["ticker"] = ticker
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in keep[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().loc[
        lambda x: (x[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1)
        & x["volume"].ge(0)
        & x["high"].ge(x[["open", "close", "low"]].max(axis=1))
        & x["low"].le(x[["open", "close", "high"]].min(axis=1))
    ].drop_duplicates(["date", "ticker"]).sort_values("date")


def download_daily_batch(tickers: list[str]) -> list[pd.DataFrame]:
    """Download in small batches so invalid symbols cannot stop the pool."""
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), 20):
        batch = tickers[start:start + 20]
        try:
            raw = yf.download(
                batch, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=False,
                actions=False, progress=False, threads=True, group_by="column",
            )
        except Exception as error:
            print(f"Batch {batch}: download failed ({error})")
            continue
        for ticker in batch:
            try:
                one = raw.xs(ticker, axis=1, level=1, drop_level=True) if isinstance(raw.columns, pd.MultiIndex) else raw
                one = clean_daily(one.dropna(how="all"), ticker)
                if len(one) >= 260:
                    frames.append(one)
                else:
                    print(f"{ticker}: insufficient valid daily history")
            except (KeyError, ValueError):
                print(f"{ticker}: price data unavailable")
    return frames


def weekly_last_day(daily: pd.DataFrame) -> pd.DataFrame:
    output = daily.sort_values(["ticker", "date"]).copy()
    output["_week"] = output["date"].dt.to_period("W-FRI")
    return output.groupby(["ticker", "_week"], group_keys=False).tail(1).drop(columns="_week")


def add_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    output = weekly.sort_values(["ticker", "date"]).copy()
    output["target"] = output.groupby("ticker")["adj_close"].transform(
        lambda values: np.log(values.shift(-1) / values)
    )
    next_date = output.groupby("ticker")["date"].shift(-1)
    consecutive = next_date.dt.to_period("W-FRI").eq(output["date"].dt.to_period("W-FRI") + 1)
    next_end = output.groupby("ticker")["date"].shift(-1)
    complete_next_week = next_market_day(next_end).dt.to_period("W-FRI").gt(next_end.dt.to_period("W-FRI"))
    complete_current_week = next_market_day(output["date"]).dt.to_period("W-FRI").gt(output["date"].dt.to_period("W-FRI"))
    output["target"] = output["target"].where(consecutive & complete_next_week & complete_current_week)
    output["target2"] = np.select(
        [
            output["target"] < -0.03,
            output["target"].ge(-0.03) & output["target"].lt(-0.01),
            output["target"].ge(-0.01) & output["target"].le(0.01),
            output["target"].gt(0.01) & output["target"].le(0.03),
        ],
        [0, 1, 2, 3], default=4,
    ).astype("int8")
    return output


def company_names() -> dict[str, str]:
    """Read ticker names from the SEC ticker mapping used by the factor collector."""
    payload = fetch_json("https://www.sec.gov/files/company_tickers_exchange.json")
    fields = {field: index for index, field in enumerate(payload["fields"])}
    return {
        str(row[fields["ticker"]]).upper(): str(row[fields["name"]])
        for row in payload["data"] if row[fields["ticker"]]
    }


def finalize_batches() -> None:
    """Combine completed small batches and write the three requested outputs."""
    weekly_files = sorted(WORK_DIR.glob("*_weekly.csv"))
    daily_files = sorted(WORK_DIR.glob("*_daily.csv"))
    if not weekly_files or not daily_files:
        raise RuntimeError("No completed batch files found. Run all collection batches first.")
    panel = pd.concat([pd.read_csv(file, parse_dates=["date"]) for file in weekly_files], ignore_index=True)
    daily = pd.concat([pd.read_csv(file, parse_dates=["date"]) for file in daily_files], ignore_index=True)
    panel = panel.drop_duplicates(["date", "ticker"])
    daily = daily.drop_duplicates(["date", "ticker"])
    valid_counts = panel.groupby("ticker")["date"].nunique()
    selected = valid_counts.loc[valid_counts.ge(MIN_COMPLETE_WEEKS)].sort_values(ascending=False).index.tolist()[:40]
    if len(selected) < 40:
        raise RuntimeError(f"Only {len(selected)} candidates satisfy the two-year complete-week requirement.")
    final = panel.loc[panel["ticker"].isin(selected)].sort_values(["ticker", "date"]).reset_index(drop=True)
    final_daily = daily.loc[daily["ticker"].isin(selected)].sort_values(["ticker", "date"]).reset_index(drop=True)
    if final.isna().any().any() or final.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Final panel contains missing values or duplicate date+ticker keys")
    names = company_names()
    universe = pd.DataFrame({"ticker": selected})
    universe["company_name"] = universe["ticker"].map(names).fillna("Name not returned by SEC mapping")
    universe["complete_weeks"] = universe["ticker"].map(valid_counts).astype(int)
    universe["first_week"] = universe["ticker"].map(final.groupby("ticker")["date"].min()).dt.strftime("%Y-%m-%d")
    universe["last_week"] = universe["ticker"].map(final.groupby("ticker")["date"].max()).dt.strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_csv(OUTPUT_DIR / "additional_40_universe.csv", index=False, encoding="utf-8-sig")
    final_daily.to_csv(OUTPUT_DIR / "additional_40_daily_ohlcv.csv", index=False, encoding="utf-8-sig")
    final.to_csv(OUTPUT_DIR / "additional_40_weekly_5_factors.csv", index=False, encoding="utf-8-sig")
    print(f"Selected 40 non-overlapping stocks. Complete weekly rows: {len(final)}")


def main() -> None:
    require_user_agent()
    if os.getenv("ADDITIONAL_FINALIZE") == "1":
        finalize_batches()
        return
    original = {ticker.upper() for ticker in ORIGINAL_TICKERS}
    candidates = list(dict.fromkeys(ticker.upper() for ticker in CANDIDATES if ticker.upper() not in original))
    batch = os.getenv("ADDITIONAL_BATCH", "").strip()
    if batch:
        batch_number = int(batch)
        candidates = candidates[batch_number * 10:(batch_number + 1) * 10]
        if not candidates:
            raise RuntimeError(f"Batch {batch_number} has no candidates")
    if not batch and len(candidates) < 40:
        raise RuntimeError("Candidate pool has fewer than 40 non-overlapping tickers")

    daily_frames = download_daily_batch(candidates)
    if not daily_frames:
        raise RuntimeError("No additional daily price data could be downloaded")
    daily = pd.concat(daily_frames, ignore_index=True)
    weekly = weekly_last_day(daily)

    print("Downloading SEC Company Facts and calculating the five requested factors ...")
    financial, log = collect_financial_factors(weekly[["date", "ticker"]], daily["date"])
    panel = weekly.merge(financial[["date", "ticker", *FIVE_FACTORS]], on=["date", "ticker"], how="left")
    panel = add_targets(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=[*FIVE_FACTORS, "target", "target2"]).reset_index(drop=True)
    valid_counts = panel.groupby("ticker")["date"].nunique()
    if batch:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(WORK_DIR / f"batch_{batch}_daily.csv", index=False, encoding="utf-8-sig")
        panel.to_csv(WORK_DIR / f"batch_{batch}_weekly.csv", index=False, encoding="utf-8-sig")
        print(f"Saved batch {batch}: {len(panel)} complete weekly rows; SEC status: {log['status'].value_counts().to_dict() if not log.empty else {}}")
        return
    selected = valid_counts.loc[valid_counts.ge(MIN_COMPLETE_WEEKS)].sort_values(ascending=False).index.tolist()[:40]
    if len(selected) < 40:
        raise RuntimeError(
            f"Only {len(selected)} candidates have at least {MIN_COMPLETE_WEEKS} complete weeks. "
            "Add candidates rather than weakening the data-quality rule."
        )

    final = panel.loc[panel["ticker"].isin(selected)].sort_values(["ticker", "date"]).reset_index(drop=True)
    final_daily = daily.loc[daily["ticker"].isin(selected)].sort_values(["ticker", "date"]).reset_index(drop=True)
    if final.isna().any().any() or final.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Final panel contains missing values or duplicate date+ticker keys")
    names = company_names()
    universe = pd.DataFrame({"ticker": selected})
    universe["company_name"] = universe["ticker"].map(names).fillna("Name not returned by SEC mapping")
    universe["complete_weeks"] = universe["ticker"].map(valid_counts).astype(int)
    universe["first_week"] = universe["ticker"].map(final.groupby("ticker")["date"].min()).dt.strftime("%Y-%m-%d")
    universe["last_week"] = universe["ticker"].map(final.groupby("ticker")["date"].max()).dt.strftime("%Y-%m-%d")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_csv(OUTPUT_DIR / "additional_40_universe.csv", index=False, encoding="utf-8-sig")
    final_daily.to_csv(OUTPUT_DIR / "additional_40_daily_ohlcv.csv", index=False, encoding="utf-8-sig")
    final.to_csv(OUTPUT_DIR / "additional_40_weekly_5_factors.csv", index=False, encoding="utf-8-sig")
    print(f"Selected 40 non-overlapping stocks. Complete weekly rows: {len(final)}")
    print(f"SEC status: {log['status'].value_counts().to_dict() if not log.empty else {}}")
    print(f"Saved files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
