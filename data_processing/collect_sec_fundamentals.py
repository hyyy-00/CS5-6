"""Point-in-time SEC financial-factor collection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
CIK_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_REQUEST_TIMEOUT_SECONDS = 15
MAX_REQUEST_ATTEMPTS = 3
SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.2  # 5 requests/second, below SEC's 10 rps fair-access limit.

FUNDAMENTAL_DEFINITIONS = [
    ("revenue_growth_yoy", "Quarter revenue / revenue from the same fiscal quarter last year - 1"),
    ("gross_margin_change_yoy", "Current TTM gross margin minus TTM gross margin four quarters earlier"),
    ("operating_profitability", "TTM operating income / average assets versus the same quarter last year"),
    ("cashflow_to_assets", "TTM operating cash flow / average assets versus the same quarter last year"),
    ("debt_to_assets", "Interest-bearing debt / total assets at the reporting date"),
    ("asset_growth_yoy", "Current total assets / total assets four quarters earlier - 1"),
]
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "assets": ["Assets"],
    "debt_total": ["LongTermDebtAndCurrent", "LongTermDebtAndShortTermBorrowings"],
    "debt_short": ["ShortTermBorrowings", "ShortTermDebt", "LongTermDebtCurrent", "CurrentPortionOfLongTermDebt"],
    "debt_long": ["LongTermDebtNoncurrent", "LongTermDebt"],
}


def require_user_agent() -> None:
    user_agent = os.getenv("SEC_USER_AGENT", "")
    if "@" not in user_agent or "your.email" in user_agent.lower():
        raise RuntimeError("Please set SEC_USER_AGENT to a contact e-mail, e.g. $env:SEC_USER_AGENT = 'SafeCapitalCoursework/1.0 your.name@example.com'")


def sec_user_agent() -> str:
    """Read the user agent at request time so the caller's environment is authoritative."""
    require_user_agent()
    return os.environ["SEC_USER_AGENT"]


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON SEC endpoint with bounded retries and fair-access pacing."""
    last_error: Exception | None = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            request = Request(url, headers={"User-Agent": sec_user_agent(), "Accept": "application/json"})
            with urlopen(request, timeout=SEC_REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read()
                encoding = response.headers.get_content_charset() or "utf-8"
            decoded = json.loads(payload.decode(encoding))
            if not isinstance(decoded, dict):
                raise ValueError("SEC response is not a JSON object")
            time.sleep(SEC_MIN_REQUEST_INTERVAL_SECONDS)
            return decoded
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                time.sleep(max(SEC_MIN_REQUEST_INTERVAL_SECONDS, 1.5 * (attempt + 1)))
    raise RuntimeError(f"SEC request failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}")


def cik_by_ticker() -> dict[str, int]:
    payload = fetch_json(CIK_URL)
    position = {name: index for index, name in enumerate(payload["fields"])}
    return {str(row[position["ticker"]]).upper(): int(row[position["cik"]]) for row in payload["data"] if row[position["ticker"]]}


def fact_rows(facts: dict[str, Any], tags: list[str], *, duration: bool) -> pd.DataFrame:
    """Read earliest-filed standard US-GAAP facts and retain source tags."""
    frames: list[pd.DataFrame] = []
    concepts = facts.get("facts", {}).get("us-gaap", {})
    for priority, tag in enumerate(tags):
        records = concepts.get(tag, {}).get("units", {}).get("USD", [])
        if records:
            frame = pd.DataFrame(records)
            frame["source_tag"] = tag
            frame["tag_priority"] = priority
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data["form"].isin(["10-Q", "10-K"])].copy()
    for name in ("start", "end", "filed"):
        data[name] = pd.to_datetime(data[name], errors="coerce") if name in data else pd.NaT
    data["val"] = pd.to_numeric(data["val"], errors="coerce")
    data = data.dropna(subset=["end", "filed", "val"])
    # A filing cannot make a period public before that period has ended.  This
    # also removes malformed duplicate observations before choosing a version.
    data = data.loc[data["filed"] >= data["end"]]
    if duration:
        data = data.dropna(subset=["start"])
    return data.sort_values(["filed", "tag_priority"]).drop_duplicates(["source_tag", "start", "end", "form"], keep="first")


def _period_facts(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return facts
    data = facts.copy()
    data["days"] = (data["end"] - data["start"]).dt.days + 1
    data = data.loc[data["days"].between(20, 400)]
    # Prefer the earliest contemporaneous filing, not a later restatement or a
    # tag merely because it has higher priority.
    return data.sort_values(["end", "filed", "tag_priority", "days"]).drop_duplicates("end", keep="first")


def duration_values(facts):
    """Derive quarters only from compatible intervals; retain all input availability dates."""
    columns = ['end', 'quarter', 'quarter_available_at', 'ttm', 'ttm_available_at', 'source_tag', 'status']
    data = _period_facts(facts)
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data.sort_values('end')
    raw = []
    quarters = []
    for row in data.itertuples(index=False):
        start, end, filed = (pd.Timestamp(row.start), pd.Timestamp(row.end), pd.Timestamp(row.filed))
        value = float(row.val)
        days = (end - start).days + 1
        q = np.nan
        available = filed
        qstart = start
        status = 'unsupported_duration'
        if 70 <= days <= 125:
            q = value
            status = 'discrete'
        elif 126 <= days <= 400:
            prefix = [r for r in raw if r['start'] == start and 70 <= (end - r['end']).days <= 125]
            if prefix:
                prior = max(prefix, key=lambda r: r['end'])
                q = value - prior['value']
                qstart = prior['end'] + pd.Timedelta(days=1)
                available = max(filed, prior['filed'])
                status = 'derived_q4_from_annual' if days >= 300 else 'derived_from_ytd'
            else:
                preceding = [r for r in quarters if r['start'] >= start and r['end'] < end and pd.notna(r['quarter'])]
                contiguous = bool(preceding) and preceding[0]['start'] == start
                contiguous &= all((b['start'] == a['end'] + pd.Timedelta(days=1) for a, b in zip(preceding, preceding[1:])))
                if contiguous and 70 <= (end - preceding[-1]['end']).days <= 125:
                    q = value - sum((r['quarter'] for r in preceding))
                    qstart = preceding[-1]['end'] + pd.Timedelta(days=1)
                    available = max([filed] + [r['quarter_available_at'] for r in preceding])
                    status = 'derived_q4_from_annual' if days >= 300 else 'derived_from_ytd'
                else:
                    status = 'missing_compatible_prior_periods'
        quarters.append({'start': qstart, 'end': end, 'quarter': q, 'quarter_available_at': available, 'source_tag': row.source_tag, 'status': status})
        raw.append({'start': start, 'end': end, 'filed': filed, 'value': value})
    output = pd.DataFrame(quarters).sort_values('end').reset_index(drop=True)
    output['ttm'] = np.nan
    output['ttm_available_at'] = pd.NaT
    for i in range(3, len(output)):
        w = output.iloc[i - 3:i + 1]
        contiguous = all((b.start == a.end + pd.Timedelta(days=1) for a, b in zip(w.itertuples(index=False), list(w.itertuples(index=False))[1:])))
        if w.quarter.notna().all() and contiguous:
            output.loc[i, 'ttm'] = w.quarter.sum()
            output.loc[i, 'ttm_available_at'] = w.quarter_available_at.max()
    return output[columns]


def instant_values(facts: pd.DataFrame, name: str) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame(columns=["end", name, f"{name}_available_at", f"{name}_source"])
    data = facts.sort_values(["end", "filed", "tag_priority"]).drop_duplicates("end", keep="first")
    return data[["end", "val", "filed", "source_tag"]].rename(columns={"val": name, "filed": f"{name}_available_at", "source_tag": f"{name}_source"})


def _instant_tag(facts: dict[str, Any], tag: str, name: str) -> pd.DataFrame:
    """Read one instantaneous debt concept without mixing overlapping tags."""
    return instant_values(fact_rows(facts, [tag], duration=False), name)


def debt_values(facts):
    """Select nonoverlapping debt components and their own publication dates together."""
    specs = {'total': TAGS['debt_total'], 'noncurrent': ['LongTermDebtNoncurrent'], 'ambiguous': ['LongTermDebt'], 'current': ['LongTermDebtCurrent', 'CurrentPortionOfLongTermDebt'], 'borrow': ['ShortTermBorrowings'], 'short_total': ['ShortTermDebt']}
    tables = {k: instant_values(fact_rows(facts, tags, duration=False), k).set_index('end') for k, tags in specs.items()}
    ends = sorted(set().union(*(set(f.index) for f in tables.values())))
    records = []

    def get(k, end):
        if end not in tables[k].index:
            return None
        r = tables[k].loc[end]
        return (float(r[k]), pd.Timestamp(r[k + '_available_at']), str(r[k + '_source']))
    for end in ends:
        v = {k: get(k, end) for k in specs}
        used = []
        if v['total']:
            used = [v['total']]
        elif v['noncurrent']:
            if v['short_total']:
                used = [v['noncurrent'], v['short_total']]
            elif v['borrow'] or v['current']:
                used = [v['noncurrent']] + [v[k] for k in ['borrow', 'current'] if v[k]]
        if not used and v['ambiguous']:
            used = [v['ambiguous']]
        records.append({'end': end, 'debt': sum((x[0] for x in used)) if used else np.nan, 'debt_available_at': max((x[1] for x in used)) if used else pd.NaT, 'debt_source': '+'.join((x[2] for x in used))})
    return pd.DataFrame(records, columns=['end', 'debt', 'debt_available_at', 'debt_source'])


def financial_quarters(facts: dict[str, Any]) -> pd.DataFrame:
    duration_names = ("revenue", "gross_profit", "cost_of_revenue", "operating_income", "operating_cash_flow")
    duration = {name: duration_values(fact_rows(facts, TAGS[name], duration=True)) for name in duration_names}
    quarters = duration["revenue"].rename(columns=lambda column: "revenue_" + column if column != "end" else column)
    for name in duration_names[1:]:
        quarters = quarters.merge(duration[name].rename(columns=lambda column: name + "_" + column if column != "end" else column), on="end", how="outer")
    quarters = quarters.merge(instant_values(fact_rows(facts, TAGS["assets"], duration=False), "assets"), on="end", how="outer")
    quarters = quarters.merge(debt_values(facts), on="end", how="outer").sort_values("end").reset_index(drop=True)
    average_assets = (quarters["assets"] + quarters["assets"].shift(4)) / 2
    gross_profit = quarters["gross_profit_ttm"].combine_first(quarters["revenue_ttm"] - quarters["cost_of_revenue_ttm"])
    gross_margin = gross_profit / quarters["revenue_ttm"]
    names = [name for name, _ in FUNDAMENTAL_DEFINITIONS]
    prior_revenue = quarters["revenue_quarter"].shift(4)
    prior_assets = quarters["assets"].shift(4)
    quarters["revenue_growth_yoy"] = (quarters["revenue_quarter"] / prior_revenue - 1).where(
        quarters["revenue_quarter"].gt(0) & prior_revenue.gt(0)
    )
    quarters["gross_margin_change_yoy"] = (gross_margin - gross_margin.shift(4)).where(
        quarters["revenue_ttm"].gt(0) & quarters["revenue_ttm"].shift(4).gt(0)
    )
    quarters["operating_profitability"] = (quarters["operating_income_ttm"] / average_assets).where(average_assets.gt(0))
    quarters["cashflow_to_assets"] = (quarters["operating_cash_flow_ttm"] / average_assets).where(average_assets.gt(0))
    quarters["debt_to_assets"] = (quarters["debt"] / quarters["assets"]).where(quarters["debt"].ge(0) & quarters["assets"].gt(0))
    quarters["asset_growth_yoy"] = (quarters["assets"] / prior_assets - 1).where(quarters["assets"].gt(0) & prior_assets.gt(0))
    quarters[names] = quarters[names].replace([np.inf, -np.inf], np.nan)
    # Each factor carries the latest public date of every one of its inputs.
    # This includes the earlier observations used in YoY, TTM and differences.
    def latest(*items: pd.Series) -> pd.Series:
        return pd.concat(items, axis=1).apply(pd.to_datetime, errors="coerce").max(axis=1)

    quarters["available_at_revenue_growth_yoy"] = latest(
        quarters["revenue_quarter_available_at"], quarters["revenue_quarter_available_at"].shift(4)
    )
    quarters["available_at_gross_margin_change_yoy"] = latest(
        quarters["revenue_ttm_available_at"], quarters["gross_profit_ttm_available_at"],
        quarters["cost_of_revenue_ttm_available_at"], quarters["revenue_ttm_available_at"].shift(4),
        quarters["gross_profit_ttm_available_at"].shift(4), quarters["cost_of_revenue_ttm_available_at"].shift(4),
    )
    quarters["available_at_operating_profitability"] = latest(
        quarters["operating_income_ttm_available_at"], quarters["assets_available_at"], quarters["assets_available_at"].shift(4)
    )
    quarters["available_at_cashflow_to_assets"] = latest(
        quarters["operating_cash_flow_ttm_available_at"], quarters["assets_available_at"], quarters["assets_available_at"].shift(4)
    )
    quarters["available_at_debt_to_assets"] = latest(quarters["debt_available_at"], quarters["assets_available_at"])
    quarters["available_at_asset_growth_yoy"] = latest(quarters["assets_available_at"], quarters["assets_available_at"].shift(4))
    return quarters[["end", *names, *[f"available_at_{name}" for name in names]]]


def next_market_day(dates: pd.Series, market_dates: pd.Series | None = None) -> pd.Series:
    """Map a filing date to the next official NYSE session.
    """
    values = pd.to_datetime(dates, errors="coerce").dt.normalize()
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = values.notna()
    if not valid.any():
        return result
    calendar = xcals.get_calendar("XNYS")
    first, last = calendar.first_session.tz_localize(None), calendar.last_session.tz_localize(None)
    supported = valid & values.between(first, last)
    if not supported.any():
        return result
    start = values.loc[supported].min()
    end = min(values.loc[supported].max() + pd.Timedelta(days=14), last)
    sessions = calendar.sessions_in_range(start, end).tz_localize(None).normalize()
    positions = sessions.searchsorted(pd.DatetimeIndex(values.loc[supported]), side="right")
    matched = [sessions[position] if position < len(sessions) else pd.NaT for position in positions]
    result.loc[supported] = matched
    return result


def collect_financial_factors(base: pd.DataFrame, market_dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect financial factors and return a terminal-only fetch summary."""
    names = [name for name, _ in FUNDAMENTAL_DEFINITIONS]
    mapping, logs, events = cik_by_ticker(), [], []
    for ticker in sorted(base["ticker"].unique()):
        cik = mapping.get(ticker) or mapping.get(ticker.replace("-", "."))
        if cik is None:
            logs.append({"ticker": ticker, "status": "not_in_sec_mapping", "message": "No SEC ticker/CIK mapping"})
            continue
        try:
            raw = fetch_json(FACTS_URL.format(cik=cik))
            taxonomies = raw.get("facts", {})
            if not taxonomies.get("us-gaap"):
                status = "ifrs_or_non_us_gaap_not_supported" if taxonomies.get("ifrs-full") else "no_us_gaap_facts"
                logs.append({"ticker": ticker, "cik": cik, "status": status,
                             "message": "Only standard US-GAAP 10-Q/10-K concepts are mapped"})
                continue
            quarters = financial_quarters(raw)
            if quarters.empty:
                raise ValueError("No usable US-GAAP 10-Q/10-K facts")
            quarters["ticker"], quarters["cik"] = ticker, cik
            for name in names:
                availability = f"available_at_{name}"
                quarters[availability] = next_market_day(quarters[availability])
            events.append(quarters)
            logs.append({"ticker": ticker, "cik": cik, "status": "success", "message": ""})
        except Exception as error:
            logs.append({"ticker": ticker, "cik": cik, "status": "failed", "message": str(error)})
    log = pd.DataFrame(logs)
    result = base[["date", "ticker"]].copy()
    for name in names:
        result[name] = np.nan
    if not events:
        return result, log
    all_events = pd.concat(events, ignore_index=True)
    frames = []
    for ticker, keys in base[["date", "ticker"]].groupby("ticker", sort=False):
        joined = keys.sort_values("date").copy()
        for name in names:
            availability = f"available_at_{name}"
            source = all_events.loc[
                (all_events["ticker"] == ticker) & all_events[name].notna() & all_events[availability].notna(),
                [availability, "end", name],
            ].sort_values([availability, "end"]).drop_duplicates(availability, keep="last")
            if source.empty:
                joined[name] = np.nan
            else:
                joined[name] = pd.merge_asof(
                    joined[["date"]], source[[availability, name]], left_on="date", right_on=availability, direction="backward"
                )[name].to_numpy()
        frames.append(joined)
    return pd.concat(frames, ignore_index=True), log
