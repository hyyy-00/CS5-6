"""Training preparation from supplied processed weekly data only."""
from __future__ import annotations
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from . import ROOT
from .contracts import require, sha, write_frame, write_json

FEATURES = ['revenue_growth_yoy', 'operating_profitability', 'cashflow_to_assets',
            'debt_to_assets', 'asset_growth_yoy']

def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip', parse_dates=['date'])


def nyse_calendar(start, end):
    return xcals.get_calendar('XNYS', start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()))


def execution_dates(signal_dates):
    dates = pd.Series(pd.to_datetime(signal_dates)).reset_index(drop=True)
    cal = nyse_calendar(dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=21))
    sessions = pd.Series(cal.sessions.tz_localize(None))
    ends = sessions.groupby(sessions.dt.to_period('W-FRI')).max()
    starts = sessions.groupby(sessions.dt.to_period('W-FRI')).min()
    periods = dates.dt.to_period('W-FRI')
    require(all(dates.eq(periods.map(ends))), 'Signal dates must be final official NYSE sessions of complete weeks')
    return pd.DataFrame({'date': dates, 'expected_buy_date': (periods + 1).map(starts),
                         'expected_sell_date': (periods + 1).map(ends)})


def validate_panel(frame, name, expected_tickers, numeric):
    require({'date', 'ticker', *numeric} <= set(frame), f'{name}: missing columns')
    require(not frame.empty and not frame.duplicated(['date', 'ticker']).any(), f'{name}: empty or duplicate keys')
    require(frame.date.notna().all() and frame.ticker.notna().all(), f'{name}: missing keys')
    require(set(frame.ticker) == expected_tickers, f'{name}: unexpected/missing ticker universe')
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), f'{name}: NaN or infinite numeric inputs')
    execution_dates(frame.date.drop_duplicates())



def preflight():
    files = [ROOT / 'data/reproducible/train66_corrected_complete.csv',
             ROOT / 'data/reproducible/train_universe.csv']
    return files, [str(p) for p in files if not p.is_file()]


def prepare_training(train):
    universe = pd.read_csv(ROOT / 'data/reproducible/train_universe.csv')
    require(len(universe) == 66 and universe.ticker.notna().all() and not universe.ticker.duplicated().any(),
            'Training universe must contain 66 distinct tickers')
    train = train.sort_values(['date', 'ticker'], kind='stable').reset_index(drop=True).copy()
    validate_panel(train, 'train', set(universe.ticker), ['adj_close', 'target', *FEATURES])
    require(train.adj_close.gt(0).all(), 'Nonpositive training price')
    calendar = pd.DatetimeIndex(sorted(train.date.unique()))
    periods = calendar.to_period('W-FRI')
    require(all(periods[1:] == periods[:-1] + 1), 'Training calendar skips an entire trading week')
    require(len(calendar) == 521 and calendar[0] == pd.Timestamp('2016-09-02')
            and calendar[-1] == pd.Timestamp('2026-08-21'), 'Unexpected supplied weekly training coverage')
    train['week_index'] = train.date.map({d: i for i, d in enumerate(calendar)}).astype(int)
    train = train.drop(columns=['label_end_date'], errors='ignore').merge(
        execution_dates(calendar).rename(columns={'expected_sell_date': 'label_end_date'}),
        on='date', validate='many_to_one').drop(columns='expected_buy_date')
    ordered = train.sort_values(['ticker', 'date'])
    future_index = ordered.groupby('ticker').week_index.shift(-1)
    future_price = ordered.groupby('ticker').adj_close.shift(-1)
    check = future_index.eq(ordered.week_index + 1)
    require(np.allclose(ordered.loc[check, 'target'],
                        np.log(future_price[check] / ordered.loc[check, 'adj_close']),
                        rtol=0, atol=1e-8), 'Training target is not its documented next-week close return')
    return train


def prepare(run, options):
    destination = run / 'prepare'
    source = ROOT / 'data/reproducible/train66_corrected_complete.csv'
    train = prepare_training(read_csv(source))
    write_frame(train, destination / 'train.csv')
    calendar = train[['date', 'week_index', 'label_end_date']].drop_duplicates().sort_values('date')
    write_frame(calendar, destination / 'calendar.csv')
    info = dict(protocol='processed-weekly-ensemble-training-v1', weeks=len(calendar),
                training_rows=len(train), training_tickers=int(train.ticker.nunique()),
                first_signal=str(calendar.date.min().date()), last_signal=str(calendar.date.max().date()),
                source_sha256=sha(source), input_scope='supplied processed weekly data',
                target_check='adjacent supplied weekly adjusted closes',
                terminal_target='supplied processed label; raw source reconstruction outside delivery scope',
                external_data_used=False)
    write_json(destination / 'data_audit.json', info)
    return info
