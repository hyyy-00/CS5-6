from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd

from . import ROOT
from .contracts import load_module, require



def implementation():
    return load_module('repro_integer_engine', ROOT / 'backtesting/external40/integer_backtest.py')




def independent_case(summary, weekly, orders, holdings, prices, predictions, *, quote_lookup=None, rank_lookup=None):
    """Rebuild an account using order cashflows; do not invoke execute/performance."""
    initial = float(summary.capital)
    cost = float(summary.cost_bps) / 10000
    cash, prior, peak = initial, initial, initial
    held = {}
    returns, fees, navs = [], [], []
    maximum_error = 0.
    checks = 0
    def near(actual, expected, label):
        nonlocal maximum_error, checks
        error = abs(float(actual) - float(expected))
        maximum_error = max(maximum_error, error)
        require(error <= 1e-7 + 1e-10 * abs(float(expected)), f'{label}: actual={actual}, expected={expected}')
        checks += 1
    quote = quote_lookup if quote_lookup is not None else {
        (r.date, r.ticker): (float(r.buy_price), float(r.sell_price)) for r in prices.itertuples(index=False)}
    ranks = rank_lookup if rank_lookup is not None else {
        d: list(g.sort_values(['prediction', 'ticker'], ascending=[False, True], kind='stable').ticker)
        for d, g in predictions.groupby('date')}
    lag13 = None
    if summary.strategy == 'Top5_NoLoss13w':
        closes = prices.pivot(index='date', columns='ticker', values='sell_price').sort_index()
        observed = closes.shift(1)
        trailing = observed / observed.shift(13) - 1
        lag13 = {d: row.dropna().to_dict() for d, row in trailing.iterrows()}
    order_groups = {d: g for d, g in orders.groupby('date', sort=False)}
    holding_groups = {d: g for d, g in holdings.groupby('date', sort=False)}
    require(set(orders.date) <= set(weekly.date), 'Orders outside weekly account')
    require(set(holdings.date) <= set(weekly.date), 'Holdings outside weekly account')
    for row in weekly.sort_values('date').itertuples(index=False):
        date = row.date
        near(row.opening_cash, cash, 'opening cash')
        near(row.start_capital, prior, 'prior NAV')
        equity = cash + math.fsum(q * quote[(date, ticker)][0] for ticker, q in held.items())
        near(row.opening_capital, equity, 'opening equity')
        day = order_groups.get(date, orders.iloc[:0])
        day_fees = 0.
        def apply(order):
            nonlocal cash, day_fees
            qty = int(order.shares)
            require(qty == order.shares and qty > 0, 'Noninteger or nonpositive order')
            expected_price = quote[(date, order.ticker)][0 if order.timing == 'OPEN' else 1]
            near(order.price, expected_price, 'execution quote')
            amount = qty * expected_price
            fee = amount * cost
            near(order.amount, amount, 'trade amount')
            near(order.fee, fee, 'trade fee')
            if order.side == 'BUY':
                cash -= amount + fee
                held[order.ticker] = held.get(order.ticker, 0) + qty
            else:
                require(order.side == 'SELL' and held.get(order.ticker, 0) >= qty, 'Invalid sell/short position')
                cash += amount - fee
                held[order.ticker] -= qty
                if not held[order.ticker]:
                    del held[order.ticker]
            require(cash >= -1e-7, 'Negative cash')
            near(order.cash_after, cash, 'order cash balance')
            day_fees += fee
        require(day.timing.isin(['OPEN', 'CLOSE']).all(), 'Unknown execution timing')
        for order in day[day.timing.eq('OPEN')].itertuples(index=False):
            apply(order)
        position = holding_groups.get(date, holdings.iloc[:0])
        require(not position.ticker.duplicated().any(), 'Duplicate holding')
        observed = dict(zip(position.ticker, position.shares))
        require(observed == held, f'Holdings disagree with reconstructed shares on {date}')
        for item in position.itertuples(index=False):
            near(item.market_value, held[item.ticker] * quote[(date, item.ticker)][1], 'position mark')
        if summary.strategy in ['Top5', 'Top10', 'Top5_RankWeighted', 'Top5_NoLoss13w']:
            count = 10 if summary.strategy == 'Top10' else 5
            expected_names = ranks[date]
            if summary.strategy == 'Top5_NoLoss13w':
                expected_names = [
                    ticker for ticker in expected_names
                    if lag13[date].get(ticker, -math.inf) >= 0
                ]
            expected_names = expected_names[:count]
            require(set(held) <= set(expected_names), 'Selection includes a stock outside the predicted ranks or trailing 13-week loss filter')
            positive_targets = set()
            for ticker in expected_names:
                weight = (5 - expected_names.index(ticker)) / 15 if summary.strategy == 'Top5_RankWeighted' else 1 / count
                expected_request = math.floor(weight * equity / (quote[(date, ticker)][0] * (1 + cost)))
                if expected_request > 0:
                    positive_targets.add(ticker)
            if summary.strategy == 'Top5_NoLoss13w':
                require(set(held) == positive_targets, 'Filtered Top5 holdings do not match independently selected names')
            for item in position.itertuples(index=False):
                weight = (5 - expected_names.index(item.ticker)) / 15 if summary.strategy == 'Top5_RankWeighted' else 1 / count
                expected_request = math.floor(weight * equity / (quote[(date, item.ticker)][0] * (1 + cost)))
                near(item.requested_shares, expected_request, 'independent target share sizing')
                require(item.shares <= expected_request, 'Held shares exceed target')
        marked_nav = cash + math.fsum(q * quote[(date, ticker)][1] for ticker, q in held.items())
        near(row.cash, cash, 'post-open cash')
        near(row.cash_ratio, cash / marked_nav if marked_nav else 1., 'cash ratio')
        near(row.shares, sum(held.values()), 'marked share total')
        near(row.holdings, len(held), 'holding count')
        for order in day[day.timing.eq('CLOSE')].itertuples(index=False):
            apply(order)
        final_nav = cash + math.fsum(q * quote[(date, ticker)][1] for ticker, q in held.items())
        near(row.end_capital, final_nav, 'end NAV')
        near(row.post_execution_cash, cash, 'post-close cash')
        near(row.net_return, final_nav / prior - 1, 'weekly return')
        near(row.fees, day_fees, 'weekly fees')
        returns.append(final_nav / prior - 1)
        fees.append(day_fees)
        navs.append(final_nav)
        prior = final_nav
    require(not held, 'Final holdings were not liquidated')
    require(len(weekly) == summary.weeks, 'Summary week count')
    near(summary.final_capital, prior, 'summary final capital')
    near(summary.total_return, prior / initial - 1, 'summary return')
    near(summary.annualized_return, (prior / initial) ** (52 / len(returns)) - 1, 'annualization')
    near(summary.total_fees, math.fsum(fees), 'summary fees')
    drawdown = 0.
    for nav in navs:
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1)
    near(summary.max_drawdown, drawdown, 'drawdown including initial capital')
    near(summary.win_rate, sum(r > 0 for r in returns) / len(returns), 'win rate')
    if len(returns) > 1:
        vol = statistics.stdev(returns) * math.sqrt(52)
        near(summary.volatility, vol, 'volatility')
        if vol > 0:
            near(summary.sharpe, statistics.mean(returns) * 52 / vol, 'Sharpe')
    return {'checks': checks, 'maximum_absolute_error': maximum_error, 'orders': len(orders), 'holdings': len(holdings)}




