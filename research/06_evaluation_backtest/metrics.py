"""Prediction and portfolio metrics.

Formulas follow the report-chapter verification code
(``reference_code/verify_latest.py`` and ``verify_prediction_metrics.py``):
IC/RankIC are weekly cross-sectional correlations; ICIR is mean/std (ddof=1),
not annualised; annualised return is NAV ** (52 / weeks) - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corr(a, b) -> float:
    """Pearson correlation, NaN when either side has no cross-sectional spread."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def prediction_metrics(q: pd.DataFrame) -> dict:
    """IC, RankIC, their IRs, positive RankIC share and regression errors."""
    ic, ric = [], []
    for _, g in q.groupby('date', sort=True):
        ic.append(corr(g.prediction, g.target))
        ric.append(corr(g.prediction.rank(method='average'), g.target.rank(method='average')))
    ic, ric = np.asarray(ic), np.asarray(ric)
    good = ric[np.isfinite(ric)]
    error = q.prediction - q.target
    return dict(rows=len(q), weeks=len(ic),
                ic=float(np.nanmean(ic)), rank_ic=float(np.nanmean(ric)),
                icir=float(np.nanmean(ic) / np.nanstd(ic, ddof=1)),
                rank_icir=float(np.nanmean(ric) / np.nanstd(ric, ddof=1)),
                positive_rank_ic=float(np.mean(good > 0)),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
                zero_prediction_rmse=float(np.sqrt(np.mean(q.target ** 2))))


def performance(g: pd.DataFrame) -> dict:
    """Summary of one strategy path; ``g`` holds weekly net_return in date order."""
    g = g.sort_values('date')
    r = g.net_return.to_numpy(float)
    nav = np.r_[1., np.cumprod(1 + r)]
    vol = float(np.std(r, ddof=1) * np.sqrt(52)) if len(r) > 1 else np.nan
    return dict(weeks=len(r), first_signal=g.date.min(), last_signal=g.date.max(),
                total_return=float(nav[-1] - 1),
                annualized_return=float(nav[-1] ** (52 / len(r)) - 1),
                max_drawdown=float((nav / np.maximum.accumulate(nav) - 1).min()),
                volatility=vol,
                sharpe=float(np.mean(r) * 52 / vol) if vol and vol > 0 else np.nan,
                win_rate=float(np.mean(r > 0)),
                mean_traded_weight=float(g.traded_weight.mean()),
                mean_cash_weight=float(g.cash_weight.mean()),
                total_cost_fraction=float(g.transaction_cost_fraction.sum()))
