"""Backward-looking portfolio inputs and fixed, auditable strategy rules."""
import numpy as np
import pandas as pd

NEW_STRATEGIES=['Top5_LowVol26w','Top5_RankRisk26w','Top5_Buffer10','Top5_Rebalance2w','Top5_Rebalance4w','Top5_Trend26w']
STATEFUL=['Top5_Buffer10','Top5_Rebalance2w','Top5_Rebalance4w']
CONSENSUS='CONSENSUS-R3-F11-4MODELS'
COMPONENTS=['RF-R3-F11','RIDGE-R3-F11','XGB-R3-F11','RANKER-V2-ENSEMBLE']
LABELS={'Top5':'Equal-weight Top5','Top10':'Equal-weight Top10','Top5_RankWeighted':'Rank-weighted Top5',
        'Top5_PositiveScore':'Top5 positive score','Top10_PositiveScore':'Top10 positive score',
        'Top5_NoLoss4w':'Top5 no loss over past 4 weeks','Top10_NoLoss4w':'Top10 no loss over past 4 weeks',
        'Top5_NoLoss13w':'Top5 no loss over past 13 weeks','Top10_NoLoss13w':'Top10 no loss over past 13 weeks',
        'Top5_LowVol26w':'Low-volatility weighted','Top5_RankRisk26w':'Rank-risk weighted','Top5_Buffer10':'Top-10 retention buffer',
        'Top5_Rebalance2w':'Rebalance every 2 weeks','Top5_Rebalance4w':'Rebalance every 4 weeks','Top5_Trend26w':'Market trend filter'}

def inputs(prices):
    past=prices/prices.shift(1)-1
    vol=past.rolling(26,min_periods=26).std(ddof=1)
    market_index=(1+past.mean(axis=1).fillna(0)).cumprod()
    trend=(market_index>market_index.rolling(26,min_periods=26).mean()).fillna(False)
    return dict(vol=vol,trend=trend,stock_returns=prices.shift(-1)/prices-1,
                market_index=market_index,prices=prices)

def capped_weights(scores,budget=1.,cap=.35):
    """Redistribute weight exceeding the cap; any infeasible budget stays cash."""
    scores={t:float(v) for t,v in scores.items() if np.isfinite(v) and v>0}
    allocation={t:0. for t in scores};remaining=min(float(budget),len(scores)*cap)
    active=dict(scores)
    while active and remaining>1e-14:
        total=sum(active.values())
        proposed={t:remaining*s/total for t,s in active.items()}
        clipped=[t for t,w in proposed.items() if w>cap+1e-14]
        if not clipped:
            allocation.update(proposed);break
        for t in clipped:allocation[t]=cap;remaining-=cap;active.pop(t)
    return allocation

def consensus_predictions(ready):
    joint=None
    for component in COMPONENTS:
        p=ready[component][['date','ticker','target','prediction']].copy()
        p['percentile']=p.groupby('date').prediction.rank(method='average',pct=True)
        p=p[['date','ticker','target','percentile']].rename(columns={'percentile':component})
        if joint is None:joint=p
        else:
            joint=joint.merge(p,on=['date','ticker'],how='inner',suffixes=('','_check'),validate='one_to_one')
            assert np.allclose(joint.target,joint.target_check,atol=1e-12,rtol=0)
            joint=joint.drop(columns='target_check')
    assert len(joint)>0
    # Arithmetic means of rational ranks can differ at ~1e-16 for true ties.
    # Freeze precision before ranking or CSV serialization, then break ties by ticker.
    joint['prediction']=joint[COMPONENTS].mean(axis=1).round(12)
    return joint[['date','ticker','target','prediction']],joint

RULES={
 'Top5_LowVol26w':'Take the top 5 predicted stocks that have a complete 26-week volatility history and weight them by inverse volatility, capped at 35% per stock; with fewer than 5 stocks the equity budget is count/5 and the rest stays in cash.',
 'Top5_RankRisk26w':'From the same eligible pool take the top 5 and weight by (6 - rank) / 26-week standard deviation of weekly returns, capped at 35% per stock; with fewer than 5 stocks the equity budget is count/5.',
 'Top5_Buffer10':'Start with the top 5; each week keep holdings that are still in the current top 10, fill empty slots by current rank, then rebalance to equal weights of 20% each.',
 'Top5_Rebalance2w':'Buy the top 5 from cash on the first available signal and re-select equal weights every 2 calendar weeks; in between, holdings and cash drift naturally and no trades follow new ranks. A missing signal liquidates; a returning signal buys immediately and resets the rebalance clock.',
 'Top5_Rebalance4w':'Same as above, rebalancing every 4 calendar weeks.',
 'Top5_Trend26w':'Build an equal-weight return index from verifiable adjusted prices of the original universe up to the current week; buy the equal-weight Top5 when the index is above its 26-week moving average (including the current week), otherwise hold cash.',
 'Consensus':'Equal-weight average of the weekly within-universe rank percentiles of four fixed configurations; scores are rounded to 12 decimals to fix numerical ties, then ties break by ticker; only stocks and dates predicted by all four are used, and components were not chosen by these results. The derived series is also evaluated on all 15 strategies.',
}
