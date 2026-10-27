"""Independent arithmetic and holdings-path checks using actual prices."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
checks=0;max_error=0.

def eq(a,b,label):
    global checks,max_error
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    assert x.shape==y.shape,(label,x.shape,y.shape)
    assert np.array_equal(np.isnan(x),np.isnan(y)),label
    mask=np.isfinite(x)&np.isfinite(y)
    e=float(np.max(np.abs(x[mask]-y[mask]))) if mask.any() else 0.
    assert e<1e-9,(label,e)
    max_error=max(max_error,e);checks+=x.size

def cap(scores,budget):
    # Solve capped proportional weights by bisection, independently of waterfill.
    s=np.asarray(scores,float);budget=min(budget,.35*len(s))
    if not len(s):return s
    lo=0.;hi=budget/max(s.min(),1e-30)+1
    for _ in range(100):
        mid=(lo+hi)/2
        if np.minimum(.35,mid*s).sum()<budget:lo=mid
        else:hi=mid
    return np.minimum(.35,hi*s)

def main():
    global checks
    raw=pd.read_csv(next((HERE/'source/basic data').rglob('test_45_factors_weekly.csv')),parse_dates=['date'])
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    fwd=prices.shift(-1)/prices-1;past=prices/prices.shift(1)-1
    vol=past.rolling(26,min_periods=26).std(ddof=1)
    momentums={n:prices/prices.shift(n)-1 for n in [4,13]}
    market_index=(1+past.mean(axis=1).fillna(0)).cumprod()
    trend=(market_index>market_index.rolling(26,min_periods=26).mean()).fillna(False)
    preds={p.stem:pd.read_csv(p,parse_dates=['date']) for p in (OUT/'predictions').glob('*.csv')}
    for name,p in preds.items():
        eq(np.expm1(p.target),[fwd.loc[d,t] for d,t in zip(p.date,p.ticker)],name+' direct price returns')
    weekly=pd.read_csv(OUT/'weekly_backtest.csv',parse_dates=['date','label_end_date'])
    fullpos=pd.read_csv(OUT/'positions.csv',parse_dates=['date'])
    winpos=pd.read_csv(OUT/'stateful_window_positions.csv',parse_dates=['date'])
    fullgroups={key:g for key,g in fullpos.groupby(['experiment','strategy'])}
    wingroups={key:g for key,g in winpos.groupby(['experiment','window','strategy'])}
    predgroups={name:{d:g.sort_values(['prediction','ticker'],ascending=[False,True],kind='stable') for d,g in p.groupby('date')} for name,p in preds.items()}
    for (name,window,strategy),g in weekly.groupby(['experiment','window','strategy']):
        g=g.sort_values('date');pg=predgroups[name]
        if window!='FullHistory' and strategy in ['Top5_Buffer10','Top5_Rebalance2w','Top5_Rebalance4w']:
            positions=wingroups.get((name,window,strategy),winpos.iloc[:0])
        else:positions=fullgroups.get((name,strategy),fullpos.iloc[:0])
        holdings={d:dict(zip(z.ticker,z.weight)) for d,z in positions.groupby('date')}
        previous={};cash=1.;prev_r={};last=None
        for i,row in enumerate(g.itertuples()):
            d=row.date;weights=holdings.get(d,{})
            growth=cash+sum(w*(1+prev_r[t]) for t,w in previous.items())
            drift={t:w*(1+prev_r[t])/growth for t,w in previous.items()}
            pool=pg.get(d,preds[name].iloc[:0]);selected=pool
            k=5 if strategy.startswith('Top5') else 10
            if 'PositiveScore' in strategy:selected=selected[selected.prediction>0]
            if 'NoLoss' in strategy:
                n=4 if '4w' in strategy else 13
                momentum=momentums[n].loc[d]
                selected=selected[selected.ticker.map(momentum)>=0]
            if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                selected=selected[selected.ticker.map(vol.loc[d]).notna()]
            if strategy=='Top5_Buffer10':
                kept=[t for t in pool.head(10).ticker if t in previous]
                chosen=kept+[t for t in pool.ticker if t not in kept][:5-len(kept)]
                target=dict.fromkeys(chosen,.2)
            elif strategy in ['Top5_Rebalance2w','Top5_Rebalance4w']:
                period=2 if '2w' in strategy else 4
                if not len(pool):target={};last=None
                elif not previous or last is None or i-last>=period:
                    target=dict.fromkeys(pool.head(5).ticker,.2);last=i
                else:target={t:w for t,w in drift.items() if pd.notna(fwd.loc[d].get(t,np.nan))}
            elif strategy=='Top5_Trend26w' and not trend.loc[d]:target={}
            else:
                selected=selected.head(k);names=list(selected.ticker)
                if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                    scores=[(5-j if strategy=='Top5_RankRisk26w' else 1)/max(vol.loc[d,t],1e-8) for j,t in enumerate(names)]
                    target=dict(zip(names,cap(scores,len(names)/5)))
                elif strategy=='Top5_RankWeighted':target={t:(5-j)/15 for j,t in enumerate(names)}
                else:target=dict.fromkeys(names,1/k)
            assert set(weights)==set(target),(name,window,strategy,d,weights,target)
            checks+=1
            eq([weights[t] for t in sorted(weights)],[target[t] for t in sorted(target)],'weights')
            realized={t:float(fwd.loc[d,t]) for t in weights}
            gross=sum(weights[t]*realized[t] for t in weights)
            trade=sum(abs(weights.get(t,0)-drift.get(t,0)) for t in set(weights)|set(drift))
            eq([row.gross_return,row.traded_weight,row.cash_weight,row.transaction_cost_fraction,row.net_return],
               [gross,trade,1-sum(weights.values()),trade*.001,(1-trade*.001)*(1+gross)-1],'weekly arithmetic')
            previous=weights;prev_r=realized;cash=1-sum(weights.values())
    stats=pd.read_csv(OUT/'strategy_comparison.csv')
    groups={key:g.sort_values('date') for key,g in weekly.groupby(['experiment','window','strategy'])}
    for row in stats.itertuples():
        g=groups[row.experiment,row.window,row.strategy]
        r=(1-g.traded_weight.to_numpy()*row.cost_bps/10000)*(1+g.gross_return.to_numpy())-1
        nav=np.r_[1.,np.cumprod(1+r)];years=len(r)/52
        cagr=nav[-1]**(1/years)-1
        dd=(nav/np.maximum.accumulate(nav)-1).min();v=np.std(r,ddof=1)*np.sqrt(52)
        eq([row.total_return,row.annualized_return,row.max_drawdown,row.volatility,row.sharpe,row.win_rate],
           [nav[-1]-1,cagr,dd,v,np.mean(r)*52/v if v>0 else np.nan,np.mean(r>0)],'summary metrics')
    phases=pd.read_csv(OUT/'phase_sensitivity.csv')
    zero=phases[phases.initial_delay_weeks.eq(0)].merge(stats,on=['experiment','window','strategy','cost_bps'],suffixes=('_phase','_main'),validate='one_to_one')
    for col in ['total_return','annualized_return','max_drawdown','sharpe']:
        eq(zero[col+'_phase'],zero[col+'_main'],'phase zero '+col)
    for n in [5,7,11]:
        name=f'CONSENSUS-R6-F{n}'
        p=preds[name];joint=p[['date','ticker','prediction']]
        cols=[]
        for f in ['RF','RIDGE','XGB','RANKER']:
            x=preds[f'{f}-R6-F{n}'].copy();x[f]=x.groupby('date').prediction.rank(pct=True,method='average')
            joint=joint.merge(x[['date','ticker',f]],on=['date','ticker'],validate='one_to_one');cols.append(f)
        eq(joint.prediction,joint[cols].mean(axis=1).round(12),'consensus scores')
    result=dict(status='passed',checks=checks,max_abs_difference=max_error,scope='direct price returns; independently reconstructed every model/window/strategy position and drift/cost; all summary cases at all fees; phase-zero equivalence; consensus scores',summary_cases=len(stats))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    manifest=json.loads((OUT/'run_manifest.json').read_text());manifest['status']='completed_and_verified';manifest['independent_checks']=checks
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(result)

if __name__=='__main__':main()
