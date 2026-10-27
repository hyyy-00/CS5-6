"""Latest delivered model histories: all strategies, independent windows and costs."""
from pathlib import Path
import sys,json,hashlib,html,platform
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import full_history_backtest as bt
import run as engine
from portfolio_strategies import inputs,STATEFUL,RULES,LABELS

OUT=HERE/'results';OUT.mkdir(exist_ok=True)
PRED=OUT/'predictions';PRED.mkdir(exist_ok=True)
CHART=OUT/'charts';CHART.mkdir(exist_ok=True)
SOURCE=HERE/'source';REPLAY=HERE/'replay'
DATA=next((SOURCE/'basic data').rglob('test_45_factors_weekly.csv'))
bt.OUT=OUT;bt.PREDS=PRED
engine.COLORS=['#2563eb','#f97316','#16a34a','#dc2626','#9333ea','#0891b2','#be185d','#a16207','#475569','#7c3aed','#0f766e','#c026d3','#0284c7','#be123c','#65a30d','#111827']

def csv(name,data):data.to_csv(OUT/name,index=False,float_format='%.17g')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def chart(name,title,series,ylabel='NAV (initial = 1)',baseline=1):
    engine.svg_chart(CHART/name,title,series,ylabel,baseline)
    # Extra right margin prevents long strategy names being clipped.
    f=CHART/name;s=f.read_text();s=s.replace('width="1120"','width="1450"').replace('viewBox="0 0 1120 620"','viewBox="0 0 1450 620"');f.write_text(s)

def windows(p,calendar):
    w=bt.windows(p,calendar)
    w.pop('Development3y',None)
    return w

def load_records():
    records=[]
    for path in sorted((REPLAY/'predictions').glob('*.csv')):
        records.append((path.stem,pd.read_csv(path,parse_dates=['date']),'latest_native_replay',str(path)))
    ridge=pd.read_csv(REPLAY/'ridge/predictions.csv',parse_dates=['date']).rename(columns={'actual':'target'})
    for key,p in ridge.groupby('feature_set'):
        n=11 if key=='7_macro' else int(key)
        records.append((f'RIDGE-R6-F{n}',p,'latest_native_replay','Ridge supplied script'))
    for path in sorted((REPLAY/'ranker').glob('selected_predictions_*.csv.gz')):
        key=path.name.removeprefix('selected_predictions_').removesuffix('.csv.gz')
        n=11 if key=='7_macro' else int(key)
        p=pd.read_csv(path,parse_dates=['date']).rename(columns={'y_true':'target','y_pred':'prediction'})
        records.append((f'RANKER-R6-F{n}',p,'latest_native_replay',str(path)))
    return records

def source_reconciliation():
    recovered=pd.read_csv(REPLAY/'recovered_source_metrics.csv')
    rd=pd.read_csv(REPLAY/'ridge_results.csv').assign(family='RIDGE')
    rk=pd.read_csv(REPLAY/'ranker/xgboostranker_optimized_results.csv').assign(family='RANKER')
    recovered=pd.concat([recovered,rd,rk],ignore_index=True)
    supplied=[]
    for family,pattern in [('RF','random_forest_5_7_11_45_6y26w_results.csv'),('XGB','xgboost_5_7_11_45_6y26w_results.csv'),('RIDGE','ridge_5_7_11_45_6y26w_results.csv'),('RANKER','xgboostranker_optimized_results.csv')]:
        frame=pd.read_csv(next(SOURCE.rglob(pattern))).rename(columns={'stage':'Stage'})
        supplied.append(frame.assign(family=family))
    original=pd.concat(supplied,ignore_index=True)
    joined=original.merge(recovered,on=['family','Dataset','Stage'],suffixes=('_supplied','_replayed'),validate='one_to_one')
    columns=['IC','ICIR','Rank IC','Rank ICIR','Annualized Return','Information Ratio','Top5 Excess','Positive Rank IC']
    for c in columns:joined[c+'_difference']=joined[c+'_replayed']-joined[c+'_supplied']
    csv('source_metric_reconciliation.csv',joined)
    return float(joined[[c+'_difference' for c in columns]].abs().max().max())

def main():
    raw=pd.read_csv(DATA,parse_dates=['date'])
    ready,coverage=bt.prepare(load_records(),raw)
    status=[]
    for family in ['RF','RIDGE','XGB','RANKER']:
        for n in [5,7,11,45]:
            name=f'{family}-R6-F{n}'
            status.append(dict(experiment=name,available=name in ready,status='replayed' if name in ready else 'no_candidate_passed'))
    csv('official_model_status.csv',pd.DataFrame(status))
    components=[]
    # Same-factor consensus is fixed in advance; never choose components by return.
    for n in [5,7,11,45]:
        names=[f'{f}-R6-F{n}' for f in ['RF','RIDGE','XGB','RANKER']]
        if not all(k in ready for k in names):continue
        joint=None
        for name in names:
            p=ready[name][['date','ticker','target','prediction']].copy()
            p[name]=p.groupby('date').prediction.rank(pct=True,method='average')
            p=p[['date','ticker','target',name]]
            if joint is None:joint=p
            else:
                joint=joint.merge(p,on=['date','ticker'],suffixes=('','_check'),validate='one_to_one')
                assert np.allclose(joint.target,joint.target_check,atol=1e-12,rtol=0)
                joint=joint.drop(columns='target_check')
        joint['prediction']=joint[names].mean(axis=1).round(12)
        name=f'CONSENSUS-R6-F{n}'
        p=joint[['date','ticker','target','prediction']]
        derived,cov=bt.prepare([(name,p,'derived_consensus','fixed equal-weight same-factor four models')],raw,write_audits=False)
        ready.update(derived);coverage=pd.concat([coverage,cov],ignore_index=True)
        components.append(joint.assign(experiment=name))
    csv('consensus_components.csv',pd.concat(components,ignore_index=True))
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    context=inputs(prices);dates_all=list(prices.index)
    next_dates=dict(zip(dates_all[:-1],dates_all[1:]))
    market=context['stock_returns'].mean(axis=1).to_dict()
    lag={n:prices/prices.shift(n)-1 for n in [4,13]}
    metrics=[];summaries=[];weeklies=[];positions=[];window_positions=[];phase_rows=[];chart_records=[]
    for name,p in ready.items():
        calendar=[d for d in dates_all if p.date.min()<=d<=p.date.max()]
        coverage.loc[coverage.experiment.eq(name),'backtest_calendar_weeks']=len(calendar)
        coverage.loc[coverage.experiment.eq(name),'no_signal_cash_weeks']=len(calendar)-p.date.nunique()
        full,pos=bt.precompute(p,lag,calendar,market,next_dates,context)
        positions.append(pos.assign(experiment=name))
        for window,dates in windows(p,calendar).items():
            q=p[p.date.isin(dates)]
            m=engine.evaluate(q)
            wi=[engine.corr(g.prediction,g.target) for _,g in q.groupby('date')]
            m['icir']=float(np.nanmean(wi)/np.nanstd(wi,ddof=1))
            m['zero_prediction_rmse']=float(np.sqrt(np.mean(q.target**2)))
            if name.startswith(('RANKER','CONSENSUS')):m['mae']=m['rmse']=m['zero_prediction_rmse']=np.nan
            metrics.append(dict(experiment=name,window=window,first=q.date.min(),last=q.date.max(),**m))
            w=full[full.date.isin(dates)].copy()
            if window!='FullHistory':
                replay,wp=bt.precompute(q,lag,dates,market,next_dates,context,STATEFUL)
                w=pd.concat([w[~w.strategy.isin(STATEFUL)],replay],ignore_index=True)
                window_positions.append(wp.assign(experiment=name,window=window))
            first=w.date.eq(w.date.min());w.loc[first,'traded_weight']=1-w.loc[first,'cash_weight']
            for cost in [0,10,25]:
                z=w.copy();z['transaction_cost_fraction']=z.traded_weight*cost/10000
                z['net_return']=(1-z.transaction_cost_fraction)*(1+z.gross_return)-1
                for strategy,g in z.groupby('strategy'):
                    summaries.append(dict(experiment=name,window=window,cost_bps=cost,strategy=strategy,**engine.performance(g)))
                if cost==10:
                    weeklies.append(z.assign(experiment=name,window=window,cost_bps=cost))
                    nav=[];dd=[]
                    for strategy,g in z.groupby('strategy',sort=False):
                        g=g.sort_values('date');ds=[g.date.iloc[0]]+list(g.label_end_date)
                        v=np.r_[1.,np.cumprod(1+g.net_return)]
                        nav.append((strategy,ds,v));dd.append((strategy,ds,v/np.maximum.accumulate(v)-1))
                    g=z[z.strategy.eq('Top5')].sort_values('date');ds=[g.date.iloc[0]]+list(g.label_end_date)
                    nav.append(('EqualWeight gross',ds,np.r_[1.,np.cumprod(1+g.benchmark_return)]))
                    stem=f'{name}_{window}'
                    chart(stem+'.svg',f'{name} | {window} | 10bps',nav)
                    chart(stem+'_drawdown.svg',f'{name} | {window} | drawdown',dd,'Drawdown',0)
                    chart_records.append(dict(experiment=name,window=window,nav=f'charts/{stem}.svg',drawdown=f'charts/{stem}_drawdown.svg'))
            if window in ['FullHistory','Validation','FinalTest']:
                benchmarks={d:float(np.expm1(g.target).mean()) for d,g in q.groupby('date')}
                for period in [2,4]:
                    strategy=f'Top5_Rebalance{period}w'
                    for phase in range(period):
                        ds=dates[phase:]
                        z,_=bt.precompute(q[q.date.isin(ds)],lag,ds,market,next_dates,context,[strategy])
                        if phase:
                            pad=pd.DataFrame([dict(date=d,label_end_date=next_dates[d],strategy=strategy,gross_return=0.,benchmark_return=benchmarks.get(d,market[d]),traded_weight=0.,holdings=0,cash_weight=1.,no_signal_cash_week=False,rebalance_event=False,forced_exit_count=0) for d in dates[:phase]])
                            z=pd.concat([pad,z],ignore_index=True)
                        for cost in [0,10,25]:
                            t=z.copy();t['transaction_cost_fraction']=t.traded_weight*cost/10000;t['net_return']=(1-t.transaction_cost_fraction)*(1+t.gross_return)-1
                            phase_rows.append(dict(experiment=name,window=window,strategy=strategy,cost_bps=cost,initial_delay_weeks=phase,**engine.performance(t)))
        print('Evaluated',name,len(calendar),'weeks',flush=True)
    summary=pd.DataFrame(summaries);weekly=pd.concat(weeklies,ignore_index=True)
    csv('coverage.csv',coverage);csv('model_evaluation.csv',pd.DataFrame(metrics));csv('strategy_comparison.csv',summary)
    csv('weekly_backtest.csv',weekly);csv('positions.csv',pd.concat(positions,ignore_index=True));csv('stateful_window_positions.csv',pd.concat(window_positions,ignore_index=True))
    csv('phase_sensitivity.csv',pd.DataFrame(phase_rows))
    csv('full_history_comparison.csv',summary[summary.window.eq('FullHistory')&summary.cost_bps.eq(10)])
    csv('market_trend.csv',pd.DataFrame(dict(date=prices.index,market_index=context['market_index'],risk_on=context['trend'])))
    (OUT/'strategy_rules.json').write_text(json.dumps(dict(strategies=bt.STRATEGIES,labels=LABELS,rules=RULES,consensus='same factor four-family percentile average; unavailable family means no consensus',positive_score='literal prediction > 0; Ranker scores do not imply positive expected return',cost_bps=[0,10,25],windows_start_cash=True),ensure_ascii=False,indent=2))
    reconciliation=source_reconciliation()
    (OUT/'run_manifest.json').write_text(json.dumps(dict(status='backtest_completed_pending_independent_verification',model_series=len(ready),official_available=sum(r['available'] for r in status),official_configurations=16,derived_series=len(components),strategy_count=15,summary_rows=len(summary),phase_cases=len(phase_rows),charts=len(list(CHART.glob('*.svg'))),data_sha256=digest(DATA),source_metric_max_abs_difference=reconciliation,raw_first=str(raw.date.min().date()),raw_last=str(raw.date.max().date()),signal_first=str(coverage.signal_first.min().date()),signal_last=str(coverage.signal_last.max().date()),python=platform.python_version()),indent=2))
    dashboard(summary,chart_records)

def dashboard(summary,records):
    data=summary[summary.cost_bps.eq(10)].copy()
    payload=json.dumps(dict(rows=json.loads(data.to_json(orient='records',date_format='iso')),charts=records),ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>New model evaluation and backtest</title><style>body{font:15px system-ui;margin:30px;color:#172033;background:#f8fafc}select{padding:8px;margin:5px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;background:white;margin-top:16px}p{line-height:1.7}.links a{margin-right:18px}</style><h1>New model evaluation and backtest</h1><p>Six-year rolling training, 26-week refits, 15 strategies. This page shows one-way 10bps; the CSV files include 0/10/25bps. Every window starts from cash. A positive Ranker score does not imply a positive expected return; the benchmark is the gross equal-weight return of the universe. The full history includes the tuning period, and the Final Test has been viewed in earlier experiments.</p><p class="links"><a href="model_evaluation_report.md">Full written report</a><a href="strategy_comparison.csv">All backtest metrics</a><a href="model_evaluation.csv">Prediction metrics</a><a href="coverage.csv">Time coverage</a><a href="source_metric_reconciliation.csv">Reconciliation with delivered results</a></p><label>Model <select id="model"></select></label><label>Window <select id="window"></select></label><table><thead><tr><th>Strategy</th><th>Net annualised return</th><th>Max drawdown</th><th>Sharpe</th><th>Mean traded weight</th><th>Mean cash weight</th></tr></thead><tbody id="table"></tbody></table><img id="nav"><img id="dd"><script>const DATA=PAYLOAD;const m=document.getElementById('model'),w=document.getElementById('window');const pct=x=>(100*x).toFixed(2)+'%';for(const x of [...new Set(DATA.rows.map(r=>r.experiment))])m.add(new Option(x,x));function windows(){w.innerHTML='';for(const x of DATA.charts.filter(r=>r.experiment===m.value))w.add(new Option(x.window,x.window));draw()}function draw(){let rows=DATA.rows.filter(r=>r.experiment===m.value&&r.window===w.value).sort((a,b)=>b.annualized_return-a.annualized_return);document.getElementById('table').innerHTML=rows.map(r=>`<tr><td>${r.strategy}</td><td>${pct(r.annualized_return)}</td><td>${pct(r.max_drawdown)}</td><td>${r.sharpe===null?'—':r.sharpe.toFixed(3)}</td><td>${pct(r.mean_traded_weight)}</td><td>${pct(r.mean_cash_weight)}</td></tr>`).join('');const c=DATA.charts.find(r=>r.experiment===m.value&&r.window===w.value);document.getElementById('nav').src=c.nav;document.getElementById('dd').src=c.drawdown}m.onchange=windows;w.onchange=draw;windows();</script></html>'''.replace('PAYLOAD',payload)
    (OUT/'strategy_dashboard.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
