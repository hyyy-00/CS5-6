"""Replay every supplied score over its own complete available signal history."""
from pathlib import Path
import sys,json,zipfile,hashlib,html,platform
import pandas as pd
import numpy as np
import run as engine
from portfolio_strategies import NEW_STRATEGIES,STATEFUL,CONSENSUS,COMPONENTS,RULES,inputs,capped_weights,consensus_predictions

ROOT=engine.ROOT
OUT=Path(__file__).resolve().parent/'full_history'
PREDS=OUT/'predictions'
NS=OUT/'native_predictions'
STRATEGIES=['Top5','Top10','Top5_PositiveScore','Top10_PositiveScore','Top5_NoLoss4w','Top10_NoLoss4w','Top5_NoLoss13w','Top10_NoLoss13w','Top5_RankWeighted']+NEW_STRATEGIES
checks=0

def csv(path,frame):frame.to_csv(path,index=False)

def load_sources():
    records=[];sources=[]
    for file in sorted(NS.glob('*.csv')):
        p=pd.read_csv(file,parse_dates=['date'])
        records.append((file.stem,p,'native_code_replay',str(file)))
    with zipfile.ZipFile(ROOT/'xgboost-ranker-v2.zip') as z:
        for n in sorted(z.namelist()):
            if not (n.endswith('.csv') and 'prediction' in n.lower() and not n.startswith('__MACOSX') and '.venv/' not in n):continue
            d=pd.read_csv(z.open(n),parse_dates=['date'])
            source=n.split('/data/')[-1]
            source_hash=hashlib.sha256(z.read(n)).hexdigest()
            sources.append(dict(source=source,rows=len(d),first=d.date.min(),last=d.date.max(),sha256=source_hash))
            if 'ensemble_ranking_score' in d:
                prefix=('RANKER-V2' if '/XGboost_ranker/' in n else 'RANKER-CONTINUOUS-FINAL' if '/continuous_target_final_run/' in n
                    else 'RANKER-C' if '/experiment_C_continuous/' in n else 'RANKER-D')
                for component,col in [('ENSEMBLE','ensemble_ranking_score'),('M1','model_1_raw_score'),('M2','model_2_raw_score'),('M3','model_3_raw_score')]:
                    p=d[['date','ticker','target',col]].rename(columns={col:'prediction'})
                    records.append((f'{prefix}-{component}',p,'supplied_prediction',source+'::'+col))
            else:
                keys=[c for c in ['factor_set_name','training_mode','config_id'] if c in d]
                for group,p in d.groupby(keys,dropna=False,sort=True):
                    vals=group if isinstance(group,tuple) else (group,)
                    prefix='RANKER-HOLDOUT' if '/rankic_optimization/' in n else 'RANKER-LEGACY' if '/legacy_priority7/' in n else 'RANKER-HISTORY'
                    name=prefix+'-'+'-'.join(str(v) for v in vals)
                    p=p[['date','ticker','actual_return','ranking_score']].rename(columns={'actual_return':'target','ranking_score':'prediction'})
                    records.append((name,p,'supplied_prediction',source))
    csv(OUT/'source_files.csv',pd.DataFrame(sources))
    return records

def prepare(records,raw,write_audits=True):
    global checks
    dates=sorted(raw.date.unique());next_dates=dict(zip(dates[:-1],dates[1:]))
    indexed=raw.set_index(['date','ticker']).adj_close
    label=raw[['date','ticker','target','adj_close']].rename(columns={'target':'data_target'})
    label['label_end_date']=label.date.map(next_dates)
    label['next_price']=indexed.reindex(pd.MultiIndex.from_arrays([label.label_end_date,label.ticker])).to_numpy()
    ready={};coverage=[];excluded=[];gaps=[]
    for name,p,origin,source in records:
        assert not p.duplicated(['date','ticker']).any(),name
        p=p.drop(columns=['label_end_date'],errors='ignore').merge(label,on=['date','ticker'],how='left',validate='one_to_one',indicator=True)
        assert p._merge.eq('both').all(),name
        assert np.isfinite(p[['prediction','target']].to_numpy()).all(),name
        assert np.max(abs(p.target-p.data_target))<1e-10,name
        known=p.label_end_date.notna()&np.isfinite(p.next_price)
        error=np.max(abs(np.log(p.loc[known,'next_price']/p.loc[known,'adj_close'])-p.loc[known,'target']))
        assert error<1e-10,(name,error)
        bad=p.loc[~known,['date','ticker']].copy();bad['experiment']=name
        bad['reason']=np.where(p.loc[~known,'label_end_date'].isna(),'terminal_next_week_unavailable','next_global_week_stock_price_missing')
        excluded.append(bad)
        first,last,observed=p.date.min(),p.date.max(),set(p.date)
        missing_dates=[d for d in dates if first<=d<=last and d not in observed]
        gaps.extend(dict(experiment=name,date=d,reason='source_has_no_prediction_on_this_week') for d in missing_dates)
        coverage.append(dict(experiment=name,origin=origin,source=source,input_rows=len(p),input_weeks=p.date.nunique(),input_first=p.date.min(),input_last=p.date.max(),
            usable_rows=int(known.sum()),excluded_rows=int((~known).sum()),signal_first=p.loc[known,'date'].min(),signal_last=p.loc[known,'date'].max(),
            realization_last=p.loc[known,'label_end_date'].max(),weeks=p.loc[known,'date'].nunique(),missing_source_weeks=len(missing_dates),label_max_error=error))
        q=p.loc[known].drop(columns=['_merge','data_target','adj_close','next_price'])
        ready[name]=q;csv(PREDS/f'{name}.csv',q);checks+=5
    if write_audits:
        csv(OUT/'coverage.csv',pd.DataFrame(coverage));csv(OUT/'excluded_rows.csv',pd.concat(excluded,ignore_index=True))
        csv(OUT/'missing_source_weeks.csv',pd.DataFrame(gaps,columns=['experiment','date','reason']))
    return ready,pd.DataFrame(coverage)

def precompute(p,lagreturns,calendar=None,market=None,next_dates=None,context=None,strategies=None):
    rows=[];positions=[]
    grouped={d:g.sort_values(['prediction','ticker'],ascending=[False,True],kind='stable') for d,g in p.groupby('date',sort=True)}
    weeks=[(d,grouped.get(d,p.iloc[:0])) for d in (calendar if calendar is not None else sorted(grouped))]
    for strategy in (STRATEGIES if strategies is None else strategies):
        k=5 if strategy.startswith('Top5') else 10
        previous={};prior_returns={};prior_cash=1.;last_rebalance=None
        for week_index,(date,g) in enumerate(weeks):
            growth=prior_cash+sum(w*(1+prior_returns[t]) for t,w in previous.items())
            drift={t:w*(1+prior_returns[t])/growth for t,w in previous.items()}
            selected=g
            if 'PositiveScore' in strategy:selected=selected[selected.prediction.gt(0)]
            if 'NoLoss' in strategy:
                lag=4 if '4w' in strategy else 13
                selected=selected[selected.ticker.map(lagreturns[lag].loc[date]).ge(0)]
            if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                selected=selected[selected.ticker.map(context['vol'].loc[date]).notna()]
            rebalance_event=True;forced_exit_count=0
            if strategy=='Top5_Buffer10':
                keep=[t for t in g.head(10).ticker if t in previous]
                additions=[t for t in g.ticker if t not in keep][:k-len(keep)]
                weights=dict.fromkeys(keep+additions,1/k)
            elif strategy in ['Top5_Rebalance2w','Top5_Rebalance4w']:
                period=2 if '2w' in strategy else 4
                if not len(g):
                    weights={};last_rebalance=None
                elif not previous or last_rebalance is None or week_index-last_rebalance>=period:
                    weights=dict.fromkeys(g.head(k).ticker,1/k);last_rebalance=week_index
                else:
                    rebalance_event=False
                    usable=context['stock_returns'].loc[date]
                    weights={t:w for t,w in drift.items() if pd.notna(usable.get(t,np.nan))}
                    forced_exit_count=len(drift)-len(weights)
            elif strategy=='Top5_Trend26w' and not context['trend'].loc[date]:weights={}
            else:
                selected=selected.head(k)
                if strategy in ['Top5_LowVol26w','Top5_RankRisk26w']:
                    scores={t:(5-rank if strategy=='Top5_RankRisk26w' else 1)/max(float(context['vol'].loc[date,t]),1e-8)
                            for rank,t in enumerate(selected.ticker)}
                    weights=capped_weights(scores,budget=len(selected)/k)
                else:
                    weights=(dict(zip(selected.ticker,[(5-rank)/15 for rank in range(len(selected))]))
                             if strategy=='Top5_RankWeighted' else dict.fromkeys(selected.ticker,1/k))
            cash_weight=1-sum(weights.values())
            source_returns=dict(zip(g.ticker,np.expm1(g.target)))
            returns={t:source_returns[t] if t in source_returns else float(context['stock_returns'].loc[date,t]) for t in weights}
            assert all(np.isfinite(list(returns.values())))
            assert -1e-12<=cash_weight<=1+1e-12
            traded=sum(abs(weights.get(t,0)-drift.get(t,0)) for t in set(weights)|set(drift))
            rows.append(dict(date=date,label_end_date=g.label_end_date.iloc[0] if len(g) else next_dates[date],strategy=strategy,
                gross_return=sum(weights[t]*returns[t] for t in weights),benchmark_return=np.expm1(g.target).mean() if len(g) else market[date],
                traded_weight=traded,holdings=len(weights),cash_weight=cash_weight,no_signal_cash_week=not len(g),
                rebalance_event=rebalance_event,forced_exit_count=forced_exit_count))
            positions.extend(dict(date=date,strategy=strategy,ticker=t,weight=w,realized_return=returns[t]) for t,w in weights.items())
            previous,prior_returns,prior_cash=weights,returns,cash_weight
    return pd.DataFrame(rows),pd.DataFrame(positions)

def windows(p,calendar=None):
    dates=sorted(p.date.unique()) if calendar is None else calendar
    result={'FullHistory':dates}
    for n in [13,26,52,104,156]:
        if len(dates)>=n:result[f'Last{n}w']=dates[-n:]
    for year in sorted(set(pd.Timestamp(d).year for d in dates)):result[f'Year{year}']=[d for d in dates if pd.Timestamp(d).year==year]
    for stage,start,end in [('Development3y','2019-09-01','2024-09-01'),('Development6y','2022-09-01','2024-09-01'),('Validation','2024-09-01','2025-09-01'),('FinalTest','2025-09-01','2026-09-01')]:
        chosen=[d for d in dates if pd.Timestamp(start)<=d<pd.Timestamp(end)]
        if chosen:result[stage]=chosen
    return result

def render(p,folder,name,title):
    series=[]
    for strategy,g in p.groupby('strategy',sort=False):
        ds=[g.date.iloc[0]]+list(g.label_end_date)
        series.append((strategy,ds,np.r_[1.,np.cumprod(1+g.net_return)]))
    g=p[p.strategy.eq('Top5')]
    series.append(('EqualWeight',[g.date.iloc[0]]+list(g.label_end_date),np.r_[1.,np.cumprod(1+g.benchmark_return)]))
    engine.svg_chart(folder/f'{name}.svg',title,series)

def main():
    global checks
    OUT.mkdir(exist_ok=True);PREDS.mkdir(exist_ok=True);chart=OUT/'charts';chart.mkdir(exist_ok=True)
    engine.COLORS=engine.COLORS[:8]+['#a16207','#7c3aed','#0f766e','#c026d3','#0284c7','#be123c','#65a30d','#111827']
    raw=pd.read_csv(ROOT/'test_45_factors_weekly.csv',parse_dates=['date'])
    ready,coverage=prepare(load_sources(),raw)
    consensus,joint=consensus_predictions(ready)
    csv(OUT/'consensus_components.csv',joint)
    consensus_ready,consensus_coverage=prepare([(CONSENSUS,consensus,'derived_consensus','equal percentile average: '+','.join(COMPONENTS))],raw,write_audits=False)
    assert consensus_coverage.excluded_rows.sum()==0
    ready.update(consensus_ready);coverage=pd.concat([coverage,consensus_coverage],ignore_index=True)
    dates_all=sorted(raw.date.unique())
    consensus_gaps=[d for d in dates_all if consensus.date.min()<=d<=consensus.date.max() and d not in set(consensus.date)]
    gaps=pd.read_csv(OUT/'missing_source_weeks.csv',parse_dates=['date'])
    gaps=pd.concat([gaps,pd.DataFrame([dict(experiment=CONSENSUS,date=d,reason='consensus_component_missing') for d in consensus_gaps])],ignore_index=True)
    csv(OUT/'missing_source_weeks.csv',gaps)
    engine.dump(OUT/'strategy_rules.json',dict(strategies=STRATEGIES,rules=RULES,consensus_components=COMPONENTS,
        consensus_component_weights=[.25]*4,selection='fixed R3-F11 designs plus original Ranker, no outcome-based selection',risk_window=26,stock_cap=.35))
    status=[]
    for family in ['RF','XGB','RIDGE']:
        for design in ['R3','R6','FIXED']:
            for factors in [7,11,45]:
                name=f'{family}-{design}-F{factors}'
                status.append(dict(experiment=name,status='native_replay_available' if name in ready else 'no_candidate_passed',
                    source='original code + recorded parameters',available=name in ready))
    status.append(dict(experiment='RANKER-V2-ENSEMBLE',status='supplied_prediction_available',source='XGboost_ranker/data/stock_predictions.csv',available=True))
    csv(OUT/'official_model_status.csv',pd.DataFrame(status))
    prices=raw.pivot(index='date',columns='ticker',values='adj_close')
    context=inputs(prices)
    csv(OUT/'market_trend.csv',pd.DataFrame(dict(date=prices.index,market_index=context['market_index'],
        mean26=context['market_index'].rolling(26,min_periods=26).mean(),risk_on=context['trend'])))
    lagreturns={n:prices/prices.shift(n)-1 for n in [4,13]}
    rawdates=list(prices.index);next_dates=dict(zip(rawdates[:-1],rawdates[1:]))
    future=prices.shift(-1);valid_prices=(prices.notna()&future.notna())
    market=(future/prices-1).where(valid_prices).mean(axis=1).to_dict()
    common_keys=set.intersection(*(set(zip(p.date,p.ticker)) for p in ready.values()))
    metrics=[];summaries=[];weekly=[];holdings=[];common_weekly=[];stateful_holdings=[]
    for name,p in ready.items():
        calendar=[d for d in rawdates if p.date.min()<=d<=p.date.max()]
        coverage.loc[coverage.experiment.eq(name),'backtest_calendar_weeks']=len(calendar)
        coverage.loc[coverage.experiment.eq(name),'no_signal_cash_weeks']=len(calendar)-p.date.nunique()
        for window,dates in windows(p,calendar).items():
            q=p[p.date.isin(dates)]
            m=engine.evaluate(q)
            if name.startswith('RANKER') or name==CONSENSUS:m['mae']=m['rmse']=np.nan
            metrics.append(dict(experiment=name,window=window,first=q.date.min(),last=q.date.max(),**m))
        full,pos=precompute(p,lagreturns,calendar,market,next_dates,context)
        holdings.append(pos.assign(experiment=name))
        for window,dates in windows(p,calendar).items():
            base=full[full.date.isin(dates)].copy()
            if window!='FullHistory':
                replay,window_pos=precompute(p[p.date.isin(dates)],lagreturns,dates,market,next_dates,context,STATEFUL)
                base=pd.concat([base[~base.strategy.isin(STATEFUL)],replay],ignore_index=True)
                stateful_holdings.append(window_pos.assign(experiment=name,window=window))
            # All windows start from cash. Subsequent drift is unchanged inside a contiguous interval.
            mask=base.date.eq(base.date.min());base.loc[mask,'traded_weight']=1-base.loc[mask,'cash_weight']
            for cost in [0,10,25]:
                w=base.copy();w['transaction_cost_fraction']=w.traded_weight*cost/10000
                w['net_return']=(1-w.transaction_cost_fraction)*(1+w.gross_return)-1
                for strategy,g in w.groupby('strategy'):
                    values=engine.performance(g)
                    nav=np.r_[1.,np.cumprod(1+g.net_return)]
                    assert abs(values['total_return']-(nav[-1]-1))<1e-10
                    assert abs(values['max_drawdown']-(nav/np.maximum.accumulate(nav)-1).min())<1e-10
                    checks+=2
                    summaries.append(dict(experiment=name,window=window,cost_bps=cost,strategy=strategy,**values))
                if cost==10:
                    weekly.append(w.assign(experiment=name,window=window,cost_bps=cost))
                    render(w,chart,f'{name}_{window}',f'{name} | {window} | 10bps')
        q=p[[k in common_keys for k in zip(p.date,p.ticker)]]
        if len(q):
            # Auxiliary comparison keeps the previous sparse shared-date convention.
            w,_=precompute(q,lagreturns,context=context);w['transaction_cost_fraction']=w.traded_weight*.001
            w['net_return']=(1-w.transaction_cost_fraction)*(1+w.gross_return)-1
            common_weekly.append(w.assign(experiment=name,window='CommonAuxiliary',cost_bps=10))
            for strategy,g in w.groupby('strategy'):summaries.append(dict(experiment=name,window='CommonAuxiliary',cost_bps=10,strategy=strategy,**engine.performance(g)))
        print('Full history backtested',name,p.date.min().date(),p.date.max().date(),p.date.nunique(),flush=True)
    summary=pd.DataFrame(summaries);week=pd.concat(weekly,ignore_index=True)
    csv(OUT/'coverage.csv',coverage)
    csv(OUT/'strategy_comparison.csv',summary);csv(OUT/'model_evaluation.csv',pd.DataFrame(metrics))
    csv(OUT/'weekly_backtest.csv',week);csv(OUT/'positions.csv',pd.concat(holdings,ignore_index=True))
    csv(OUT/'stateful_window_positions.csv',pd.concat(stateful_holdings,ignore_index=True))
    if common_weekly:csv(OUT/'common_auxiliary_weekly.csv',pd.concat(common_weekly,ignore_index=True))
    # Full-history native-family plots use original start dates, without imposing a common starting date.
    for family in ['RF','RIDGE','XGB']:
        for factors in [7,11,45]:
            lines=[]
            for design in ['R3','R6','FIXED']:
                exp=f'{family}-{design}-F{factors}'
                g=week[week.experiment.eq(exp)&week.window.eq('FullHistory')&week.strategy.eq('Top5')]
                if len(g):lines.append((design,[g.date.iloc[0]]+list(g.label_end_date),np.r_[1.,np.cumprod(1+g.net_return)]))
            if lines:engine.svg_chart(chart/f'{family}_F{factors}_native_history_windows.svg',f'{family} F{factors} | original full histories | Top5 | unequal starts',lines)
    fullsummary=summary[summary.window.eq('FullHistory')&summary.cost_bps.eq(10)]
    csv(OUT/'full_history_comparison.csv',fullsummary)
    table=coverage[['experiment','origin','input_first','input_last','signal_first','signal_last','realization_last','weeks','usable_rows','excluded_rows']].to_html(index=False)
    links=''.join(f'<li><a href="charts/{f.name}">{html.escape(f.stem)}</a></li>' for f in sorted(chart.glob('*.svg')))
    OUT.joinpath('index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}img{max-width:1120px;width:100%}</style><h1>Full original history backtest of all models</h1><p>Each model is backtested over its own prediction start and end dates, with no common 2022 cut-off and no common-sample cropping of the main backtest. Predictions in the final week lack a realised next-week price; they are kept in the audit records and are not counted as realised returns.</p><p><a href="report.md">Report</a> · <a href="coverage.csv">Model time coverage</a> · <a href="full_history_comparison.csv">Full-history metrics</a></p><h2>Data period and coverage by model</h2>'+table+'<h2>Official Ranker full history</h2><img src="charts/RANKER-V2-ENSEMBLE_FullHistory.svg"><h2>All charts</h2><ul>'+links+'</ul>',encoding='utf-8')
    report=f'''# Full-history backtest of all models (corrected version)

This version corrects the previous version's error of cropping every model to after 2022 and restores the configurations of the original project code. The main backtest uses every signal each model actually has: {len(ready)} model prediction series. For each series, coverage.csv lists the input period, the period with realisable returns, excluded records and the source. Full-history returns with different start dates are not a fair ranking over the same market period; the common period is saved separately as the CommonAuxiliary auxiliary result.

## Inputs and timing rules

All 7 per-stock prediction files in the archive are read; fixed/rolling outputs are grouped by factor set and mode, and a refitted model_id is an update of the same configuration rather than a separate short history. All 4 ensemble files are backtested for the three components and the ensemble (identical sources are kept so each file can be reconciled). The original RF restores the full Development, Validation and Final stages from the parameters frozen in README1 and random_forest.py; the original Ridge is restored with the original 6-alpha grid and the original 3-year/6-year scripts, and the 45-factor rolling groups that failed the gate are not given fabricated Final results; the original regression XGB is restored with the original 12 candidates, the original monthly rolling and fixed-stage rules. The previous version's two-candidate experiment is not used as a substitute for the original models.

Histories that start on 2019-08-30 or 2019-09-06 in the original predictions are kept in full; the six-year-history design actually starts in 2022; the holdout source itself starts in 2024. Raw prices from 2016-09-02 are used for training and past-return filters, not as the start of existing predictions. The 2026-08-28 signal has no next-week price yet, so no return is invented and it is recorded in excluded_rows.csv; the last verifiable signal is 2026-08-21, with its return realised on 2026-08-28. Records whose next-global-week price is missing for the same stock are excluded individually, and the main backtest is not cropped because another model lacks records.

Under its end-of-stage label-maturity rule, the original RF code does not output a few weeks that span stages; these missing weeks are listed in missing_source_weeks.csv and no signals are fabricated. In those weeks the main backtest sells the previous holdings and holds cash, charging the sell cost, and buys again in the next signal week. Annual windows, rolling windows and annualisation keep these real calendar weeks, and coverage.csv distinguishes prediction weeks from backtest calendar weeks. The 3-year and 6-year Ridge rolling 45-factor experiments failed the original screening gate, so the original code has no Validation/Final predictions; they remain untested and no strategy results are invented. 26 of the original 28 formal configurations are available and restored over their full history; archive histories, components and label comparisons are also kept, file by file even when sources or signals are identical.

## Strategies and windows

Each model runs 15 strategies: the original 9 (equal-weight Top5/Top10, score filters, 4/13-week loss filters, rank-weighted Top5) are kept, and low-volatility weighting, combined rank-risk weighting, a top-10 retention buffer, 2-week/4-week rebalancing and a 26-week market moving-average filter are added. Risk weighting uses only the 26-week standard deviation of simple returns up to the signal date, capped at 35% per stock; with fewer than 5 stocks the equity budget is count/5 and the rest is cash. See strategy_rules.json for the exact rules.

Beyond the original 58 prediction series, a four-model equal-weight rank-percentile consensus series is added (CONSENSUS-R3-F11-4MODELS): the fixed RF/Ridge/XGB three-year 11-factor models plus the original Ranker, with consensus predictions only for stocks and dates shared by all four. It does not replace any model's full history. In weeks where a component is missing the consensus holds cash and predictions are not forward-filled. The consensus also runs all 15 strategies, and the treatment of the 26 available original formal configurations is unchanged.

The buffer and 2/4-week rebalancing strategies are path dependent and are replayed from cash in every window; in non-rebalancing weeks holdings drift naturally and no trades follow new predictions. When the original model has no signal the portfolio is liquidated; when signals return it buys immediately and resets the rebalance clock. When a held stock has no computable next-week return the holding is removed and recorded in forced_exit_count; this is an evaluability treatment and does not represent an executable delisting sale. All strategies still use the universe of stocks with verifiable returns, which introduces an ex-post evaluability selection bias.

Each model covers FullHistory, the last 13/26/52/104/156 weeks (when the history is long enough), each calendar year, Development3y/Development6y, Validation and FinalTest. Windows are sliced from each model's own available dates with no artificial common start; one-way costs are 0/10/25bps. Each window starts from cash with net value 1, both buy and sell traded weights are charged, and later turnover is computed from weights after holding returns have drifted; the end is marked to market without a forced liquidation cost. Net return = (1 - cost fraction) * (1 + gross portfolio return) - 1. The benchmark is the gross equal-weight return of the model's evaluable stocks in that week.

## Result files

- coverage.csv: actual period and row count for each model; this is the entry point for checking time coverage.
- source_files.csv: the 7 archive prediction sources and their hashes; predictions/: complete valid predictions; excluded_rows.csv: all records whose returns cannot be verified.
- full_history_comparison.csv: full-history 10bps metrics for the 15 strategies of every model.
- strategy_comparison.csv: all windows, strategies and costs; model_evaluation.csv: model metrics for all periods.
- weekly_backtest.csv, positions.csv: weekly returns and full-history per-stock holdings; charts/: strategy curves for every model and window, plus full-history curves of the original training windows.
- stateful_window_positions.csv: holdings of the buffer/2-week/4-week strategies replayed from cash in each non-full-history window, for independent cost checks.
- consensus_components.csv: the four rank percentiles and the combined score; market_trend.csv: market trend condition using past data only.
- common_auxiliary_weekly.csv: auxiliary comparison on shared dates and stocks; the main tables do not depend on this file.

## Interpretation limits and reproduction

Original predictions are replayed directly; RF/Ridge/regression XGB are rebuilt from the original code and configurations in this environment, not recovered original trained weights. Historical results in the original READMEs may depend on library versions, and this batch of records takes precedence. The full history includes stages originally used for parameter selection and cannot be called unseen data throughout; the original RF configuration was selected on Validation, and the original XGB convention of a Development final-week label crossing the stage boundary is kept, with stage results reported separately. Existing factors and the stock universe were screened on the full period, so the results remain exploratory. The executability of same-close factors with same-close entry is not verified, and costs are scenario estimates only.

Reproduction: run restore_ridge.py, native_replay.py RF and native_replay.py XGB first, then full_history_backtest.py. All scripts are in the evaluation_backtest directory; the original data and training code are unchanged.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    engine.dump(OUT/'run_manifest.json',dict(status='completed',protocol='native_full_history_v3_multistrategy',model_series=len(ready),original_series=58,derived_consensus_series=1,
        total_native_rows=int(coverage.input_rows.sum()),usable_rows=int(coverage.usable_rows.sum()),excluded_rows=int(coverage.excluded_rows.sum()),
        signal_first=coverage.signal_first.min(),signal_last=coverage.signal_last.max(),realization_last=coverage.realization_last.max(),
        summary_rows=len(summary),chart_count=len(list(chart.glob('*.svg'))),in_process_checks=checks,strategies=STRATEGIES,
        full_history_common_crop=False,common_auxiliary_keys=len(common_keys),source_sha256=engine.ridge.sha(ROOT/'test_45_factors_weekly.csv'),
        archive_sha256=engine.ridge.sha(ROOT/'xgboost-ranker-v2.zip'),python=platform.python_version()))

if __name__=='__main__':main()
