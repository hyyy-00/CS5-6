"""Independently verify prediction statistics from persisted score histories."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def main():
    records=pd.read_csv(OUT/'model_evaluation.csv',parse_dates=['first','last'])
    predictions={p.stem:pd.read_csv(p,parse_dates=['date']) for p in (OUT/'predictions').glob('*.csv')}
    count=0;maximum=0.
    for r in records.itertuples():
        p=predictions[r.experiment];p=p[(p.date>=r.first)&(p.date<=r.last)]
        ic=[];ric=[]
        for _,g in p.groupby('date'):
            a=g.prediction.to_numpy();b=g.target.to_numpy()
            if np.ptp(a)>0 and np.ptp(b)>0:
                ic.append(np.corrcoef(a,b)[0,1])
                ric.append(np.corrcoef(g.prediction.rank(method='average'),g.target.rank(method='average'))[0,1])
            else:
                ic.append(np.nan);ric.append(np.nan)
        ic=np.asarray(ic);ric=np.asarray(ric)
        good=ric[np.isfinite(ric)]
        vals=[len(p),len(ic),np.nanmean(ic),np.nanmean(ric),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.nanmean(ric)/np.nanstd(ric,ddof=1),np.mean(good>0)]
        expected=[r.rows,r.weeks,r.ic,r.rank_ic,r.icir,r.rank_icir,r.positive_rank_ic]
        if not r.experiment.startswith(('RANKER','CONSENSUS')):
            vals.extend([np.abs(p.prediction-p.target).mean(),np.sqrt(np.mean((p.prediction-p.target)**2)),np.sqrt(np.mean(p.target**2))]);expected.extend([r.mae,r.rmse,r.zero_prediction_rmse])
        e=float(np.max(np.abs(np.asarray(vals)-np.asarray(expected))))
        assert e<1e-10,(r.experiment,r.window,e)
        maximum=max(maximum,e);count+=len(vals)
    path=OUT/'verification.json';v=json.loads(path.read_text());v['prediction_metric_checks']=count;v['prediction_metric_max_abs_difference']=maximum
    v.setdefault('portfolio_checks',v['checks'])
    v['checks']=v['portfolio_checks']+count;v['max_abs_difference']=max(v['max_abs_difference'],maximum)
    v['scope']+='; independent IC/RankIC/ICIR/positive weeks/MAE/RMSE/zero baseline across all windows'
    path.write_text(json.dumps(v,indent=2))
    mf=OUT/'run_manifest.json';m=json.loads(mf.read_text());m['independent_checks']=v['checks'];mf.write_text(json.dumps(m,indent=2))
    print(count,maximum)

if __name__=='__main__':main()
