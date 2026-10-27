"""Recover supplied RF frozen configurations and original XGB development grid."""
from pathlib import Path
import importlib.util, ast, json, hashlib
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source'
OUT=HERE/'replay'
OUT.mkdir(exist_ok=True)
PRED=OUT/'predictions';PRED.mkdir(exist_ok=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def save(family,key,p):
    n={'7_macro':11}.get(key,int(key) if key!='7_macro' else 11)
    p=p.rename(columns={'y_pred':'prediction','y_true':'target'})
    p.to_csv(PRED/f'{family}-R6-F{n}.csv',index=False,float_format='%.17g')

def main():
    rf=load('new_rf',next(SOURCE.rglob('random_forest_standard_6y_26w.py')))
    xgb=load('new_xgb',next(SOURCE.rglob('train_xgboost_5_7_11_45_6y26w.py')))
    rows=[];selections=[]
    original=pd.read_csv(rf.SCRIPT_DIR/'random_forest_5_7_11_45_6y26w_results.csv')
    for key,features in rf.factor_sets(rf.DEFAULT_INPUT).items():
        panel=rf.load_panel(rf.DEFAULT_INPUT,features)
        panel=rf.add_stages(panel,rf.resolve_stage_dates(panel))
        label={'5':'5 factors','7':'7 factors','7_macro':'7 factors + 4 macro factors','45':'45 factors'}[key]
        candidate=ast.literal_eval(original.loc[original.Dataset.eq(label),'selected_params'].iloc[0])
        pred=rf.run_rolling(panel,features,candidate,rf.rolling_folds(panel,6,26),6)
        save('RF',key,pred)
        selections.append(dict(family='RF',feature_set=key,parameters=candidate,source='supplied frozen selected_params'))
        for stage,part in pred.groupby('stage'):
            rows.append(dict(family='RF',Dataset=label,Stage=stage,**rf.stage_metrics(part)))
        print('Recovered RF',key,flush=True)
    for key,features in xgb.factor_sets(xgb.DEFAULT_INPUT).items():
        panel=xgb.load_panel(xgb.DEFAULT_INPUT,features)
        panel=xgb.add_stages(panel,xgb.resolve_stage_dates(panel))
        folds=xgb.rolling_folds(panel,6)
        # Preserve the supplied XGB 105-week selection convention. Only
        # Development signals reach candidate scoring; its last label crosses
        # the selection boundary, which is disclosed in the new report.
        devfolds=[]
        for fold in folds:
            d=panel.loc[(panel.date>=fold['fit_date'])&(panel.date<=fold['end'])&panel.stage.eq('development'),'date']
            if len(d):devfolds.append({**fold,'end':d.max()})
        scores=[];grid=list(xgb.candidates())
        for i,candidate in enumerate(grid,1):
            pred=xgb.run_rolling(panel,features,candidate,devfolds,6)
            scores.append(dict(candidate=i,**candidate,**xgb.stage_metrics(pred[pred.stage.eq('development')])))
            print('XGB development candidate',key,i,flush=True)
        scores=pd.DataFrame(scores)
        scores['passes_gate']=(scores['Rank IC']>0)&(scores['Rank ICIR']>=.05)&(scores['Positive Rank IC']>=.5)
        scores.to_csv(OUT/f'xgb_candidates_{key}.csv',index=False)
        eligible=scores[scores.passes_gate & scores['Top5 Excess'].notna()].sort_values(['Top5 Excess','Rank IC','Rank ICIR'],ascending=False)
        if eligible.empty:
            selections.append(dict(family='XGB',feature_set=key,status='no_candidate_passed'));continue
        candidate=grid[int(eligible.iloc[0].candidate)-1]
        pred=xgb.run_rolling(panel,features,candidate,folds,6)
        save('XGB',key,pred)
        selections.append(dict(family='XGB',feature_set=key,parameters=candidate,source='original 12-candidate development selection'))
        label={'5':'5 factors','7':'7 factors','7_macro':'7 factors + 4 macro factors','45':'45 factors'}[key]
        for stage,part in pred.groupby('stage'):
            rows.append(dict(family='XGB',Dataset=label,Stage=stage,**xgb.stage_metrics(part)))
        print('Recovered XGB',key,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'recovered_source_metrics.csv',index=False)
    (OUT/'selections.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')

if __name__=='__main__':main()
