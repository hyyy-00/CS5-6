"""Seven-factor statistical screening and a specified five-factor candidate set.
Requires pandas and numpy. Reads data/test_45_factors_weekly.csv by default.
The five-factor membership is an explicit design input, not selected by IC gates.
Its financial-dimension rationale does not establish statistical optimality.
Outputs CSV/JSON to --output (default: outputs/ beside this script).
"""
from pathlib import Path
import json, hashlib, argparse
import pandas as pd
import numpy as np
# 1. Read CLI paths. Only the raw CSV is required; no historical results are loaded.
parser=argparse.ArgumentParser(description=__doc__)
ROOT=Path(__file__).resolve().parents[2]
parser.add_argument("--input",type=Path,default=ROOT/"data"/"test_45_factors_weekly.csv")
parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"outputs")
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
full=pd.read_csv(args.input,parse_dates=["date"])
# 2. Define the original 45-feature universe. Date, ticker and labels are excluded.
factors=['ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_120d', 'momentum_20_5', 'overnight_gap', 'intraday_return', 'ema20_distance', 'ema60_distance', 'ema20_60_spread', 'rsi14_centered', 'macd_hist_atr', 'breakout_position_60', 'realized_vol_5', 'realized_vol_20', 'vol_ratio_5_20', 'parkinson_vol_20', 'overnight_vol_20', 'downside_vol_20', 'volume_zscore_20', 'volume_trend_5_20', 'amihud_illiq_20', 'signed_volume_pressure_20', 'close_location_value', 'body_range_ratio', 'cs_rank_ret20', 'cs_rank_ret60', 'cs_rank_vol20', 'cs_rank_amihud20', 'trend_r2_20', 'return_autocorr_20', 'volume_cv_20', 'return_volume_corr_20', 'gap_followthrough_20', 'revenue_growth_yoy', 'gross_margin_change_yoy', 'operating_profitability', 'cashflow_to_assets', 'debt_to_assets', 'asset_growth_yoy', 'macro_vix_level', 'macro_vix_change_5obs', 'macro_yield_spread_10y_2y', 'macro_dgs10_change_5obs']
macro=[f for f in factors if f.startswith('macro_')]
fund=['revenue_growth_yoy','gross_margin_change_yoy','operating_profitability','cashflow_to_assets','debt_to_assets','asset_growth_yoy']
# Macro features must be constant across stocks within each observation week.
assert full.groupby('date')[macro].nunique().le(1).all().all()
mac=full.groupby('date')[macro].first().reset_index()
# Validate the supplied complete 66-stock panel; this script does not impute missing values.
d=full.copy()
assert d.ticker.nunique()==66 and not d.isna().any().any()
assert not d.duplicated(['date','ticker']).any()
# Exclude the final signal week: its following-week price is absent from the supplied data.
d=d[(d.date>=pd.Timestamp('2016-09-02')) & (d.date<pd.Timestamp('2026-08-28'))].copy()
# Verify the four known duplicate columns before excluding them from selection.
dup={'cs_rank_ret20':'ret_20d','cs_rank_ret60':'ret_60d','cs_rank_vol20':'realized_vol_20','cs_rank_amihud20':'amihud_illiq_20'}
for f,g in dup.items():assert d[f].equals(d[g])
# 3. Compute weekly cross-sectional Spearman Rank IC against next-week target.
# Average ranks handle ties; macro features have no cross-sectional variation.
weekly=[]
for dt,g in d.groupby('date'):
 for f in factors:
  if f in macro:continue
  a=g[[f,'target']].dropna()
  if len(a)<20 or a[f].nunique()<5 or a.target.nunique()<2:continue
  ic=a[f].rank(method='average').corr(a.target.rank(method='average'))
  weekly.append({'date':dt,'year':dt.year,'factor':f,'n':len(a),'rank_ic':ic})
# 4. Aggregate weekly results. Eligible years require at least 40 valid IC weeks.
w=pd.DataFrame(weekly)
annual=w.groupby(['factor','year']).agg(weeks=('rank_ic','size'),mean_ic=('rank_ic','mean')).reset_index()
# Rank ICIR uses sample standard deviation (ddof=1) and is not annualised.
# Annual agreement compares each eligible annual sign with the full-sample sign.
rows=[]
for f in factors:
 s=w.loc[w.factor==f,'rank_ic'];mean=s.mean();std=s.std(ddof=1)
 sign=1 if mean>=0 else -1
 a=annual[(annual.factor==f)&(annual.weeks>=40)]
 ratio=float((a.mean_ic*sign>0).mean()) if len(a) else np.nan
 ir=mean/std if std>0 else np.nan
 coverage=float(d[f].notna().mean())
 positive=float((s*sign>0).mean()) if len(s) else np.nan
 direction=('Positive' if sign==1 else 'Negative') if len(s) else 'N/A'
 if f in dup:status='Duplicate removed';reason='Identical to '+dup[f]
 elif f in macro:status='Macro separately retained';reason='Constant across stocks within each week; Rank IC is undefined. Optional market-state input.'
 elif coverage<.8:status='Insufficient data';reason='Coverage below 80%.'
 elif len(s)<104 or len(a)<3:status='Insufficient data';reason='Too few valid weeks or eligible years.'
 elif abs(mean)>=.01 and abs(ir)>=.08 and ratio>=.6:status='Selected in 7';reason='Full-sample |Rank IC| >=0.01, |Rank ICIR| >=0.08, and annual sign agreement >=60%.'
 elif abs(mean)<.005:status='Weak-signal alternative';reason='Full-sample |Rank IC| <0.005; optional for compact experiments, not necessarily useless in models.'
 else:status='Alternative';reason='Does not meet all signal and annual stability thresholds; retained as an alternative for nonlinear or interaction effects.'
 rows.append({'factor':f,'name':f,'type':('Fundamental' if f in fund else 'Macro' if f in macro else 'Technical'),'status':status,'direction':direction,'coverage':coverage,'weeks':len(s),'mean_ic':mean,'icir':ir,'oriented_positive_weeks':positive,'eligible_years':len(a),'same_direction_years':int((a.mean_ic*sign>0).sum()),'year_consistency':ratio,'reason':reason})
# 5. Remove redundant candidates greedily, ordered by absolute Mean Rank IC.
# This list contains both technical and fundamental factors despite its legacy name.
tech=[f for f in factors if f not in macro and f not in dup]
corr=pd.DataFrame(np.mean([g[tech].corr(method='spearman').to_numpy() for _,g in d.groupby('date')],axis=0),index=tech,columns=tech)
picked=[]
for r in sorted([r for r in rows if r['status']=='Selected in 7'],key=lambda r:(-abs(r['mean_ic']),r['factor'])):
 match=next((p for p in picked if abs(corr.loc[r['factor'],p])>=.85),None)
 if match:r['status']='Correlated alternative';r['reason']=f'Passes signal thresholds but mean weekly correlation with {match} is {corr.loc[r["factor"],match]:.3f}; retain the factor with higher absolute Mean Rank IC.'
 else:picked.append(r['factor'])
order={s:i for i,s in enumerate(['Selected in 7','Alternative','Correlated alternative','Weak-signal alternative','Insufficient data','Macro separately retained','Duplicate removed'])}
rows.sort(key=lambda r:(order[r['status']],-abs(r['mean_ic']) if np.isfinite(r['mean_ic']) else 0))
# 6. Export all-factor results and the weekly, annual and correlation diagnostics.
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'full_sample_factor_screening.csv',index=False,encoding='utf-8-sig')
w.to_csv(OUT/'weekly_rank_ic.csv',index=False,encoding='utf-8-sig')
annual.to_csv(OUT/'annual_rank_ic.csv',index=False,encoding='utf-8-sig')
corr.to_csv(OUT/'factor_correlations.csv',encoding='utf-8-sig')
pairs=[{'factor_1':f,'factor_2':g,'correlation':float(corr.loc[f,g])} for i,f in enumerate(tech) for g in tech[i+1:] if abs(corr.loc[f,g])>=.85]
pd.DataFrame(pairs).to_csv(OUT/'high_correlation_pairs.csv',index=False,encoding='utf-8-sig')
config={'selection_type':'FULL_SAMPLE_EXPLORATORY','independent_test':False,'universe':'user-confirmed 66 stocks from test_45_factors_weekly.csv','period':['2016-09-02','2026-08-21'],'primary_factors':picked,'broad_factors':tech,'macro_optional':macro,'macro_join_key':'date','ranking_target':'target','classification_target':'target2','never_as_features':['target','target2','date','ticker'],'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'thresholds':{'coverage':.8,'min_weeks':104,'abs_ic':.01,'abs_icir':.08,'year_consistency':.6,'min_weeks_per_year':40,'correlation':.85},'icir_annualized':False,'excluded_unverified_tail':'2026-08-28','warning':'Feature selection uses returns from the full period; splitting the same history again does not create an independent test.'}
(OUT/'model_features.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')

# User-specified candidate design; no IC sign or threshold determines membership.
five_dimensions={
 'revenue_growth_yoy':'Business growth',
 'cashflow_to_assets':'Operating cash generation',
 'operating_profitability':'Operating profitability',
 'asset_growth_yoy':'Asset expansion',
 'debt_to_assets':'Financial leverage'}
five=list(five_dimensions)
five_rows=summary[summary.factor.isin(fund)].drop(columns=['status','reason']).copy()
five_rows['financial_dimension']=five_rows.factor.map(
 lambda f:five_dimensions.get(f,'Change in profitability'))
five_rows['candidate_membership']=five_rows.factor.map(
 lambda f:'Included by design' if f in five else 'Not included; comparison')
five_rows['design_note']=five_rows.factor.map(
 lambda f:'Represents '+five_dimensions[f] if f in five else
 'The compact candidate uses operating profitability to represent profitability; omission is a design choice, not evidence of weak or redundant signal.')
five_rows.to_csv(OUT/'five_factor_comparison.csv',index=False,encoding='utf-8-sig')
summary[summary.factor.isin(picked)].to_csv(OUT/'seven_factor_results.csv',index=False,encoding='utf-8-sig')
# 7. Export model-ready feature lists. The 45-feature baseline keeps original duplicates.
# The 11-feature set adds the four macro variables to the screened seven.
sets={
 '7_factors':picked,
 '5_factors_candidate':five,
 '11_factors':picked+macro,
 '45_factors':factors,
 'five_factor_method':'Specified financial-dimension candidate; membership is an explicit input, not computed from metrics.',
 'five_factor_dimensions':five_dimensions,
 'limitation':'Negative correlation can predict. The profitability-dimension choice does not establish redundancy or inferiority of gross margin change. Compare five versus six to assess incremental value. This full-sample report does not establish independent validity or optimality.',
 'annualized_returns_calculated':False,
 'independent_validation':False}
(OUT/'factor_sets.json').write_text(json.dumps(sets,ensure_ascii=False,indent=2),encoding='utf8')
payload={'rows':json.loads(summary.to_json(orient='records',force_ascii=False)),
 'five_rows':json.loads(five_rows.to_json(orient='records',force_ascii=False)),
 'sets':sets,'config':config,
 'info':{'rows':len(d),'weeks':d.date.nunique(),'stocks':d.ticker.nunique()}}
(OUT/'collection_report.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(sets,ensure_ascii=False,indent=2))
