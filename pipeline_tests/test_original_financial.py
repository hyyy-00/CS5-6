from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import pipeline

sys.path.insert(0,str(BASE/'data_processing'))
from collect_sec_fundamentals import next_market_day
from ast_helpers import functions
ROOT=BASE
sec=functions(ROOT/'data_processing/collect_sec_fundamentals.py')

def setUpModule():
    (BASE/'runs').mkdir(parents=True, exist_ok=True)

def duration(rows):
    frame=pd.DataFrame([dict(start=a,end=b,val=v,filed=f,fy=2020,source_tag='Revenues',tag_priority=0) for a,b,v,f in rows])
    for c in ['start','end','filed']:frame[c]=pd.to_datetime(frame[c])
    return sec['duration_values'](frame)
def facts(values):
    return {'facts':{'us-gaap':{k:{'units':{'USD':[{'end':'2020-12-31','filed':f,'form':'10-K','val':v}]}} for k,(v,f) in values.items()}}}

class FinancialTests(unittest.TestCase):
    def test_mixed_discrete_and_ytd(self):
        q=duration([('2020-01-01','2020-03-31',100,'2020-05-01'),('2020-04-01','2020-06-30',120,'2020-08-01'),('2020-01-01','2020-09-30',360,'2020-11-01')])
        self.assertEqual(q.quarter.iloc[-1],140)
    def test_all_ytd_and_annual(self):
        q=duration([('2020-01-01','2020-03-31',100,'2020-05-01'),('2020-01-01','2020-06-30',220,'2020-08-01'),('2020-01-01','2020-09-30',360,'2020-11-01'),('2020-01-01','2020-12-31',500,'2021-03-01')])
        np.testing.assert_array_equal(q.quarter,[100,120,140,140]);self.assertEqual(q.ttm.iloc[-1],500)
    def test_discrete_then_annual(self):
        q=duration([('2020-01-01','2020-03-31',100,'2020-05-01'),('2020-04-01','2020-06-30',120,'2020-08-01'),('2020-07-01','2020-09-30',140,'2020-11-01'),('2020-01-01','2020-12-31',500,'2021-03-01')]);self.assertEqual(q.quarter.iloc[-1],140)
    def test_missing_prefix_not_subtracted_as_ytd(self):
        q=duration([('2020-04-01','2020-06-30',120,'2020-08-01'),('2020-01-01','2020-09-30',360,'2020-11-01')]);self.assertTrue(pd.isna(q.quarter.iloc[-1]))
    def test_same_start_ytd_prefix_works_without_q1(self):
        q=duration([('2020-01-01','2020-06-30',220,'2020-08-01'),('2020-01-01','2020-09-30',360,'2020-11-01')]);self.assertEqual(q.quarter.iloc[-1],140)
    def test_availability_includes_late_dependency(self):
        q=duration([('2020-01-01','2020-06-30',220,'2020-12-01'),('2020-01-01','2020-09-30',360,'2020-11-01')]);self.assertEqual(q.quarter_available_at.iloc[-1],pd.Timestamp('2020-12-01'))
    def test_retail_sixteen_week_quarter_is_discrete(self):
        q=duration([('2015-08-31','2015-11-22',700,'2015-12-18'),('2015-11-23','2016-02-14',800,'2016-03-09'),('2016-02-15','2016-05-08',900,'2016-06-01'),('2016-05-09','2016-08-28',1200,'2016-10-12')])
        self.assertEqual(q.quarter.iloc[-1],1200);self.assertEqual(q.ttm.iloc[-1],3600)
    def test_seventeen_week_quarter_is_discrete(self):
        q=duration([('2017-05-08','2017-09-03',1450,'2017-10-18')]);self.assertEqual(q.quarter.iloc[0],1450)
    def test_missing_quarter_cannot_form_ttm(self):
        q=duration([('2020-01-01','2020-03-31',100,'2020-05-01'),('2020-07-01','2020-09-30',120,'2020-11-01'),('2020-10-01','2020-12-31',140,'2021-02-01'),('2021-01-01','2021-03-31',160,'2021-05-01')]);self.assertTrue(pd.isna(q.ttm.iloc[-1]))
    def test_empty_duration(self):self.assertTrue(sec['duration_values'](pd.DataFrame()).empty)
    def test_three_nonoverlapping_debt_components(self):
        q=sec['debt_values'](facts({'LongTermDebtNoncurrent':(90,'2021-01-01'),'LongTermDebtCurrent':(20,'2021-02-01'),'ShortTermBorrowings':(10,'2021-03-01')}));self.assertEqual(q.debt.iloc[0],120);self.assertEqual(q.debt_available_at.iloc[0],pd.Timestamp('2021-03-01'))
    def test_short_total_not_double_counted_with_current(self):
        q=sec['debt_values'](facts({'LongTermDebtNoncurrent':(90,'2021-01-01'),'LongTermDebtCurrent':(20,'2021-01-01'),'ShortTermDebt':(30,'2021-01-01'),'ShortTermBorrowings':(10,'2021-01-01')}));self.assertEqual(q.debt.iloc[0],120)
    def test_current_portion_alias(self):
        q=sec['debt_values'](facts({'LongTermDebtNoncurrent':(90,'2021-01-01'),'CurrentPortionOfLongTermDebt':(20,'2021-01-01')}));self.assertEqual(q.debt.iloc[0],110)
    def test_reported_total_preferred(self):
        q=sec['debt_values'](facts({'LongTermDebtAndShortTermBorrowings':(200,'2021-01-01'),'LongTermDebtNoncurrent':(90,'2021-01-01'),'LongTermDebtCurrent':(20,'2021-01-01'),'ShortTermBorrowings':(10,'2021-01-01')}));self.assertEqual(q.debt.iloc[0],200)
    def test_debt_value_and_date_select_same_source(self):
        q=sec['debt_values'](facts({'LongTermDebtNoncurrent':(90,'2021-01-01'),'LongTermDebt':(120,'2022-01-01')}));self.assertEqual(q.debt.iloc[0],120);self.assertEqual(q.debt_available_at.iloc[0],pd.Timestamp('2022-01-01'))
    def test_empty_debt(self):self.assertTrue(sec['debt_values']({'facts':{}}).empty)

def selected_functions(relative):
    import ast
    p=ROOT/relative;tree=ast.parse(p.read_text('utf-8-sig'));keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign,ast.AnnAssign))]
    ns={'__file__':str(p),'np':np,'pd':pd,'Path':Path,'FUNDAMENTAL_DEFINITIONS':sec['FUNDAMENTAL_DEFINITIONS'],'next_market_day':next_market_day}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),ns);return ns

class LabelTests(unittest.TestCase):
    def technical(self,dates):
        ns=selected_functions('data_processing/build_technical_factors.py');return ns['add_weekly_open_to_close_target'](pd.DataFrame({'ticker':['A']*len(dates),'date':pd.to_datetime(dates),'adj_close':np.arange(len(dates))*10+100.}))
    def test_normal_next_week_label(self):self.assertAlmostEqual(self.technical(['2020-01-03','2020-01-10']).target.iloc[0],np.log(1.1))
    def test_truncated_final_monday_is_not_a_weekly_label(self):self.assertTrue(pd.isna(self.technical(['2026-08-28','2026-08-31']).target.iloc[0]))
    def test_good_friday_thursday_is_complete(self):self.assertAlmostEqual(self.technical(['2020-04-03','2020-04-09']).target.iloc[0],np.log(1.1))
    def test_missing_week_label_rejected(self):self.assertTrue(pd.isna(self.technical(['2020-01-03','2020-01-17']).target.iloc[0]))
    def test_holiday_date_gap_still_next_calendar_week(self):self.assertAlmostEqual(self.technical(['2020-07-02','2020-07-10']).target.iloc[0],np.log(1.1))
    def test_additional_collector_rejects_missing_week(self):
        ns=selected_functions('data_processing/collect_additional_40_staples.py');out=ns['add_targets'](pd.DataFrame({'ticker':['A','A'],'date':pd.to_datetime(['2020-01-03','2020-01-17']),'adj_close':[100.,121.]}));self.assertTrue(pd.isna(out.target.iloc[0]))
    def test_additional_collector_rejects_truncated_week(self):
        ns=selected_functions('data_processing/collect_additional_40_staples.py');out=ns['add_targets'](pd.DataFrame({'ticker':['A','A'],'date':pd.to_datetime(['2026-08-28','2026-08-31']),'adj_close':[100.,121.]}));self.assertTrue(pd.isna(out.target.iloc[0]))
    def test_exporter_does_not_attach_prices_two_weeks_later(self):
        ns=selected_functions('data_processing/export_additional_40_excel.py')
        with tempfile.TemporaryDirectory(dir=BASE/'runs') as tmp:
            p=Path(tmp)/'daily.csv';pd.DataFrame({'date':['2020-01-03','2020-01-17'],'ticker':['A','A'],'open':[100.,120.],'close':[100.,121.]}).to_csv(p,index=False)
            weekly=pd.DataFrame({'ticker':['A'],'date':pd.to_datetime(['2020-01-03'])})
            with self.assertRaises(ValueError):ns['add_next_week_prices'](weekly,p)
    def test_exporter_rejects_incomplete_next_week(self):
        ns=selected_functions('data_processing/export_additional_40_excel.py')
        with tempfile.TemporaryDirectory(dir=BASE/'runs') as tmp:
            p=Path(tmp)/'daily.csv';pd.DataFrame({'date':['2026-08-28','2026-08-31'],'ticker':['A','A'],'open':[100.,120.],'close':[100.,121.]}).to_csv(p,index=False)
            weekly=pd.DataFrame({'ticker':['A'],'date':pd.to_datetime(['2026-08-28'])})
            with self.assertRaises(ValueError):ns['add_next_week_prices'](weekly,p)
    def test_exporter_accepts_holiday_thursday_endpoint(self):
        ns=selected_functions('data_processing/export_additional_40_excel.py')
        with tempfile.TemporaryDirectory(dir=BASE/'runs') as tmp:
            p=Path(tmp)/'daily.csv';pd.DataFrame({'date':['2020-04-03','2020-04-06','2020-04-09'],'ticker':['A']*3,'open':[100.,110.,120.],'close':[100.,115.,121.]}).to_csv(p,index=False)
            weekly=pd.DataFrame({'ticker':['A'],'date':pd.to_datetime(['2020-04-03'])})
            out=ns['add_next_week_prices'](weekly,p);self.assertEqual(out.buy_price.iloc[0],110.);self.assertEqual(out.sell_price.iloc[0],121.)

if __name__=='__main__':unittest.main(verbosity=2)
