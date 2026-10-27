from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.contracts import ContractError, output_manifest, validate_outputs, write_json
from pipeline.inputs import execution_dates, prepare_training, preflight, read_csv
from pipeline.learning import specifications, training_slice, ranked
from pipeline.trading import implementation, independent_case
from pipeline.runner import validate_stage_artifacts, REQUIRED_OUTPUTS
import pipeline.runner as runner
import pandas as pd
import numpy as np

def setUpModule():
    (ROOT/'runs').mkdir(parents=True, exist_ok=True)


class InputContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_csv(ROOT / 'data/reproducible/train66_corrected_complete.csv')
    def prepare(self, train=None):
        return prepare_training(self.train.copy() if train is None else train)
    def test_521_complete_weeks_are_accepted(self):
        train = self.prepare()
        self.assertEqual(train.week_index.nunique(), 521)
        self.assertEqual(train.label_end_date.max(), pd.Timestamp('2026-08-28'))
    def test_missing_entire_week_rejected(self):
        d = sorted(self.train.date.unique())[300]
        with self.assertRaisesRegex(ContractError, 'skips'):
            self.prepare(self.train[self.train.date.ne(d)])
    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ContractError, 'duplicate'):
            self.prepare(pd.concat([self.train, self.train.iloc[:1]]))
    def test_nonfinite_factor_rejected(self):
        frame = self.train.copy();frame.loc[0, 'debt_to_assets'] = np.inf
        with self.assertRaisesRegex(ContractError, 'infinite'):
            self.prepare(frame)
    def test_wrong_label_rejected(self):
        frame = self.train.copy();frame.loc[0, 'target'] += .2
        with self.assertRaisesRegex(ContractError, 'target'):
            self.prepare(frame)
    def test_good_friday_endpoints(self):
        d = execution_dates(['2020-04-03']).iloc[0]
        self.assertEqual(d.expected_buy_date, pd.Timestamp('2020-04-06'))
        self.assertEqual(d.expected_sell_date, pd.Timestamp('2020-04-09'))
    def test_training_cannot_use_future_label(self):
        train = self.prepare()
        train = ranked(train)
        refit = 312;date = train[train.week_index.eq(refit)].date.iloc[0]
        train.loc[train.week_index.eq(refit-1), 'label_end_date'] = date + pd.Timedelta(days=7)
        with self.assertRaisesRegex(ContractError, 'future training label'):
            training_slice(train, specifications()[0], refit, date)


class TrainingScope(unittest.TestCase):
    def test_only_ensemble_components_are_fitted(self):
        specs = specifications()
        self.assertEqual([x['name'] for x in specs], ['Model1_v1', 'Model2_v1', 'Model3_v1'])
        self.assertEqual([x['window'] for x in specs], [156, 52, 52])
    def test_training_input_closure_has_no_external_or_raw_data(self):
        files, missing = preflight()
        self.assertFalse(missing)
        self.assertEqual({p.name for p in files}, {'train66_corrected_complete.csv', 'train_universe.csv'})
    def test_removed_workflows_are_rejected(self):
        import io
        from contextlib import redirect_stderr
        for args in [['--resume', 'unused'], ['--stage', 'backtest'], ['--mode', 'raw'], ['--grid', 'standard']]:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                runner.main(args)
            self.assertEqual(caught.exception.code, 2)


class OutputContracts(unittest.TestCase):
    def test_empty_success_cannot_satisfy_any_stage(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            for stage in REQUIRED_OUTPUTS:
                with self.subTest(stage=stage), self.assertRaisesRegex(ContractError,'missing/empty'):
                    validate_stage_artifacts(stage,Path(d))
    def test_empty_model_manifest_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            p=Path(d);write_json(p/'model_manifest.json',[])
            with self.assertRaisesRegex(ContractError,'no models'):validate_stage_artifacts('train',p)
    def test_missing_or_changed_artifact_invalidates_stage(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            run = Path(d);(run/'train').mkdir();file=run/'train/model.json';file.write_text('real model fixture')
            record={'status':'passed','outputs':output_manifest(run/'train')}
            validate_outputs(run,'train',record)
            file.write_text('changed')
            with self.assertRaises(ContractError):validate_outputs(run,'train',record)
            file.unlink()
            with self.assertRaises(ContractError):validate_outputs(run,'train',record)
    def test_failed_stage_cannot_unlock_downstream(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            with self.assertRaisesRegex(ContractError,'not passed'):
                validate_outputs(Path(d),'verify',{'status':'failed','outputs':{}})
    def test_unexpected_artifact_invalidates_stage(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            run=Path(d);(run/'train').mkdir();record={'status':'passed','outputs':{}}
            (run/'train/old_result.csv').write_text('stale')
            with self.assertRaises(ContractError):validate_outputs(run,'train',record)


class IndependentAccounting(unittest.TestCase):
    def test_original_file_prepare_interface_remains_compatible(self):
        s,w,o,h,prices,pred=self.fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
            directory=Path(d);(directory/'inputs').mkdir();(directory/'prepared').mkdir()
            prices.to_csv(directory/'inputs/test40_with_prices.csv',index=False)
            pred.to_csv(directory/'prepared/predictions.csv',index=False)
            engine=implementation();engine.HERE=directory
            file_result=engine.prepare()
            explicit_result=engine.prepare(raw_frame=prices,prediction_frame=pred)
            pd.testing.assert_frame_equal(file_result[0],explicit_result[0])
            pd.testing.assert_frame_equal(file_result[1],explicit_result[1])
            self.assertEqual(file_result[3],explicit_result[3])
            self.assertEqual(file_result[4]['open'],explicit_result[4]['open'])
    def fixture(self):
        prices=pd.DataFrame([dict(date=pd.Timestamp(d),ticker=f'S{i}',buy_price=10.+i,sell_price=(10.+i)*ratio)
                             for d,ratio in [('2020-01-03',.9),('2020-01-10',1.1)] for i in range(6)])
        pred=prices[['date','ticker']].copy();pred['prediction']=np.tile([6,5,4,3,2,1],2).astype(float)
        engine=implementation();_,_,groups,calendar,context=engine.prepare(raw_frame=prices,prediction_frame=pred)
        w,o,h=engine.execute(calendar,groups,context,'Top5',1000.,10,detail=True)
        summary=pd.Series(dict(capital=1000.,cost_bps=10,strategy='Top5',**engine.performance(w,1000.)))
        return summary,w,o,h,prices,pred
    def test_independent_cash_and_initial_drawdown(self):
        args=self.fixture();result=independent_case(*args)
        self.assertGreater(result['checks'],50)
        self.assertLess(args[0].max_drawdown,-.09)
    def test_modified_cash_detected(self):
        s,w,o,h,p,pred=self.fixture();w.loc[0,'end_capital']+=1
        with self.assertRaisesRegex(ContractError,'end NAV'):independent_case(s,w,o,h,p,pred)
    def test_modified_fee_detected(self):
        s,w,o,h,p,pred=self.fixture();o.loc[0,'fee']+=1
        with self.assertRaisesRegex(ContractError,'trade fee'):independent_case(s,w,o,h,p,pred)
    def test_wrong_predicted_stock_selection_detected(self):
        s,w,o,h,p,pred=self.fixture();pred['prediction']=-pred.prediction
        with self.assertRaisesRegex(ContractError,'outside the predicted'):independent_case(s,w,o,h,p,pred)
    def test_fractional_shares_detected(self):
        s,w,o,h,p,pred=self.fixture();o['shares']=o.shares.astype(float);o.loc[0,'shares']+=.5
        with self.assertRaisesRegex(ContractError,'Noninteger'):independent_case(s,w,o,h,p,pred)

    def test_noloss13w_filter_is_independently_verified(self):
        dates = pd.date_range('2020-01-03', periods=16, freq='7D')
        prices = pd.DataFrame([
            dict(date=date, ticker=f'S{i}', buy_price=10. + i,
                 sell_price=(10. + i) * ((.98 ** week) if i == 0 else (1.002 ** week)))
            for week, date in enumerate(dates) for i in range(6)
        ])
        predictions = prices[['date', 'ticker']].copy()
        predictions['prediction'] = np.tile([6, 5, 4, 3, 2, 1], len(dates)).astype(float)
        engine = implementation()
        _, _, groups, calendar, context = engine.prepare(raw_frame=prices, prediction_frame=predictions)
        weekly, orders, holdings = engine.execute(
            calendar, groups, context, 'Top5_NoLoss13w', 1000., 10, detail=True)
        summary = pd.Series(dict(
            capital=1000., cost_bps=10, strategy='Top5_NoLoss13w',
            **engine.performance(weekly, 1000.)))
        independent_case(summary, weekly, orders, holdings, prices, predictions)

        wrong_weekly, wrong_orders, wrong_holdings = engine.execute(
            calendar, groups, context, 'Top5', 1000., 10, detail=True)
        wrong_summary = pd.Series(dict(
            capital=1000., cost_bps=10, strategy='Top5_NoLoss13w',
            **engine.performance(wrong_weekly, 1000.)))
        with self.assertRaisesRegex(ContractError, '13-week loss filter'):
            independent_case(
                wrong_summary, wrong_weekly, wrong_orders, wrong_holdings,
                prices, predictions)


if __name__ == '__main__':
    unittest.main(verbosity=2)
