"""Run with unittest; FCL_TEST_SOURCE and the two FCL_*_FIXTURE vars are optional."""
import ast
import calendar
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


SOURCE = Path(os.environ.get('FCL_TEST_SOURCE', Path(__file__).resolve().parents[1] / 'fcl_planner.py'))


def load_core(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), str(path))
    first = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import) and any(a.name == 'streamlit' for a in node.names):
                continue
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.Assign) and node.lineno < first:
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns, tree


def independent_daily(ns, row, args):
    transit, earliest, target, today, cutoff = args
    alloc = ns['waterpool_allocation_v2'](row, transit, earliest, target, today, sales_cutoff=cutoff)
    arrivals = ns['compute_arrivals'](transit, earliest, target)
    events = {}
    stock = sum(row[f'{r}_在仓'] for r in ns['REGIONS'])
    for r in ns['REGIONS']:
        for day, qty in ns['parse_in_transit'](row[f'{r}_多批次在途']).items():
            events[day] = events.get(day, 0) + qty
        events[arrivals[r]] = events.get(arrivals[r], 0) + alloc[r]
    daily = ns['build_daily_sales_fn'](row, today, sales_cutoff=cutoff)
    trace = []
    for offset in range(1, (cutoff - today).days + 1):
        day = today + dt.timedelta(days=offset)
        inbound = events.get(day, 0)
        stock += inbound
        demand = daily(day)
        sold = min(stock, demand)
        stock -= sold
        trace.append(dict(date=str(day), inbound=inbound, demand=demand,
                          sold=sold, shortage=demand - sold, closing=stock))
    return trace


def run_button(ns, tree, frame, args, contract_file, loader=None):
    transit, earliest, target, today, cutoff = args
    errors, state = [], {}
    stub = SimpleNamespace(
        session_state=state, error=errors.append, warning=errors.append,
        dataframe=Mock(), stop=Mock(side_effect=AssertionError('unexpected stop')),
        progress=Mock(return_value=SimpleNamespace(progress=Mock(), empty=Mock())),
        rerun=Mock(),
    )
    scope = dict(ns, st=stub, btn_recommend=True, edited_df=frame,
                 today=today, sales_cutoff=cutoff, transit_times=transit,
                 earliest_etd=earliest, target_eta=target, d_diff_invalid=False,
                 contract_file=contract_file, contract_ocean_cycle_days=34,
                 contract_sales_cycle_days=30)
    if loader:
        scope['load_contract_workbook'] = loader
    node = next(n for n in tree.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'btn_recommend')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    if errors:
        raise AssertionError(errors)
    return state


class DailyGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.tree = load_core(SOURCE)
        cls.today = dt.date(2026, 11, 17)
        cls.earliest = dt.date(2026, 11, 24)
        cls.target = dt.date(2027, 2, 1)
        cls.cutoff = dt.date(2027, 2, 28)
        cls.transit = dict(zip(cls.ns['REGIONS'], [1, 45, 45, 45, 50]))
        cls.args = (cls.transit, cls.earliest, cls.target, cls.today, cls.cutoff)

    def row(self, qty=0, late=300):
        row = dict(SKU='TEST', SPU='SPU', 店铺='Shop', 组别='Group', 运营='Operator')
        row['本次总发货量'] = float(qty)
        for r in self.ns['REGIONS']:
            row[self.ns['ratio_col_name'](r)] = 25.0 if r != 'CG' else 0.0
            row[f'{r}_在仓'] = 0.0
            row[f'{r}_多批次在途'] = ''
        for col in self.ns['FORECAST_COLS']:
            row[col] = 0.0
        row[self.ns['FORECAST_COLS'][3]] = 140.0
        row['美西_多批次在途'] = f'2027-02-14:{late}' if late else ''
        return row

    def status(self, row, args=None):
        return self.ns['compute_row_status'](row, *(args or self.args))

    def recommend(self, row, args=None):
        return self.ns['recommend_q_ship'](row, *(args or self.args), include_stock_balance=True)

    def test_gap_and_balance_coexist(self):
        self.assertEqual(self.recommend(self.row()), (65, 65, 'ok', 225))

    def test_daily_ledger_and_same_sd_partial_shortage(self):
        for q, lost, balance in ((0, 65, 225), (64, 1, 225), (65, 0, 225), (66, 0, 226)):
            with self.subTest(q=q):
                trace = independent_daily(self.ns, self.row(q), self.args)
                feb = [r for r in trace if r['date'] >= '2027-02-01']
                self.assertAlmostEqual(sum(r['shortage'] for r in feb), lost)
                self.assertAlmostEqual(feb[-1]['closing'], balance)
                self.assertAlmostEqual(self.status(self.row(q))['RQ'], balance)
        a, b = self.status(self.row(64)), self.status(self.row(65))
        self.assertEqual(a['SD'], b['SD'])
        self.assertFalse(self.ns['shortage_not_worse'](a, b))

    def test_stage2_keeps_gap_fill_and_removes_only_extra(self):
        records, after = self.ns['stage2_independent_reduction'](
            pd.DataFrame([self.row(65), self.row(66), self.row(100)]), *self.args)
        self.assertEqual(after['本次总发货量'].tolist(), [65, 65, 65])
        self.assertEqual(len(records), 2)
        self.assertEqual(self.ns['stage2_independent_reduction'](after, *self.args)[0], [])

    def test_main_board_agrees_with_stage2(self):
        board = self.ns['compute_main_board'](pd.DataFrame([self.row(65), self.row(66)]), *self.args)
        self.assertEqual(board['建议减量至'].tolist(), ['-', '65'])

    def test_late_supply_is_added_once(self):
        row = self.row()
        row['美东_多批次在途'] = '2027-03-02:40'
        self.assertEqual(self.recommend(row), (65, 65, 'ok', 265))

    def test_cutoff_day_arrival_is_not_late(self):
        row = self.row(late=0)
        row['美西_多批次在途'] = '2027-02-28:300'
        self.assertEqual(self.recommend(row), (135, 135, 'ok', 295))

    def test_zero_forecast(self):
        row = self.row()
        for col in self.ns['FORECAST_COLS']:
            row[col] = 0.0
        self.assertEqual(self.recommend(row), (0, 0, 'redundant', 300))
        row['美西_多批次在途'] = ''
        self.assertEqual(self.recommend(row), (0, 0, 'ok', 0))

    def test_full_coverage(self):
        row = self.row()
        row['美西_在仓'] = 1000.0
        result = self.recommend(row)
        self.assertEqual(result[:3], (0, 0, 'redundant'))
        self.assertIsInstance(result[3], int)

    def test_small_quantity_rule_preserved(self):
        row = self.row()
        row['美西_多批次在途'] = '2027-02-02:300'
        self.assertEqual(self.recommend(row), (0, 5, 'small', 165))

    def test_no_supply_recommends_full_reachable_month(self):
        self.assertEqual(self.recommend(self.row(late=0)), (140, 140, 'ok', 0))

    def test_no_order_for_unreachable_demand(self):
        args = (self.transit, self.earliest, dt.date(2027, 3, 1), self.today, self.cutoff)
        self.assertEqual(self.recommend(self.row(late=0), args), (0, 0, 'ok', 0))

    def test_multiple_gaps(self):
        row = self.row(late=0)
        row['美西_多批次在途'] = '2027-02-06:25;2027-02-20:300'
        self.assertEqual(self.recommend(row), (70, 70, 'ok', 255))

    def test_precise_daily_spikes(self):
        row = self.row()
        for day in range(1, 29):
            row[self.ns['daily_forecast_col_name'](4, day)] = 65.0 if day == 1 else 75.0 if day == 14 else 0.0
        self.assertEqual(self.ns['validate_daily_forecast_input'](pd.DataFrame([row]), self.today), [])
        self.assertEqual(self.recommend(row), (65, 65, 'ok', 225))

    def test_fractional_demand_rounds_up_without_new_gap(self):
        row = self.row(late=0)
        row[self.ns['FORECAST_COLS'][3]] = 140.2
        result = self.recommend(row)
        self.assertEqual(result, (141, 141, 'ok', 1))
        row['本次总发货量'] = result[0]
        self.assertEqual(self.ns['stage2_independent_reduction'](pd.DataFrame([row]), *self.args)[0], [])

    def test_input_untouched_and_old_three_value_api(self):
        row = self.row(999)
        original = copy.deepcopy(row)
        self.assertEqual(self.ns['recommend_q_ship'](row, *self.args), (65, 65, 'ok'))
        self.assertEqual(row, original)

    def test_stage1_cannot_donate_gap_filling_shipment(self):
        frame = pd.DataFrame([self.row(65), self.row(late=0)])
        statuses = [self.status(r.to_dict()) for _, r in frame.iterrows()]
        qty = self.ns['binary_search_max_transfer'](
            frame, 0, 1, '本次发货量', None, None, 65, *self.args,
            False, (self.cutoff-self.today).days, *statuses)
        self.assertLess(qty, 1)
        self.assertEqual(frame['本次总发货量'].tolist(), [65, 0])

    def test_stage3_guards_same_sd_and_rollback(self):
        frame = pd.DataFrame([self.row(), self.row()])
        frame.at[0, '美西_在仓'] = 10.0
        old = dict(SD=28, RQ=225, CZ=10, daily_shortage={}, shortage_qty=0)
        bad = dict(old, CZ=5, daily_shortage={10: 1}, shortage_qty=1)
        good = dict(old, CZ=0)
        before = frame.copy(deep=True)
        with patch.dict(self.ns, compute_stage3_row_status=Mock(side_effect=[bad, good])):
            _, valid = self.ns['evaluate_single_transfer'](
                frame, 0, 1, '在仓', '美西', None, 1, *self.args, False, {0:old, 1:old})
        self.assertFalse(valid)
        pd.testing.assert_frame_equal(frame, before)
        frame.at[1, '美东_在仓'] = 10.0
        before = frame.copy(deep=True)
        with patch.dict(self.ns, compute_stage3_row_status=Mock(side_effect=[good, bad])):
            _, valid = self.ns['evaluate_swap_transfer'](
                frame, 0, 1, '在仓', '美西', None, 1,
                '在仓', '美东', None, 1, *self.args, False, {0:old, 1:old})
        self.assertFalse(valid)
        pd.testing.assert_frame_equal(frame, before)

    def test_button_recalculates_actual_balance_if_contract_short(self):
        row = self.row(late=0)
        row['美西_多批次在途'] = '2027-02-28:300'
        # With no contracts, actual q=0: only the arrival day's 5 units sell, leaving 295.
        row['美西_在仓'] = 8.0
        state = run_button(self.ns, self.tree, pd.DataFrame([row]), self.args, object(),
                           loader=lambda _: ([], {'SPU':55}, []))
        output = state['recommended_df'].iloc[0]
        self.assertEqual(output['本次总发货量'], 0)
        expected = self.ns['stock_balance_at_cutoff'](row, self.status(row), self.cutoff)
        self.assertEqual(output[self.ns['STOCK_BALANCE_COL']], expected)
        self.assertGreater(output['本轮理论发货量'], 0)
        self.assertIn('合同数量不足', output['备注'])

    def test_fractional_stage2_never_increases_quantity(self):
        for q in (64.4, 65.4, 65.6, 100.6):
            with self.subTest(q=q):
                row = self.row(q)
                new_q = self.ns['service_preserving_ship_quantity'](row, *self.args)
                self.assertLessEqual(new_q, q)
                self.assertTrue(self.ns['shortage_not_worse'](self.status(self.row(new_q)), self.status(row)))

    def test_seeded_arrival_and_daily_forecast_scenarios(self):
        rng = random.Random(20261117)
        baseline_path = os.environ.get('FCL_BASELINE_SOURCE')
        baseline = load_core(Path(baseline_path))[0] if baseline_path else None
        today = dt.date(2027, 2, 1)
        for case in range(80):
            with self.subTest(case=case):
                row = self.row()
                for r in self.ns['REGIONS']:
                    row[f'{r}_在仓'] = float(rng.randrange(0, 15))
                    date = today + dt.timedelta(days=rng.randrange(2, 50))
                    row[f'{r}_多批次在途'] = f'{date}:{rng.randrange(0, 40)}'
                for i, col in enumerate(self.ns['FORECAST_COLS'], 1):
                    row[col] = float(rng.randrange(25, 200)) if i <= 2 else 0.0
                    if case % 2 == 0 and i <= 2:
                        cols = self.ns['daily_forecast_cols_for_month'](today, i)
                        values = [rng.randrange(0, 12) / 2 for _ in cols]
                        row.update(dict(zip(cols, values)))
                        row[col] = sum(values)
                transit = {r:rng.randrange(1, 40) for r in self.ns['REGIONS']}
                args = (transit, today+dt.timedelta(days=1), today+dt.timedelta(days=5),
                        today, today+dt.timedelta(days=45))
                for q in (0, 65):
                    row['本次总发货量'] = q
                    status = self.status(row, args)
                    if baseline:
                        old = baseline['compute_row_status'](row, *args)
                        for key in old:
                            self.assertEqual(status[key], old[key])
                    trace = independent_daily(self.ns, row, args)
                    self.assertAlmostEqual(status['RQ'], trace[-1]['closing'], delta=0.002)
                    self.assertAlmostEqual(status['shortage_qty'], sum(r['shortage'] for r in trace), delta=0.002)
                final, raw, flag, _ = self.recommend(row, args)
                daily = self.ns['build_daily_sales_fn'](row, today, sales_cutoff=args[-1])
                upper = int(sum(daily(today+dt.timedelta(days=i)) for i in range(1,46))) + 10
                row['本次总发货量'] = upper
                reference = self.status(row, args)
                row['本次总发货量'] = raw
                self.assertTrue(self.ns['shortage_not_worse'](self.status(row, args), reference))
                row['本次总发货量'] = max(100, final)
                before = self.status(row, args)
                reduced = self.ns['service_preserving_ship_quantity'](row, *args)
                self.assertLessEqual(reduced, row['本次总发货量'])
                row['本次总发货量'] = reduced
                self.assertTrue(self.ns['shortage_not_worse'](self.status(row, args), before))

    def test_main_editor_shows_both_quantities_as_integers(self):
        from streamlit.testing.v1 import AppTest
        frame = pd.DataFrame([self.row(65)])
        frame['本轮理论发货量'] = 65
        col = self.ns['STOCK_BALANCE_COL']
        frame[col] = pd.Series([225], dtype='Int64')
        app = AppTest.from_file(str(SOURCE), default_timeout=30)
        app.session_state['recommended_df'] = frame
        app.run()
        self.assertFalse(app.exception)
        editor = app.dataframe[0]
        self.assertEqual(editor.value['本次总发货量'].tolist(), [65])
        self.assertEqual(editor.value['本轮理论发货量'].tolist(), [65])
        self.assertEqual(editor.value[col].tolist(), [225])
        config = json.loads(editor.proto.columns)
        self.assertEqual(config[col]['type_config']['format'], '%d')
        self.assertTrue(config[col]['disabled'])


@unittest.skipUnless(os.environ.get('FCL_FORECAST_FIXTURE') and os.environ.get('FCL_CONTRACT_FIXTURE'),
                     'Set FCL_FORECAST_FIXTURE and FCL_CONTRACT_FIXTURE for private workbook integration')
class WorkbookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DailyGapTests.setUpClass.__func__(cls)

    def status(self, row):
        return self.ns['compute_row_status'](row, *self.args)

    def test_uploaded_workbooks_button_pipeline_and_sandbox(self):
        frame = pd.read_excel(os.environ['FCL_FORECAST_FIXTURE'], sheet_name=0)
        contracts, leads, errors = self.ns['load_contract_workbook'](os.environ['FCL_CONTRACT_FIXTURE'])
        self.assertEqual(errors, [])
        self.assertEqual(leads['NBSS004'], 55)
        self.assertEqual(self.ns['validate_daily_forecast_input'](frame, self.today), [])
        state = run_button(self.ns, self.tree, frame, self.args, os.environ['FCL_CONTRACT_FIXTURE'])
        output = state['recommended_df']
        self.assertEqual(output['本轮理论发货量'].tolist(), [65, 65])
        self.assertEqual(output['本次总发货量'].tolist(), [65, 65])
        self.assertEqual(output[self.ns['STOCK_BALANCE_COL']].tolist(), [225, 225])
        self.assertEqual(str(output[self.ns['STOCK_BALANCE_COL']].dtype), 'Int64')
        self.assertEqual(output['本轮可用合同量'].tolist(), [362, 362])
        self.assertEqual(output['剩余合同数量'].tolist(), [297, 297])
        self.assertEqual(state['recommend_msgs']['redundant_count'], 0)
        regenerated = run_button(self.ns, self.tree, output, self.args, os.environ['FCL_CONTRACT_FIXTURE'])
        pd.testing.assert_frame_equal(regenerated['recommended_df'], output)
        s1, s2, s3, s4, after = self.ns['run_full_pipeline'](output, *self.args)
        self.assertEqual(s1, [])
        self.assertEqual(s2, [])
        self.assertEqual(s3, [])
        self.assertEqual(after['本次总发货量'].tolist(), [65,65])
        self.assertEqual([r['死冗余量'] for r in s4], [225,225])
        board = self.ns['compute_main_board'](after, *self.args)
        self.assertEqual(board['建议减量至'].tolist(), ['-', '-'])
        for _, row in after.iterrows():
            row = row.to_dict()
            status = self.status(row)
            sandbox = self.ns['compute_sandbox_metrics'](row, status['alloc'], *self.args, self.cutoff)
            self.assertAlmostEqual(sandbox['total'], 225)
            ledger = independent_daily(self.ns, row, self.args)
            self.assertAlmostEqual(sum(r['shortage'] for r in ledger if r['date']>='2027-02-01'), 0)
            contract = self.ns['compute_contract_recommendation'](
                row, contracts, leads, 65, self.today, self.earliest, self.target, 34, 30)
            if os.environ.get('FCL_BASELINE_SOURCE'):
                baseline, _ = load_core(Path(os.environ['FCL_BASELINE_SOURCE']))
                old_contract = baseline['compute_contract_recommendation'](
                    row, contracts, leads, 65, self.today, self.earliest, self.target, 34, 30)
                for key in ('本轮实际发货量', '本轮可用合同量', '剩余合同数量', '合同扣减明细',
                            '剩余合同发货计划', '_含剩余合同预估耗尽日'):
                    self.assertEqual(contract[key], old_contract[key])
                print('CONTRACT_BEFORE_AFTER', row['SKU'],
                      [old_contract['推荐下单量'], old_contract['最晚下单日']],
                      [contract['推荐下单量'], contract['最晚下单日']])
            initial_stock, events = self.ns['get_global_supply_events'](row)
            events.append((self.target, 65))
            for plan in contract['剩余合同发货计划']:
                if plan['本次计划发货数量'] > 0:
                    events.append((dt.date.fromisoformat(plan['预计到美日']), plan['本次计划发货数量']))
            arrival = self.today + dt.timedelta(days=leads[row['SPU']] + 34)
            end = arrival + dt.timedelta(days=30)
            if contract['_含剩余合同预估耗尽日'] and contract['_含剩余合同预估耗尽日'] < arrival:
                end -= dt.timedelta(days=1)
            daily = self.ns['build_contract_daily_sales_fn'](row, self.today)
            stock, reachable_gap, first_gap = initial_stock, 0.0, None
            for offset in range(1,(end-self.today).days+1):
                day = self.today+dt.timedelta(days=offset)
                stock += sum(q for date,q in events if date == day)
                shortage = max(0,daily(day)-stock)
                stock = max(0,stock-daily(day))
                if day >= arrival and shortage > 1e-9:
                    reachable_gap += shortage
                    first_gap = first_gap or day
            expected_qty = max(0,int(math.ceil(reachable_gap-1e-9)))
            if 0 < expected_qty < 10:
                expected_qty = 0
            self.assertEqual(contract['推荐下单量'], expected_qty)
            self.assertEqual(contract['最晚下单日'],
                             str(first_gap-dt.timedelta(days=leads[row['SPU']]+34)) if expected_qty else '-')
        print('WORKBOOK_RESULT', output[['SKU', '本轮理论发货量', '本次总发货量',
                                       self.ns['STOCK_BALANCE_COL'], '本轮可用合同量',
                                       '剩余合同数量', '推荐下单量', '最晚下单日']].to_json(orient='records', force_ascii=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
