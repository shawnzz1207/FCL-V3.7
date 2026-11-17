"""Contract order timing regression tests, independent of private workbooks."""
import datetime as dt
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from test_daily_gap import SOURCE, load_core


class ContractGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, _ = load_core(SOURCE)
        cls.today = dt.date(2026, 11, 17)
        cls.arrival = dt.date(2027, 2, 14)
        cls.cutoff = dt.date(2027, 3, 16)

    def row(self):
        row = dict(SKU='TEST', SPU='SPU', 店铺='Shop', 组别='Group', 运营='Operator')
        row['本次总发货量'] = 0.0
        for region in self.ns['REGIONS']:
            row[f'{region}_在仓'] = 0.0
            row[f'{region}_多批次在途'] = ''
            row[self.ns['ratio_col_name'](region)] = 100.0 if region == '美西' else 0.0
        for i, col in enumerate(self.ns['FORECAST_COLS']):
            row[col] = 140.0 if i == 3 else 0.0
        row['美西_多批次在途'] = '2027-03-01:300'
        return row

    def recommend(self, row, lead=55, contracts=None, theoretical=0):
        return self.ns['compute_contract_recommendation'](
            row, contracts or [], {'SPU': lead}, theoretical, self.today,
            dt.date(2026, 11, 24), dt.date(2027, 2, 1), 34, 30)

    def contract(self, qty, delivery):
        row = self.row()
        return {**{key: row[key] for key in ('SKU', '店铺', '组别', '运营')},
                'PO号': 'PO-1', '合同可用数量': qty, '交货日期': delivery}

    def gap(self, row, stock=0, events=None, start=None, cutoff=None, daily=None):
        daily = daily or self.ns['build_contract_daily_sales_fn'](row, self.today)
        return self.ns['compute_contract_order_gap'](
            row, stock, events or [], self.today, start or self.arrival,
            cutoff or self.cutoff, daily)

    def test_explicit_cutoff_does_not_include_later_arrival(self):
        today = dt.date(2027, 1, 1)
        result = self.ns['simulate_global_stock'](
            {}, 100, [(dt.date(2027, 1, 21), 100)], today,
            until=dt.date(2027, 1, 6), daily_sales_fn=lambda _: 1.0)
        self.assertEqual(result['stock_at_end'], 95)
        self.assertIsNone(result['first_need_date'])

    def test_explicit_cutoff_arrival_is_included_once(self):
        today = dt.date(2027, 1, 1)
        events = [(dt.date(2027, 1, 6), 100), (dt.date(2027, 1, 7), 500)]
        result = self.ns['simulate_global_stock'](
            {}, 100, events, today, until=dt.date(2027, 1, 6), daily_sales_fn=lambda _: 1.0)
        self.assertEqual(result['stock_at_end'], 195)

    def test_explicit_today_does_not_consume_future_days(self):
        result = self.ns['simulate_global_stock'](
            {}, 100, [(self.arrival, 300)], self.today, until=self.today,
            daily_sales_fn=lambda _: 1.0)
        self.assertEqual(result['stock_at_end'], 100)

    def test_unbounded_exhaustion_preserves_previous_behavior(self):
        today = dt.date(2027, 1, 1)
        events = [(dt.date(2027, 1, 21), 100)]
        result = self.ns['simulate_global_stock']({}, 100, events, today, daily_sales_fn=lambda _:1.0)
        self.assertEqual(result['final_oos_date'], today + dt.timedelta(days=201))
        self.assertEqual(result['first_need_date'], result['final_oos_date'])
        self.assertEqual(result['stock_at_end'], 0)
        short = self.ns['simulate_global_stock'](
            {}, 100, events, today, max_days=10, daily_sales_fn=lambda _:1.0)
        self.assertEqual(short['stock_at_end'], 179)

    def test_2200_day_default_preserved(self):
        calls = []
        result = self.ns['simulate_global_stock'](
            {}, 3000, [], self.today, daily_sales_fn=lambda day: calls.append(day) or 1.0)
        self.assertEqual(len(calls), 2200)
        self.assertEqual(result['stock_at_end'], 800)
        self.assertIsNone(result['final_oos_date'])

    def test_confirmed_late_supply_case_orders_75_not_zero(self):
        result = self.recommend(self.row())
        self.assertEqual(result['推荐下单量'], 75)
        self.assertEqual(result['最晚下单日'], '2026-11-17')
        self.assertEqual(result['备货状态'], '建议下单')
        self.assertIsNone(result['_含剩余合同预估耗尽日'])

    def test_remaining_late_contract_cannot_cancel_earlier_gap(self):
        row = self.row()
        row['美西_多批次在途'] = ''
        late_contract = self.contract(300, dt.date(2027, 1, 26))
        result = self.recommend(row, contracts=[late_contract])
        self.assertEqual(result['推荐下单量'], 75)
        self.assertEqual(result['剩余合同数量'], 300)
        self.assertEqual(result['最晚下单日'], '2026-11-17')

    def test_only_unreachable_gap_does_not_order(self):
        row = self.row()
        row['美西_多批次在途'] = '2027-02-14:300'
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 0)
        self.assertEqual(result['最晚下单日'], '-')

    def test_start_day_supply_can_cover_that_day(self):
        row = self.row()
        self.assertEqual(self.gap(row, events=[(self.arrival,75)]), (0, None))

    def test_pre_arrival_stock_is_consumed_not_double_counted(self):
        row = self.row()
        self.assertEqual(self.gap(row, stock=100), (40, dt.date(2027,2,21)))

    def test_multiple_gaps_accumulate_and_late_surplus_stays_surplus(self):
        row = self.row()
        # Five days without stock, four days supplied, then six days without stock.
        result = self.gap(row, events=[(dt.date(2027,2,19),20), (dt.date(2027,3,1),300)])
        self.assertEqual(result, (55, self.arrival))

    def test_later_gap_moves_latest_order_date(self):
        row = self.row()
        row['美西_在仓'] = 100
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 40)
        self.assertEqual(result['最晚下单日'], '2026-11-24')

    def test_zero_and_blank_forecasts_remain_zero(self):
        row = self.row()
        for i, col in enumerate(self.ns['FORECAST_COLS']):
            row[col] = 0 if i % 2 else None
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 0)
        self.assertEqual(result['最晚下单日'], '-')

    def test_precise_daily_sales_and_late_supply(self):
        row = self.row()
        for day in range(1,29):
            row[self.ns['daily_forecast_col_name'](4,day)] = (
                65.0 if day == 1 else 75.0 if day == 20 else 0.0)
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 75)
        self.assertEqual(result['最晚下单日'], '2026-11-23')

    def test_small_order_zeroed_and_date_cleared(self):
        row = self.row()
        row['美西_多批次在途'] = '2027-02-15:300'
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 0)
        self.assertEqual(result['最晚下单日'], '-')
        self.assertIn('推荐下单量5pcs已归零', result['备注'])

    def test_exact_ten_is_not_zeroed(self):
        row = self.row()
        row['美西_多批次在途'] = '2027-02-16:300'
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 10)
        self.assertEqual(result['最晚下单日'], '2026-11-17')

    def test_no_supply_does_not_order_for_unreachable_65(self):
        row = self.row()
        row['美西_多批次在途'] = ''
        result = self.recommend(row)
        self.assertEqual(result['推荐下单量'], 75)
        self.assertEqual(result['最晚下单日'], '2026-11-17')

    def test_default_lead_remains_60(self):
        row = self.row()
        result = self.ns['compute_contract_recommendation'](
            row, [], {}, 0, self.today, dt.date(2026,11,24), dt.date(2027,2,1),34,30)
        self.assertEqual(result['推荐下单量'], 50)
        self.assertEqual(result['最晚下单日'], '2026-11-17')
        self.assertIn('默认60天交期', result['备注'])

    def test_fifo_deduction_is_not_changed(self):
        contracts = [self.contract(100,dt.date(2026,11,10)), self.contract(50,dt.date(2026,11,9))]
        contracts[0]['PO号'] = 'PO-2'
        result = self.recommend(self.row(),contracts=contracts,theoretical=65)
        self.assertEqual(result['本轮实际发货量'],65)
        self.assertEqual(result['剩余合同数量'],85)
        self.assertEqual([r['本轮扣减数量'] for r in result['合同扣减明细']], [50,15])

    def test_cutoff_and_empty_window(self):
        row = self.row()
        self.assertEqual(self.gap(row,cutoff=self.arrival), (5,self.arrival))
        self.assertEqual(self.gap(row,cutoff=self.arrival-dt.timedelta(days=1)), (0,None))
        self.assertEqual(self.gap(row,events=[(self.cutoff,300)]), (75,self.arrival))

    def test_coverage_boundaries_preserved(self):
        row = self.row()
        seen = []
        original = self.ns['compute_contract_order_gap']
        def capture(*args):
            seen.append(args[5])
            return original(*args)
        with patch.dict(self.ns, compute_contract_order_gap=capture):
            self.recommend(row)
            row['美西_多批次在途'] = ''
            self.recommend(row)
        self.assertEqual(seen, [dt.date(2027,3,16),dt.date(2027,3,15)])

    def test_seeded_orders_cover_every_reachable_day_and_are_minimal(self):
        rng = random.Random(170011)
        for case in range(100):
            with self.subTest(case=case):
                start = self.today+dt.timedelta(days=10)
                end = self.today+dt.timedelta(days=40)
                stock = rng.randrange(0,30)
                events = [(self.today+dt.timedelta(days=rng.randrange(1,60)),rng.randrange(1,60))
                          for _ in range(6)]
                demand = {self.today+dt.timedelta(days=i):rng.randrange(0,30)/4 for i in range(1,41)}
                qty, first = self.gap({},stock,events,start,end,daily=lambda day:demand[day])
                def trace(order_qty):
                    inventory = stock
                    gaps = {}
                    for day, sales in demand.items():
                        inventory += sum(q for date,q in events if date == day)
                        if day == start:
                            inventory += order_qty
                        missing = max(0,sales-inventory)
                        inventory = max(0,inventory-sales)
                        if day >= start and missing > 1e-8:
                            gaps[day] = missing
                    return gaps
                no_order = trace(0)
                self.assertEqual(first,min(no_order) if no_order else None)
                self.assertEqual(trace(qty),{})
                if qty:
                    self.assertTrue(trace(qty-1))

    @unittest.skipUnless(os.environ.get('FCL_BASELINE_SOURCE'), 'Optional before/after baseline')
    def test_unbounded_simulation_matches_baseline_in_100_cases(self):
        baseline, _ = load_core(Path(os.environ['FCL_BASELINE_SOURCE']))
        rng = random.Random(170012)
        for case in range(100):
            stock = rng.randrange(0,300)
            events = [(self.today+dt.timedelta(days=rng.randrange(1,120)),rng.randrange(1,80))
                      for _ in range(4)]
            kwargs = dict(max_days=rng.randrange(1,200),daily_sales_fn=lambda day:(day.day % 7)/3)
            before = baseline['simulate_global_stock']({},stock,events,self.today,**kwargs)
            after = self.ns['simulate_global_stock']({},stock,events,self.today,**kwargs)
            self.assertEqual(after,before,case)


if __name__ == '__main__':
    unittest.main(verbosity=2)
