#!/user/bin/env python
# coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : backtest_evaluation.py
#@time   : 2023-04-08 00:56:13
'''

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)

import uqer
import pprint
from codes.research.data_process.data_fetcher import DataFetcher
from codes.backtest.BackTester import backtest_quick

uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)


def backtest_report(start_date='2021-05-14', end_date='2021-05-14', product_id='rb', result_df: pd.DataFrame = None,
                    capital: int = 1000000):
    all_trade_dates = data_fetcher.get_all_trade_dates(start_date, end_date)
    # ret = []
    transactions = []
    _transaction_path = os.path.join(_base_dir, 'data\\backtest\\transactions_{0}.csv'.format(product_id))

    for d in all_trade_dates:
        result_df, ret_transaction = backtest_quick(data_fetcher=data_fetcher, product_id=product_id, trade_date=d,
                                                    result_df=result_df, capital=capital)
        # ret.append([d, ret_date[0], ret_date[1], ret_date[2]])
        [_.append(d) for _ in ret_transaction]

        transactions.extend(ret_transaction)

    df_transaction = pd.DataFrame(transactions,
                                  columns=['idx', 'instrument_id', 'trans_type', 'last_price',
                                           'fill_price', 'fill_lot', 'pos', 'fee', 'open_time', 'close_time',
                                           'holding_time', 'curr_return', 'trade_date'])
    df_transaction.to_csv(_transaction_path, index=False)
    return result_df


def evaluation_backtesting(init_capital=1000000):
    _backtest_path = os.path.join(_base_dir, 'data\\backtest\\backtest.csv')
    df = pd.read_csv(_backtest_path)
    return_df = df.groupby('trade_date', as_index=False)[['total_return_final', 'max_margin']].sum()
    live_risk_df = df.groupby('trade_date', as_index=False)[['max_risk_ratio']].max()
    mv_lst = list(return_df['total_return_final'].cumsum() + init_capital)
    nv_lst = [round(_ / init_capital, 5) for _ in mv_lst]
    live_risk_lst = list(return_df['max_margin'] / init_capital)
    n = len(nv_lst)
    log_return_lst = [0]
    import math
    for i in range(n - 1):
        _val = math.log(nv_lst[i - 1] / nv_lst[i])
        log_return_lst.append(_val)
    annual_return = sum(log_return_lst) / len(log_return_lst) * 250
    annual_vol = np.array((log_return_lst)).std() * np.sqrt(250)
    max_nv = nv_lst[0]
    max_drawdown = 0
    for item in nv_lst:
        max_drawdown = max(item / max_nv - 1, max_drawdown)
        max_nv = max(item, max_nv)
    print("annual return=>", annual_return, "annual volatility=>", annual_vol, "max risk ratio=>", max(live_risk_lst),
          "max drawdown=>", max_drawdown)
    _backtest_path = os.path.join(_base_dir, 'data\\backtest\\backtest.jpg')
    plt.plot(nv_lst)
    plt.savefig(_backtest_path)


def main():
    start_date = '2021-02-01'
    end_date = '2021-02-26'
    # product_ids = ['rb', 'm', 'p', 'TA']
    product_ids = ['RB', 'M', 'AU', 'AG', 'NI', 'I', 'SC', 'FU', 'Y', 'P', 'RM', 'CU', 'AL', 'ZN', 'RU', 'BU', 'B', 'C',
                   'SR', 'CF', 'TA']
    capital = 2000000
    risk_ratio = 0.5
    # TODO allocate by performance(e.g. sharp ratio), not equal
    each_product_capital = int(capital * risk_ratio / len(product_ids))
    result_df = pd.DataFrame(
        {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return_final': [],
         'total_return_unclose': [],
         'total_fee': [],
         'unclosed_value': [], 'precision': [], 'long_open': [], 'short_open': [],
         'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
         'average_holding_time': [], 'max_holding_time': [], 'min_holding_time': [], 'max_margin': [],
         'max_risk_ratio': []
         })
    _backtest_path = os.path.join(_base_dir, 'data\\backtest\\backtest.csv')
    for product_id in product_ids:
        result_df = backtest_report(start_date=start_date, end_date=end_date, product_id=product_id,
                                    result_df=result_df, capital=each_product_capital)
    result_df.to_csv(_backtest_path, index=False)
    evaluation_backtesting(init_capital=capital)


if __name__ == '__main__':
    main()
