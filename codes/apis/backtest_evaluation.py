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

import pandas as pd

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)

import uqer
import pprint
from codes.research.data_process.data_fetcher import DataFetcher
from codes.backtest.BackTester import backtest_quick

uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)


def backtest_report(start_date='2021-05-14', end_date='2021-05-14', product_id='rb'):
    all_trade_dates = data_fetcher.get_all_trade_dates(start_date, end_date)
    ret = []
    transactions = []
    _transaction_path = os.path.join(_base_dir, 'data\\backtest\\transactions_{0}.csv'.format(product_id))
    for d in all_trade_dates:
        ret_date = backtest_quick(data_fetcher=data_fetcher, product_id=product_id, trade_date=d)
        ret.append([d, ret_date[0], ret_date[1], ret_date[2]])
        [_.append(d) for _ in ret_date[3]]

        transactions.extend(ret_date[3])

    df_transaction = pd.DataFrame(transactions,
                                  columns=['idx', 'instrument_id', 'trans_type', 'last_price',
                                           'fill_price', 'fill_lot', 'pos', 'fee', 'open_time', 'close_time',
                                           'holding_time', 'curr_return', 'trade_date'])
    df_transaction.to_csv(_transaction_path, index=False)
    return ret


def main():
    start_date = '2021-02-01'
    end_date = '2021-02-26'
    # product_ids = ['rb', 'm', 'p', 'TA']
    product_ids = ['TA']
    for product_id in product_ids:
        backtest_report(start_date=start_date, end_date=end_date, product_id=product_id)


if __name__ == '__main__':
    main()
