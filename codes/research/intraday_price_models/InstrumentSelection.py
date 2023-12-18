#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : InstrumentSelection.py
# @Author: kiki
# @Date  : 2023/12/18
# @Desc  :
# @Contact : rpyxqi@gmail.com

import os
import sys
import uqer
from uqer import DataAPI
import numpy as np

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../")))
sys.path.append(_base_dir)

from codes.research.data_process.data_fetcher import DataFetcher


# get_data_cube 只能在客户端跑
def get_instrument_mkt(product_ids: list = [], start_date: str = '', end_date: str = ''):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df.loc[(df['mainCon'] == 1) | (df['smainCon'] == 1)]  # 筛选出主力和次主力合约
    product_ids = [item.upper() for item in product_ids]
    ret_df = df.loc[(item in product_ids for item in df['contractObject'])]
    _format_trade_date = [item.replace('-', '') for item in ret_df['tradeDate']]
    ret_df['tradeDate'] = _format_trade_date
    return ret_df


start_date = '20231101'
end_date = '20231215'
product_ids = ['rb', 'm', 'au', 'ag', 'ni', 'i', 'sc', 'fu', 'y', 'p', 'm', 'RM', 'cu', 'al', 'zn', 'ru', 'bu', 'b',
               'c', 'SR', 'CF', 'TA', 'sp', ]
df = get_instrument_mkt(
    product_ids=product_ids, start_date='20231116', end_date='20231215')
df['turnoverRate'] = df['turnoverVol'] / df['openInt']

df['contractValue'] = df['turnoverValue'] / df['turnoverVol']
_df = df[df.mainCon == 1]
_df_turnover = _df.groupby(['contractObject'], as_index=True)['turnoverRate', 'contractValue'].mean()
_df_instrument_id = _df.groupby(['contractObject'], as_index=True)['ticker', 'secShortName'].last()
_df_turnover = _df_instrument_id.join(_df_turnover).reset_index().sort_values(by='turnoverRate',
                                                                              ascending=False)

comision = {'RB': 0.0001,
            'M': 1.51,
            'AU': 2.01,
            'AG': 0.00001,  # 黄金6，12.。。 10.01？？
            'NI': 3.01,
            'I': 0.00010,
            'SC': 20.01,
            'FU': 0.000050,  # 非1,5,9:0.00001+0.01
            'Y': 2.51,
            'P': 2.51,
            'M': 1.51,
            'RM': 1.51,
            'CU': 0.000050,
            'AL': 3.01,
            'ZN': 3.01,
            'RU': 3.01,
            'BU': 0.00010,
            'B': 2.01,  # 豆二1.01， 豆一2.01
            'C': 1.21,
            'SR': 3.01,
            'CF': 4.31,
            'TA': 3.01,
            # 'SP':,

            }
close_today = {'RB': 0.0001,
               'M': 1.51,
               'AU': 0.0,  # 看手续费
               'AG': 0.00001,
               'NI': 3.01,
               'I': 0.00010,
               'SC': 0.0,
               'FU': 0.0,
               'Y': 2.51,
               'P': 2.51,
               'M': 1.51,
               'RM': 1.51,
               'CU': 0.00010,
               'AL': 3.01,
               'ZN': 0.0,  # 看手续费
               'RU': 0.0,
               'BU': 0.00010,
               'B': 2.01,  # 豆二1.01， 豆一2.01
               'C': 1.21,
               'SR': 0.0,
               'CF': 0.0,
               'TA': 0.0,
               # 'SP':,
               }
commission_lst = []
close_today_lst = []


def _process_fee(val, fee):
    if fee >= 1.0:
        return fee
    else:
        if fee > 0:
            return round(val * fee, 2) + 0.01
    return 0.0


print(_df_turnover.columns)
for product, _, _, _, contract_value in list(_df_turnover.values):
    _commission = comision.get(product.upper()) or 0.0
    _close = close_today.get(product.upper()) or 0.0
    commission_lst.append(_process_fee(contract_value, _commission))
    close_today_lst.append(_process_fee(contract_value, _close))
_df_turnover['comission'] = commission_lst
_df_turnover['close_today'] = close_today_lst

start_date = "20231211"
end_date = "20231215"
m_vol = []
for _, ticker, _, _, _, _, _ in list(_df_turnover.values):
    print('query:', ticker)
    try:
        ticker = ticker.upper()
        _df = DataAPI.get_data_cube(ticker, ["turnoverVol", "turnoverValue"], start=start_date, end=end_date, freq='1m',
                                    style='sat',
                                    adj=None)[ticker].reset_index()
        m_vol.append((_df['turnoverValue'] / _df['turnoverVol']).std())
    except Exception as ex:
        m_vol.append(0.0)
_df_turnover['min_vol'] = m_vol
_df_turnover['total_fee'] = _df_turnover['comission'] + _df_turnover['close_today']
_df_turnover['fee_rate'] = _df_turnover['total_fee'] / _df_turnover['contractValue']
_df_turnover.to_csv("instrument_turnover.csv", index=False, encoding='UTF-8', float_format='%.5f')
