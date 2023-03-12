#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 14:23
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : factor_calculation.py


import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import talib as ta
import uqer
from uqer import DataAPI
from datetime import time
from datetime import datetime

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.research.data_process.data_fetcher import DataFetcher
from codes.utils.utils import get_mul_num


def cal_oir(bid_price: list = [], bid_vol: list = [], ask_price: list = [], ask_vol: list = [],
            n_rows: int = 0) -> tuple:
    """
    calculate order imbalance factors
    :param bid_price:
    :param bid_vol:
    :param ask_price:
    :param ask_vol:
    :param n_rows:
    :return:
    """
    v_b = [0]
    v_a = [0]

    for i in range(1, n_rows):
        v_b.append(
            0 if bid_price[i] < bid_price[i - 1] else bid_vol[i] - bid_vol[i - 1] if bid_price[i] == bid_price[
                i - 1] else bid_vol[i])

        v_a.append(
            0 if ask_price[i] < ask_price[i - 1] else ask_vol[i] - ask_vol[i - 1] if ask_price[i] == ask_price[
                i - 1] else ask_vol[i])
    lst_oi = []
    lst_oir = []
    lst_aoi = []
    for idx, item in enumerate(v_a):
        lst_oi.append(v_b[idx] - item)
        lst_oir.append((v_b[idx] - item) / (v_b[idx] + item) if v_b[idx] + item != 0 else 0.0)
        lst_aoi.append((v_b[idx] - item) / (ask_price[idx] - bid_price[idx]))
    return lst_oi, lst_oir, lst_aoi


def _cal_turning_item(x: list) -> tuple:
    if not x:
        return 0, np.nan
    if len(x) == 1:
        return 0, np.nan
    if len(x) == 2:
        return -1, x[0]
    try:
        if (x[-1] - x[-2]) * (x[-2] - x[-3]) > 0:
            return -2, x[0]
        else:
            return -1, x[1]
    except Exception as ex:
        raise ValueError("Error to cal turning with error:{0}".format(ex))


def cal_turning(x: list) -> tuple:
    idx_lst = []
    val_lst = []
    if len(x) == 1:
        idx_lst.append(0)
        val_lst.append(np.nan)
    elif len(x) == 2:
        _idx, _val = _cal_turning_item(x)
        return [0], [_val]
    else:
        _len = len(x)
        idx_lst.append(0)
        val_lst.append(np.nan)
        idx_lst.append(0)
        val_lst.append(x[0])
        _idx, _val = _cal_turning_item(x)
        idx_lst.append(2 + _idx)
        val_lst.append(_val)
        for idx in range(3, _len, 1):
            _idx, _val = _cal_turning_item(x[idx - 3:idx])
            idx_lst.append(idx + _idx)
            val_lst.append(_val)
    return idx_lst, val_lst


def cal_slope(x: list, turn_idx: list, turn_val: list) -> list:
    assert len(x) == len(turn_idx)
    assert len(x) == len(turn_val)
    ret = []
    for idx, item in enumerate(x):
        if idx == 0:
            ret.append(np.nan)
            continue
        ret.append((x[idx] - turn_val[idx]) / (idx - turn_idx[idx]))
    return ret


def cal_cos(x: list, turn_idx: list, turn_val: list) -> list:
    assert len(x) == len(turn_idx)
    assert len(x) == len(turn_val)
    ret = []
    for idx, item in enumerate(x):
        if idx == 0:
            ret.append(0)
            continue
        a = np.array([idx - turn_idx[idx], item - turn_val[idx]])
        b = np.array([0, 1])
        ret.append(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return ret


def calculate_raw_features(data_fetch: DataFetcher = None, product_id: str = '', instrument_id: str = '',
                           start_date: str = '', end_date: str = ''):
    data_fetch.get_instrument_mkt(product_ids=[product_id], start_date=start_date, end_date=end_date)
    _instruments = list(set(data_fetch.instrument_cache['ticker']))
    data_fetch.get_instrument_contract(instrument_ids=_instruments, product_ids=[product_id])

    tick_mkt = data_fetch.load_tick_data(start_date=start_date, end_date=end_date, instrument_ids=_instruments,
                                         main_con_flag=1, filter_start_time=None, filter_end_time=None, if_filter=True)
    tick_mkt = tick_mkt.set_index('InstrumentID').join(
        data_fetch.contract_cache[['ticker', 'contMultNum']].set_index('ticker')).reset_index()
    # _mul_num = utils.get_mul_num(instrument_id) or 1
    tick_mkt['vwap'] = (tick_mkt['Turnover'] / tick_mkt['Volume']) / tick_mkt['contMultNum']
    tick_mkt['wap'] = (tick_mkt['BidPrice1'] * tick_mkt['AskVolume1'] + tick_mkt['AskPrice1'] * tick_mkt[
        'BidVolume1']) / (tick_mkt['AskVolume1'] + tick_mkt['BidVolume1'])
    # tick_mkt['log_return'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    tick_mkt['log_return'] = np.log(tick_mkt['LastPrice']).diff()
    # tick_mkt['wap_log_return'] = np.log(tick_mkt['wap']) - np.log(tick_mkt['LastPrice'])
    tick_mkt['wap_log_return'] = np.log(tick_mkt['wap']).diff()

    _x, _y = tick_mkt.shape
    _diff = list(tick_mkt['InterestDiff'])
    _vol = list(tick_mkt['Volume'])

    open_close_ratio = []
    for idx, item in enumerate(_diff):
        try:
            open_close_ratio.append(item / (_vol[idx] - item))
        except Exception as ex:
            open_close_ratio.append(open_close_ratio[-1])

    # cal ori factor(paper factor)
    lst_oi, lst_oir, lst_aoi = cal_oir(list(tick_mkt['BidPrice1']), list(tick_mkt['BidVolume1']),
                                       list(tick_mkt['AskPrice1']), list(tick_mkt['AskVolume1']), n_rows=_x)

    # cal cos factor(market factor)
    _lst_last_price = list(tick_mkt['LastPrice'])
    _lst_turn_idx, _lst_turn_val = cal_turning(_lst_last_price)
    _lst_slope = cal_slope(_lst_last_price, _lst_turn_idx, _lst_turn_val)
    _lst_cos = cal_cos(_lst_last_price, _lst_turn_idx, _lst_turn_val)

    dif, dea, macd = ta.MACD(tick_mkt['LastPrice'], fastperiod=12, slowperiod=26, signalperiod=9)

    tick_mkt['open_close_ratio'] = open_close_ratio
    tick_mkt['price_spread'] = (tick_mkt['BidPrice1'] - tick_mkt['AskPrice1']) / (
            tick_mkt['BidPrice1'] + tick_mkt['AskPrice1'] / 2)
    tick_mkt['buy_sell_spread'] = abs(tick_mkt['BidPrice1'] - tick_mkt['AskPrice1'])
    tick_mkt['oi'] = lst_oi
    tick_mkt['oir'] = lst_oir
    tick_mkt['aoi'] = lst_aoi
    tick_mkt['slope'] = _lst_slope
    tick_mkt['cos'] = _lst_cos
    tick_mkt['macd'] = macd
    tick_mkt['dif'] = dif
    tick_mkt['dea'] = dea
    tick_mkt['bs_tag'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: 1 if list(x)[-1] > list(x)[0] else -1)
    tick_mkt['bs_vol'] = tick_mkt['bs_tag'] * tick_mkt['Volume']
    return tick_mkt


def feature_resample(df: pd.DataFrame = None, freq: str = '60S', datetime_col: str = ''):
    df.index = pd.to_datetime(df['UpdateTime'])
    # df['trade_date'] = [item.split(' ')[0] for item in df['UpdateTime']]
    # df['time'] = [item.split(' ')[1] for item in df['UpdateTime']]
    _labels = df[['wap_log_return']].resample(freq, label='left').sum()
    # df = df.set_index(datetime_col).resample(freq)
    return df
    # df.resample(freq)
    # .apply(
    #     {"TRADE_PRICE": "ohlc", "TRADE_QTY": "sum", "TRADE_AMOUNT": "sum", "BS_VOL": "sum"}
    # )
    # .dropna()
    # .shift(1)
    # .dropna()
    # .droplevel(level=0, axis=1)
    # .rename(columns={"TRADE_QTY": "vol", "TRADE_AMOUNT": "amount", "BS_VOL": "bs_vol"})


if __name__ == '__main__':
    uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
    # data_fetch = DataFetcher(uqer_client)
    # df_features = calculate_raw_features(data_fetch=data_fetch, product_id='rb', start_date='20210704',
    #                                      end_date='20210705')
    # df_features.to_csv('feature_sample.csv')
    df_features = pd.read_csv('feature_sample.csv')
    feature_resample(df=df_features, freq='300S')
    # print(df_features.shape)
