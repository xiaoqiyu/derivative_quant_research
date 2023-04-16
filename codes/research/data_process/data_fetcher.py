#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 14:23
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : data_fetcher.py

import numpy as np
import talib as ta
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import uqer
from uqer import DataAPI
from datetime import time

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
print(_base_dir)
sys.path.append(_base_dir)
from codes.utils.logger import Logger
from codes.utils.define import *

log_name = os.path.split(__file__)[-1].strip('.py')
log_path = os.path.abspath(os.path.join(__file__, "../../../../data/logs/{}".format(log_name)))
logger = Logger(log_path, 'INFO', __name__).get_log()


class DataFetcher(object):
    def __init__(self, api_obj=None):
        self.api_obj = api_obj
        self.instrument_cache = pd.DataFrame()
        self.contract_cache = pd.DataFrame()
        self.mkt_cache_dir = TICK_MKT_DIR

    def get_all_trade_dates(self, start_date='', end_date='') -> list:
        '''
        :param start_date:str, 'yyyymmdd'
        :param end_date:str,'yyyymmdd'
        :return: list,['yyyymmdd']
        '''
        df_dates = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate=start_date, endDate=end_date, isOpen='1',
                                       pandas='1')
        t_dates = list(set(df_dates['calendarDate']))
        return sorted(t_dates)
        # return sorted([item.replace('-', '') for item in t_dates])

    def get_week_start_end(self, start_date='', end_date='') -> list:
        df_dates = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate=start_date, endDate=end_date, isOpen='1',
                                       pandas='1')
        weekend_idx = list(df_dates[df_dates.isWeekEnd == 1].index)
        all_trade_dates = list(df_dates['calendarDate'])
        ret = []
        start_idx = 0
        for idx in weekend_idx:
            ret.append((all_trade_dates[start_idx], all_trade_dates[idx]))
            start_idx = idx + 1
        return ret

    def get_instrument_mkt(self, product_ids: list = [], start_date: str = '', end_date: str = ''):
        df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
        df = df.loc[(df['mainCon'] == 1) | (df['smainCon'] == 1)]  # 筛选出主力和次主力合约
        product_ids = [item.upper() for item in product_ids]
        self.instrument_cache = df.loc[(item in product_ids for item in df['contractObject'])]
        self.instrument_cache['tradeDate'] = [item.replace('-', '') for item in self.instrument_cache['tradeDate']]

    def get_instrument_contract(self, instrument_ids: list = [], product_ids: list = []):
        df = DataAPI.FutuGet(secID=u"", ticker=instrument_ids, exchangeCD=u"", contractStatus="",
                             contractObject=product_ids,
                             prodID="", field=u"", pandas="1")
        if df.shape[0] == 0:
            logger.error(
                "Query param for contract is not correct, instrument:{0}, product id:{1}".format(instrument_ids,
                                                                                                 product_ids))
        self.contract_cache = df

    def load_tick_data(self, start_date: str = '', end_date: str = '', instrument_ids: list = [],
                       product_ids: list = [], main_con_flag: int = 1,
                       filter_start_time: time = None, filter_end_time: time = None, if_filter: bool = True):
        '''

        :param start_date:
        :param end_date:
        :param product_id:
        :param main_con_flag: 0: smain_con, 1: main_con
        :param filter_start_time
        :param filter_end_time
        :param if_filter
        :return:
        '''
        all_trading_dates = self.get_all_trade_dates(start_date=start_date, end_date=end_date)
        all_trading_dates = sorted([item.replace('-', '') for item in all_trading_dates])
        product_ids = [item.upper() for item in product_ids]
        if instrument_ids:
            df_instruments = self.instrument_cache.loc[
                (item in instrument_ids for item in self.instrument_cache['ticker'])]
        else:
            df_instruments = self.instrument_cache.loc[
                (item in product_ids for item in self.instrument_cache['contractObject'])]
        if main_con_flag:
            df_instruments = df_instruments[df_instruments.mainCon == 1]
        else:
            df_instruments = df_instruments[df_instruments.smainCon == 1]
        ret = pd.DataFrame()
        for d in all_trading_dates:
            _df_instrument = df_instruments[df_instruments.tradeDate == d]
            columns = list(_df_instrument.columns)
            row = list(_df_instrument.values[0])
            _instrument = row[columns.index('ticker')]
            _exchange = row[columns.index('exchangeCD')]
            _tick_mkt_path = os.path.join(self.mkt_cache_dir + d[:6], exchange_map.get(_exchange),
                                          '{0}_{1}.csv'.format(_instrument, d))
            if os.path.exists(_tick_mkt_path):
                tick_mkt = pd.read_csv(_tick_mkt_path, encoding='gbk')
                ret = pd.concat([ret, tick_mkt])
            else:
                logger.warn("tick mkt path not exist:{0}".format(_tick_mkt_path))
        ret.columns = tb_cols
        if if_filter:
            ret.index = pd.to_datetime(ret['UpdateTime'])
        ret = pd.concat(
            [ret.loc[time(9, 0): time(11, 30)], ret.loc[time(13, 30): time(15, 0)], ret.loc[time(21, 0):time(23, 0)]])
        return ret


if __name__ == '__main__':
    uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
    data_fetcher = DataFetcher(uqer_client)
    # data_fetcher.get_instrument_mkt(product_ids=['rb'], start_date='20210704', end_date='20210715')
    # # ret = obj.load_tick_data(start_date='20210704', end_date='20210715', product_ids=['rb'], main_con_flag=1,
    # #                          if_filter=True)
    # data_fetcher.get_instrument_contract(product_ids=['rb'])
    # tick_mkt = data_fetcher.load_tick_data(start_date='20210704', end_date='20210715', instrument_ids=['rb2110'],
    #                                        main_con_flag=1,
    #                                        if_filter=True)
    # tick_mkt = tick_mkt.set_index('InstrumentID').join(
    #     data_fetcher.contract_cache[['ticker', 'contMultNum']].set_index('ticker')).reset_index()
    data_fetcher.get_week_start_end(start_date='2021-03-01', end_date='2021-12-31')
