#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:05
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : BackTester.py
import math
import os
import sys

import pandas as pd

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)

import time
from codes.backtest.Factor import Factor
from codes.backtest.Position import Position
from codes.backtest.Account import Account
from codes.strategy.MinSignal import MinSignal
from codes.strategy.ClfSignal import ClfSignal
from codes.research.data_process.data_fetcher import DataFetcher
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from codes.utils import helper
from codes.utils.define import *
from codes.utils.utils import *
from codes.utils.logger import Logger
from copy import deepcopy
import configparser

uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)
_log_path = os.path.join(_base_dir, 'data\logs\{0}'.format(os.path.split(__file__)[-1].strip('.py')))
logger = Logger(_log_path, 'INFO', __name__).get_log()

conf_commision = {'RB': 0.0001,
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
conf_close_today = {'RB': 0.0001,
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
# 每日千分之一回报，log(1.001)=0.000999，~ 0.001，年化 0.25，假设两倍杠杆；则实际交易信号收益目标为万五
conf_stop_profit = {'RB': 0.0005,
               'M': 0.0005,
               'AU': 0.0005,
               'AG': 0.0005,
               'NI': 0.0005,
               'I': 0.0005,
               'SC': 0.0005,
               'FU': 0.0005,
               'Y': 0.0005,
               'P': 0.0005,
               'M': 0.0005,
                    'RM': 0.0005,
                    'CU': 0.0005,
                    'AL': 0.0005,
                    'ZN': 0.0005,
                    'RU': 0.0005,
                    'BU': 0.0005,
                    'B': 0.0005,
                    'C': 0.0005,
                    'SR': 0.0005,
                    'CF': 0.0005,
                    'TA': 0.0005,
                    # 'SP':,
                    }

conf_stop_loss = {'RB': 0.001,
             'M': 0.001,
             'AU': 0.001,
             'AG': 0.001,
             'NI': 0.001,
             'I': 0.001,
             'SC': 0.001,
             'FU': 0.001,
             'Y': 0.001,
             'P': 0.001,
             'M': 0.001,
                  'RM': 0.001,
                  'CU': 0.001,
                  'AL': 0.001,
                  'ZN': 0.001,
                  'RU': 0.001,
                  'BU': 0.001,
                  'B': 0.001,
                  'C': 0.001,
                  'SR': 0.001,
                  'CF': 0.001,
                  'TA': 0.001,
                  # 'SP':,
                  }

conf_vol_limit = {'RB': 5,
             'M': 5,
             'AU': 5,
             'AG': 5,
             'NI': 5,
             'I': 5,
             'SC': 5,
             'FU': 5,
             'Y': 5,
             'P': 5,
             'M': 5,
                  'RM': 5,
                  'CU': 5,
                  'AL': 5,
                  'ZN': 5,
                  'RU': 5,
                  'BU': 5,
                  'B': 5,
                  'C': 5,
                  'SR': 5,
                  'CF': 5,
                  'TA': 5,
                  # 'SP':,
                  }

conf_change_ticks = {'RB': 1,
                'M': 1,
                'AU': 1,
                'AG': 1,
                'NI': 1,
                'I': 1,
                'SC': 1,
                'FU': 1,
                'Y': 1,
                'P': 1,
                'M': 1,
                     'RM': 1,
                     'CU': 1,
                     'AL': 1,
                     'ZN': 1,
                     'RU': 1,
                     'BU': 1,
                     'B': 1,
                     'C': 1,
                     'SR': 1,
                     'CF': 1,
                     'TA': 1,
                     # 'SP':,
                     }


# 主动买/卖，则以对手方一档价格成交，否则以中间价成交，暂不考虑大单
def get_fill_ret(order=None, mkt=None):
    if mkt is None:
        mkt = []
    if order is None:
        order = []
    _order_type, _price, _lot = order
    bid_price1, ask_price1, bid_vol1, ask_vol1 = mkt[-5:-1]
    _last_price = mkt[3]
    if _order_type == LONG:
        if _lot <= ask_vol1 and (not _price or _price >= ask_price1):
            return [ask_price1, _lot]
    if _order_type == SHORT:
        if _lot <= bid_vol1 and (not _price or _price <= bid_price1):
            return [bid_price1, _lot]
    return [round((bid_price1 + ask_price1) / 2, 2), _lot]


# TODO check usage
def stop_profit_loss(risk_input: dict = {}, risk_conf: dict = {}) -> tuple:
    '''
     # the range in the past 1 min is laess than 2(N) ticks, stop profit/loss will be downgraded,then tend to close the position
    #easier; if not, then the oppisite side, try to earn more in the order, at the same time, it may lose more, so it should
    #improve the strategy/signal
    :param risk_input:
    :param risk_conf:
    :return:
    '''
    _last_price = risk_input.get('last_price').get_lst()[-120:]
    _range = max(_last_price) - min(_last_price)
    _stop_lower = risk_conf.get('stop_lower')
    _stop_upper = risk_conf.get('stop_upper')
    if _range < 2:
        return _stop_lower, _stop_lower
    else:
        return _stop_upper, _stop_upper


def backtest_quick(data_fetcher: DataFetcher = None, product_id: str = 'rb', trade_date: str = '2021-07-01',
                   options: dict = {}) -> tuple:
    # Init contract static and daily data
    # TODO 可优化，data_fetcher 只在外面对所有品种和时间查询一次
    data_fetcher.get_instrument_mkt(product_ids=[product_id], start_date=trade_date, end_date=trade_date)
    _instruments = list(set(data_fetcher.instrument_cache['ticker']))
    data_fetcher.get_instrument_contract(instrument_ids=_instruments, product_ids=[product_id])
    instrument_id = list(data_fetcher.instrument_cache[data_fetcher.instrument_cache.mainCon == 1]['ticker'])[0]
    tick_mkt = data_fetcher.load_tick_data(start_date=trade_date, end_date=trade_date, instrument_ids=_instruments,
                                           main_con_flag=1, filter_start_time=None, filter_end_time=None,
                                           if_filter=True)
    multiplier = list(data_fetcher.contract_cache[data_fetcher.contract_cache.ticker == instrument_id]['contMultNum'])[
        0]
    tick_mkt = tick_mkt.set_index('InstrumentID').join(
        data_fetcher.contract_cache[['ticker', 'contMultNum']].set_index('ticker')).reset_index()

    # init factor, position account signal
    factor = Factor(product_id=product_id, instrument_id=instrument_id, trade_date=trade_date)
    position = Position()
    account = Account()
    signal = ClfSignal(factor=factor, position=position, instrument_id=instrument_id,
                       trade_date=trade_date, product_id=product_id)

    try:
        _mkt_row = data_fetcher.instrument_cache[data_fetcher.instrument_cache.ticker == instrument_id][
            ['turnoverValue', 'turnoverVol']].values
        _contract_value = _mkt_row[0][0] / _mkt_row[0][1]
    except Exception as ex:
        raise ValueError("no mkt cache for product:{} and date:{}".format(product_id, trade_date))

    # Init backtest params
    fee_lst = [float(conf_commision.get(product_id.upper(), 0)), float(conf_close_today.get(product_id.upper(), 0))]
    fee_lst = list(map(
        lambda fee_rate: round(_contract_value * fee_rate, 2) + 0.01 if 0 < fee_rate < 1 else fee_rate,
        fee_lst))

    fee = sum(fee_lst)
    fee_ratio = round(fee / _contract_value, 4)
    profit_ratio = conf_stop_profit.get(product_id.upper())
    loss_ratio = conf_stop_loss.get(product_id.upper())
    change_tick = conf_change_ticks.get(product_id.upper())
    total_return = 0.0
    close_price = 0.0  # not the true close price, to close the position
    _text_lst = ['long_open', 'long_close', 'short_open', 'short_close']
    options.update({'instrument_id': instrument_id})
    options.update({'multiplier': multiplier})
    options.update({'trade_date': trade_date})

    options.update({'risk_duration': 30})
    options.update({'vol_limit': 5})
    options.update({'signal_names': 'ma'})
    options.update({'open_fee': fee_lst[0]})
    options.update({'close_today_fee': fee_lst[1]})

    # 逐tick backtest开始
    values = tick_mkt.values
    total_tick_num = len(values)
    ab_cnt = 0
    close_timestamp = None
    for idx, item in enumerate(values):
        _last = item[3]
        _update_time = item[2]
        _mul_num = item[-1]
        # TODO remove hardcode for the threshold
        if float(ab_cnt / total_tick_num) > 0.25:
            logger.warn("Error for mkt/tick, ignore the trade date:{0}".format(trade_date))
            return total_return, account.fee, 0, account.transaction
        try:
            curr_factor = factor.update_factor(item)
        except Exception as ex:
            logger.warn(
                "Ignore current item, Update factor error for item:{0}, last_factor:{1}, and error:{2} ".format(item,
                                                                                                                factor.curr_factor,
                                                                                                                ex))
            ab_cnt += 1
            continue

        close_price = _last
        close_timestamp = _update_time
        # options.update({'factor': curr_factor})

        # Get signal
        options.update({'stop_profit_ratio': profit_ratio + fee_ratio})
        options.update({'stop_loss_ratio': max(loss_ratio - fee_ratio, 0.0)})
        _stop_profit = _last * profit_ratio + fee_ratio
        stop_profit_val = math.floor(_stop_profit) + change_tick if _stop_profit % change_tick > 0 else _stop_profit
        _stop_loss = max(loss_ratio - fee_ratio, 0.0) * _last
        stop_loss_val = change_tick if _stop_loss == 0 else math.ceil(
            _stop_loss) - change_tick if _stop_loss % change_tick > 0 else _stop_loss
        options.update({'stop_profit': stop_profit_val})
        options.update({'stop_loss': stop_loss_val})
        _signal = signal(params=options)
        if _signal.signal_type == LONG_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[LONG, _signal.price, _signal.vol], mkt=item)
            logger.info("Long Open with update time:{0},filled price:{1}, filled lot:{2}------".format(_update_time,
                                                                                                       _fill_price,
                                                                                                       _fill_lot))
            if _fill_lot > 0:
                dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                account.add_transaction(
                    [idx,
                     instrument_id,
                     _text_lst[define.LONG_OPEN],
                     _last,  # 最新价格
                     _fill_price,  # 成交价格
                     _fill_lot,  # 成交数量
                     _fill_price,  # 持仓成本
                     fee_lst[0] * _fill_lot,  # 费用
                     _update_time.split(' ')[-1],  # 开仓时间
                     _update_time.split(' ')[-1],  # 平仓时间
                     0.0,  # 持有时间
                     0.0])  # 收益
                position.update_position(instrument_id=instrument_id, long_short=LONG, price=_fill_price,
                                         timestamp=_update_time, vol=_fill_lot, order_type=LONG_OPEN)
                account.update_fee(fee_lst[0] * _fill_lot)

        elif _signal.signal_type == SHORT_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[SHORT, _signal.price, _signal.vol], mkt=item)
            logger.info("Short Open with update time:{0},filled price:{1}, filled lot:{2}------".format(_update_time,
                                                                                                        _fill_price,
                                                                                                        _fill_lot))
            if _fill_lot > 0:
                # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                account.add_transaction(
                    [idx,
                     instrument_id,
                     _text_lst[SHORT_OPEN],
                     _last,  # 最新价格
                     _fill_price,  # 成交价格
                     _fill_lot,  # 成交数量
                     _fill_price,  # 持仓成本
                     fee_lst[0] * _fill_lot,  # 费用
                     _update_time.split(' ')[-1],  # 开仓时间
                     _update_time.split(' ')[-1],  # 平仓时间
                     0.0,  # 持有时间
                     0.0])  # 持仓收益
                position.update_position(instrument_id=instrument_id, long_short=SHORT, price=_fill_price,
                                         timestamp=_update_time,
                                         vol=_fill_lot, order_type=SHORT_OPEN)
                account.update_fee(fee_lst[0] * _fill_lot)

        elif _signal.signal_type == LONG_CLOSE:
            logger.info("Long Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, define.LONG)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[SHORT, _signal.price, _signal.vol], mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    curr_return = (((_fill_price - _pos[1]) * _mul_num) - fee_lst[1]) * _fill_lot
                    total_return += curr_return
                    # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                    account.add_transaction(
                        [idx,
                         instrument_id,
                         _text_lst[LONG_CLOSE],
                         _last,  # 最新价格
                         _fill_price,  # 成交价格
                         _fill_lot,  # 成交数量
                         _pos[1],  # 持仓成本
                         fee_lst[1] * _fill_lot,  # 费用
                         _pos[2].split(' ')[-1],  # 开仓时间
                         _update_time.split(' ')[-1],  # 平仓时间
                         holding_time,  # 持有时间
                         curr_return])  # 持有收益
                    position.update_position(instrument_id=instrument_id, long_short=SHORT, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=LONG_CLOSE)
                    account.update_fee(fee_lst[1] * _fill_lot)
        elif _signal.signal_type == SHORT_CLOSE:
            logger.info("Short Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, SHORT)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[LONG, _signal.price, _signal.vol], mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    curr_return = ((_pos[1] - _fill_price) * _mul_num - fee_lst[1]) * _fill_lot
                    total_return += curr_return
                    # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                    account.add_transaction(
                        [idx,
                         instrument_id,
                         _text_lst[SHORT_CLOSE],
                         _last,  # 最新价格
                         _fill_price,  # 成交价格
                         _fill_lot,  # 成交数量
                         _pos[1],  # 持仓成本
                         fee_lst[1] * _fill_lot,  # 费用
                         _pos[2].split(' ')[-1],  # 开仓时间
                         _update_time,  # 平仓时间
                         holding_time,  # 持有时间
                         curr_return])  # 收益
                    position.update_position(instrument_id=instrument_id, long_short=LONG, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=SHORT_CLOSE)
                    account.update_fee(fee_lst[1] * _fill_lot)
        else:  # NO_SIGNAL
            pass

    logger.info("Factor/tick error num:{0} out of:{1}".format(ab_cnt, total_tick_num))
    _pos = position.get_position(instrument_id)
    total_return_risk = total_return
    total_risk = 0.0

    # 收盘强平
    if _pos:
        _tmp_pos = deepcopy(_pos)
        for item in _tmp_pos:
            if item[0] == LONG:
                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick down
                _return = ((close_price - define.TICK - item[1]) * _mul_num - fee_lst[1]) * item[-1]
                total_return += _return
                total_risk += item[1] * item[-1]
                logger.info('final long close with return:{0},total return after:{1} for trade_date:{2}'.format(_return,
                                                                                                                total_return,
                                                                                                                trade_date))
                # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                account.add_transaction(
                    [idx,
                     instrument_id,
                     _text_lst[define.LONG_CLOSE],
                     close_price,  # 最新价格/收盘价格
                     close_price - define.TICK,  # 成交价格
                     item[-1],  # 成交数量
                     item[1],  # 持仓成本
                     fee_lst[1] * item[-1],  # 费用
                     item[2],  # 开仓时间
                     close_timestamp,  # 平仓时间
                     holding_time,  # 持有时间
                     _return])  # 持有收益
                account.update_fee(fee_lst[1] * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.LONG_CLOSE)
            else:

                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick up, tick hardcode
                _return = ((item[1] - close_price - define.TICK) * _mul_num - fee) * item[-1]
                total_risk += item[1] * item[-1]
                total_return += _return
                logger.info('final short close with return:{0},total return after:{1}'.format(_return, total_return))
                # 索引，标的，成交类别，最新价格，成交价格，成交手数，持仓成本，费用，开仓时间，平仓时间，持有时间，收益
                account.add_transaction(
                    [idx,
                     instrument_id,
                     _text_lst[define.SHORT_CLOSE],
                     close_price,  # 最新价格
                     close_price + define.TICK,  # 成交价格
                     item[-1],  # 成交数量
                     item[1],  # 持仓成本
                     fee_lst[1] * item[-1],  # 费用
                     item[2],  # 开仓时间
                     close_timestamp,  # 平仓时间
                     holding_time,  # 持有时间
                     _return])  # 持有收益
                account.update_fee(fee_lst[1] * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.LONG, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.SHORT_CLOSE)
    # 结果统计
    long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open = 0, 0, 0, 0, 0, 0
    total_holding_time = 0.0
    max_holding_time = -np.inf
    min_holding_time = np.inf
    for item in account.transaction:
        if item[2] == 'long_open':
            long_open += 1
        elif item[2] == 'short_open':
            short_open += 1
        elif item[2] == 'long_close':
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:  # short close
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1
    average_holding_time = total_holding_time / (long_open + short_open) if long_open + short_open > 0 else 0.0
    precision = (correct_long_open + correct_short_open) / (
            long_open + short_open) if long_open + short_open > 0 else 0.0
    print("******************back test models for date:{0}*********************".format(trade_date))
    print('trade date', trade_date)
    print(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    print('total return:', total_return)
    print('total fee:', account.fee)
    print('precision:', precision)
    print("average_holding_time:", average_holding_time)
    print("max_holding_time:", max_holding_time)
    print("min_holding_time:", min_holding_time)
    print("******************back test models for date:{0}*********************".format(trade_date))

    _backtest_path = os.path.join(_base_dir, 'data\\backtest\\backtest_{0}.csv'.format(product_id))
    try:
        result_df = pd.read_csv(_backtest_path)
    except Exception as ex:
        result_df = pd.DataFrame(
            {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return_final': [],
             'total_return_unclose': [],
             'total_fee': [],
             'unclosed_value': [], 'precision': [], 'long_open': [], 'short_open': [],
             'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
             'average_holding_time': [], 'max_holding_time': [], 'min_holding_time': []
             })
    new_df = pd.DataFrame([{'trade_date': trade_date, 'product_id': product_id, 'instrument_id': instrument_id,
                            'total_return_final': total_return, 'total_return_unclose': total_return_risk,
                            'total_fee': account.fee,
                            'unclosed_value': total_risk, 'precision': precision,
                            'long_open': long_open, 'short_open': short_open, 'correct_long_open': correct_long_open,
                            'wrong_long_open': wrong_long_open, 'correct_short_open': correct_short_open,
                            'wrong_short_open': wrong_short_open, 'average_holding_time': average_holding_time,
                            'max_holding_time': max_holding_time, 'min_holding_time': min_holding_time
                            }])
    result_df = pd.concat([result_df, new_df])
    result_df.to_csv(_backtest_path, index=False)

    # 画图，开平仓信号，问题处理一下
    # if plot_mkt:
    #     _idx_lst = list(range(len(factor.last_price.get_lst())))
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.last_price[define.PLT_START:define.PLT_END])
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.vwap[define.PLT_START:define.PLT_END])
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.turning[define.PLT_START:define.PLT_END])
    #     logger.info(np.array(factor.last_price).std())
    #     ax1.grid(True)
    #     ax1.set_title('{0}_{1}'.format(instrument_id, trade_date))
    #     xtick_labels = [item[:-3] for item in factor.update_time]
    #     ax1.set_xticks(_idx_lst[::3600])
    #     min_lst = []
    #     ax1.set_xticklabels(xtick_labels[::3600])
    #     # ax1.set_xticks(factor.update_time[::3600])
    #     # ax1.set_xticks(x_idx, xtick_labels, rotation=60, FontSize=6)
    #     for item in account.transaction:
    #         _t_lst = ['lo', 'lc', 'so', 'sc']
    #         ax1.text(item[0], item[3], s='{0}'.format(item[2]))
    #
    #     ax2 = ax1.twinx()
    #     ax2.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.trend_short[define.PLT_START:define.PLT_END], 'r')
    #     plt.show()
    ret = (total_return, account.fee, precision, account.transaction)
    return ret


if __name__ == '__main__':
    # backtesting()
    import pprint

    print(backtest_quick(data_fetcher=data_fetcher, product_id='rb', trade_date='2021-03-08'))
