#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 14:34
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : ClfSignal.py

import numpy as np
import pandas as pd
from strategy.Signal import Signal
from strategy.Signal import SignalField
from strategy.trade_rules import ma_rule
from strategy.trade_rules import dual_thrust
from strategy.trade_rules import spread_rule
# from strategy.trade_rules import clf_rule
from strategy.trade_rules import reg_rule
import utils.define as define
import utils.utils as utils
from backtester.Factor import Factor
from backtester.Position import Position
import os


def _get_pred(label_prob):
    if label_prob[0] == label_prob[-1]:
        return 0
    _max_idx = label_prob.index(max(label_prob))
    return [-1, 0, 1][_max_idx]


def get_final_pred(lst_pred):
    n_num = len(lst_pred)
    if sum(lst_pred) >= n_num / 2:
        return 1
    elif sum(lst_pred) <= -n_num / 2:
        return -1
    else:
        return 0


class ClfSignal(Signal):
    def __init__(self, factor, position, instrument_id, trade_date, product_id):
        super().__init__(factor, position)
        # _reg_param_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
        #                                define.TICK_MODEL_DIR,
        #                                'reg_params_{0}.json'.format(instrument_id))
        # self.reg_params = utils.load_json_file(_reg_param_name)
        self.reg_params = {}

    def __call__(self, *args, **kwargs):
        params = kwargs.get('params')  # is options
        # _k = params.get('tick')[2].split()[1].split('.')[0]
        # _update_time = params.get('tick')[2]
        curr_factor = params.get('factor')
        _long, _short, long_price, short_price = 0, 0, 0.0, 0.0
        instrument_id = params.get('instrument_id')
        # start_tick = int(params.get('start_tick')) or 2
        stop_profit = float(params.get('stop_profit')) or 5.0
        stop_loss = float(params.get('stop_loss')) or 20.0
        multiplier = int(params.get('multiplier')) or 10
        risk_duration = int(params.get('risk_duration')) or 10
        open_fee = float(params.get('open_fee')) or 1.51
        close_to_fee = float(params.get('close_t0_fee')) or 0.0

        fee = (open_fee + close_to_fee) / multiplier
        fee = 0  # FIXME remove the hardcode, consider fee in strop profit and loss
        _position = self.position.get_position(instrument_id)
        _sec = int(self.factor.update_time[-1].split()[-1].split('.')[0][-2:])
        _min = int(self.factor.update_time[-1].split()[-1].split('.')[0][-5:-3])
        # risk check
        _check_stop_profit_loss = (_min * 60 + _sec) % risk_duration
        # print("min:{0}, sec:{1}, check flag:{2}".format(_min, _sec, _check_stop_profit_loss))
        if _position:
            for item in _position:
                if item[0] == define.LONG:
                    long_price = item[1]
                    _long += item[3]
                elif item[0] == define.SHORT:
                    short_price = item[1]
                    _short += item[3]
        # order data in simulation
        _order_data_field = SignalField()
        lst_pred_label = []
        signal_lst = params.get('signal_names').split(',')
        long_signal_benchmark = len(signal_lst) * (params.get('long_score_ratio') or 0.5)
        short_signal_benchmark = len(signal_lst) * (params.get('short_score_ratio') or 0.5)

        total_score = 0.0
        signal_map = {
            'ma': ma_rule(self.factor, params=params.get('ma')),
            # 'reg': reg_rule(self.factor, params=self.reg_params),
            # 'spread': spread_rule(self.factor),
            'dual': dual_thrust(self.factor,
                                params={'daily_cache': params.get('daily_cache'), 'dual': params.get('dual')})

        }

        for sig_name in signal_lst:
            _score = signal_map.get(sig_name)
            total_score += _score
            if _score:
                print("get score {0} for signal {1} for time:{2}".format(_score, sig_name, _update_time))

        pred_label = 0
        if total_score >= long_signal_benchmark:
            pred_label = 1
        elif total_score <= -short_signal_benchmark:
            pred_label = -1
        _vol_limit = params.get('vol_limit')
        _ins_vol = _long + _short
        # print('pred_label:', pred_label)
        if pred_label == 1:  # long signal
            print("Get Long Signal=>", pred_label, 'long vol=>', _long, 'cur vol=>', _ins_vol, 'vol limit=>',
                  _vol_limit,
                  'last price=>', self.factor.last_price[-1], 'long price=>', long_price, 'short price=>', short_price,
                  'stop profit=>', stop_profit, 'fee=>', fee)
            if _long and _ins_vol < _vol_limit:  # 多仓未达风险则继续开多
                print("more long open")
                _order_data_field.direction = define.LONG
                _order_data_field.signal_type = define.LONG_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            elif _short and self.factor.last_price[-1] < short_price - stop_profit:  # 空仓则平空, 佣金暂不在此考虑
                print('short close')
                _order_data_field.direction = define.LONG
                _order_data_field.signal_type = define.SHORT_CLOSE
                _order_data_field.vol = _short
                _order_data_field.price = 0
                return _order_data_field
            elif not _long and _ins_vol < _vol_limit:  # 未持多仓且未达风险则开多仓
                print('no pos, long open')
                _order_data_field.direction = define.LONG
                _order_data_field.signal_type = define.LONG_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            else:
                print('no action for long signal')
                pass
        elif pred_label == -1:  # short signal
            print("Get Short Signal=>", pred_label, 'short pos=>', _short, 'cur pos=>', _ins_vol, 'vol limit=>',
                  _vol_limit, 'last price=>', self.factor.last_price[-1], 'long price=>', long_price,
                  'short price', short_price, 'stop profit=>', stop_profit, 'fee=>', fee)
            if _short and _ins_vol < _vol_limit:  # 空仓未达风险则继续开空
                print('more short open')
                _order_data_field.direction = define.SHORT
                _order_data_field.signal_type = define.SHORT_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            elif _long and self.factor.last_price[-1] > long_price + stop_profit:  # 多仓则平多，佣金暂不在此考虑
                print('long close')
                _order_data_field.direction = define.SHORT
                _order_data_field.signal_type = define.LONG_CLOSE
                _order_data_field.vol = _long
                _order_data_field.price = 0
                return _order_data_field
            elif not _short and _ins_vol < _vol_limit:  # 未持空仓且未达风险则开空
                print('no short, short open')
                _order_data_field.direction = define.SHORT
                _order_data_field.signal_type = define.SHORT_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            else:
                print("no action for short signal")
        else:  # no long short signal, check stop profit and loss
            pass
        if _check_stop_profit_loss == 0:
            if _position:
                for item in _position:
                    if item[0] == define.LONG:
                        # print('check close long')
                        _is_close = (self.factor.last_price[-1] > item[1] + stop_profit + fee) or (
                                self.factor.last_price[-1] < item[1] - stop_loss - fee)  # stop profit or stop  loss
                        if _is_close:
                            print('long stop profit/loss, last=>', self.factor.last_price[-1], 'open price=>', item[1])
                            _order_data_field.direction = define.SHORT
                            _order_data_field.signal_type = define.LONG_CLOSE
                            _order_data_field.vol = item[-1]
                            _order_data_field.price = 0
                            return _order_data_field
                    if item[0] == define.SHORT:
                        # print('check close short')
                        _is_close = (self.factor.last_price[-1] < item[1] - stop_profit - fee) or (
                                self.factor.last_price[-1] > item[1] + stop_loss + fee)
                        if _is_close:
                            print('short stop profit/loss, last=>', self.factor.last_price[-1], 'open price=>', item[1])
                            _order_data_field.direction = define.LONG
                            _order_data_field.signal_type = define.SHORT_CLOSE
                            _order_data_field.vol = item[-1]
                            _order_data_field.price = 0
                            return _order_data_field
        _order_data_field.signal_type = define.NO_SIGNAL
        return _order_data_field
