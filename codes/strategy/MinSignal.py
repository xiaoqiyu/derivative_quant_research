#!/user/bin/env python
# coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : MinSignal.py
#@time   : 2023-04-07 00:11:25
'''

import os
import sys

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)

import pandas as pd
from codes.strategy.Signal import Signal
from codes.strategy.Signal import SignalField
from codes.utils.define import *


class MinSignal(Signal):
    def __init__(self, factor, position, instrument_id, trade_date, product_id):
        super().__init__(factor, position)
        self.reg_params = {}
        _signal_path = os.path.join(_base_dir, 'data\models\\tsmodels\\signal_{0}.csv'.format(product_id))
        df = pd.read_csv(_signal_path)
        self.signal_map = dict(zip(df['datetime'], df['signal']))

    def __call__(self, *args, **kwargs):
        params = kwargs.get('params')  # is options
        # _k = params.get('tick')[2].split()[1].split('.')[0]
        # _update_time = params.get('tick')[2]
        curr_factor = params.get('factor')
        _long, _short, long_price, short_price = 0, 0, 0.0, 0.0
        instrument_id = params.get('instrument_id')
        # start_tick = int(params.get('start_tick')) or 2
        stop_profit = float(params.get('stop_profit') or 10.0)
        stop_loss = float(params.get('stop_loss') or 10.0)
        multiplier = int(params.get('multiplier') or 10)
        risk_duration = int(params.get('risk_duration') or 30)
        open_fee = float(params.get('open_fee') or 1.51)
        close_to_fee = float(params.get('close_t0_fee') or 0.0)
        _vol_limit = params.get('vol_limit') or 5

        fee = (open_fee + close_to_fee) / multiplier
        # TODO fee 全部按照价格看
        fee = 1  # FIXME remove the hardcode, consider fee in strop profit and loss
        _position = self.position.get_position(instrument_id)
        if _position:
            for item in _position:
                if item[0] == LONG:
                    long_price = item[1]
                    _long += item[3]
                elif item[0] == SHORT:
                    short_price = item[1]
                    _short += item[3]

        _order_data_field = SignalField()

        _factor = self.factor.get_factor()
        _update_time = _factor.get('update_time')
        _last_price = _factor.get('last_price')
        dt_key = _update_time.split('.')[0].replace(' ', '_')
        pred_label = self.signal_map.get(dt_key)
        _ins_vol = _long + _short

        _sec = int(self.factor.update_time[-1].split()[-1].split('.')[0][-2:])
        _min = int(self.factor.update_time[-1].split()[-1].split('.')[0][-5:-3])
        # risk check
        _check_stop_profit_loss = (_min * 60 + _sec) % risk_duration
        if pred_label == 2:  # long signal
            print("Get Long Signal=>", pred_label, 'long vol=>', _long, 'cur vol=>', _ins_vol, 'vol limit=>',
                  _vol_limit,
                  'last price=>', self.factor.last_price.get(-1), 'long price=>', long_price, 'short price=>', short_price,
                  'stop profit=>', stop_profit, 'fee=>', fee)
            if _long and _ins_vol < _vol_limit:  # 多仓未达风险则继续开多
                print("more long open")
                _order_data_field.direction = LONG
                _order_data_field.signal_type = LONG_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            elif _short and self.factor.last_price.get(-1) < short_price - stop_profit:  # 空仓则平空, 佣金暂不在此考虑
                print('short close')
                _order_data_field.direction = LONG
                _order_data_field.signal_type = SHORT_CLOSE
                _order_data_field.vol = _short
                _order_data_field.price = 0
                return _order_data_field
            elif not _long and _ins_vol < _vol_limit:  # 未持多仓且未达风险则开多仓
                print('no pos, long open')
                _order_data_field.direction = LONG
                _order_data_field.signal_type = LONG_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            else:
                print('no action for long signal')
                pass
        elif pred_label == 0:  # short signal
            print("Get Short Signal=>", pred_label, 'short pos=>', _short, 'cur pos=>', _ins_vol, 'vol limit=>',
                  _vol_limit, 'last price=>', self.factor.last_price.get(-1), 'long price=>', long_price,
                  'short price', short_price, 'stop profit=>', stop_profit, 'fee=>', fee)
            if _short and _ins_vol < _vol_limit:  # 空仓未达风险则继续开空
                print('more short open')
                _order_data_field.direction = SHORT
                _order_data_field.signal_type = SHORT_OPEN
                _order_data_field.vol = _vol_limit - _ins_vol
                _order_data_field.price = 0  # market order
                return _order_data_field
            elif _long and self.factor.last_price.get(-1) > long_price + stop_profit:  # 多仓则平多，佣金暂不在此考虑
                print('long close')
                _order_data_field.direction = SHORT
                _order_data_field.signal_type = LONG_CLOSE
                _order_data_field.vol = _long
                _order_data_field.price = 0
                return _order_data_field
            elif not _short and _ins_vol < _vol_limit:  # 未持空仓且未达风险则开空
                print('no short, short open')
                _order_data_field.direction = SHORT
                _order_data_field.signal_type = SHORT_OPEN
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
                    if item[0] == LONG:
                        # print('check close long')
                        _is_close = (self.factor.last_price.get(-1) > item[1] + stop_profit + fee) or (
                                self.factor.last_price.get(-1) < item[1] - stop_loss - fee)  # stop profit or stop  loss
                        if _is_close:
                            print('long stop profit/loss, last=>', self.factor.last_price.get(-1), 'open price=>', item[1])
                            _order_data_field.direction = SHORT
                            _order_data_field.signal_type = LONG_CLOSE
                            _order_data_field.vol = item[-1]
                            _order_data_field.price = 0
                            return _order_data_field
                    if item[0] == SHORT:
                        # print('check close short')
                        _is_close = (self.factor.last_price.get(-1) < item[1] - stop_profit - fee) or (
                                self.factor.last_price.get(-1) > item[1] + stop_loss + fee)
                        if _is_close:
                            print('short stop profit/loss, last=>', self.factor.last_price.get(-1), 'open price=>', item[1])
                            _order_data_field.direction = LONG
                            _order_data_field.signal_type = SHORT_CLOSE
                            _order_data_field.vol = item[-1]
                            _order_data_field.price = 0
                            return _order_data_field
        _order_data_field.signal_type = NO_SIGNAL
        return _order_data_field