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
        stop_profit = float(params.get('stop_profit') or 5.0)
        stop_loss = float(params.get('stop_loss') or 20.0)
        multiplier = int(params.get('multiplier') or 10)
        risk_duration = int(params.get('risk_duration') or 10)
        open_fee = float(params.get('open_fee') or 1.51)
        close_to_fee = float(params.get('close_t0_fee') or 0.0)

        fee = (open_fee + close_to_fee) / multiplier
        fee = 0  # FIXME remove the hardcode, consider fee in strop profit and loss
        _position = self.position.get_position(instrument_id)

        _order_data_field = SignalField()

        _factor = self.factor.get_factor()
        _update_time = _factor.get('update_time')
        _last_price = _factor.get('last_price')
        dt_key = _update_time.split('.')[0].replace(' ', '_')
        pred = self.signal_map.get(dt_key)
        if pred == 2:
            print("more long open")
            _order_data_field.direction = LONG
            _order_data_field.signal_type = LONG_OPEN
            _order_data_field.vol = 5
            _order_data_field.price = 0  # market order
            return _order_data_field
        elif pred == 0:
            print('more short open')
            _order_data_field.direction = SHORT
            _order_data_field.signal_type = SHORT_OPEN
            _order_data_field.vol = 5
            _order_data_field.price = 0  # market order
            return _order_data_field
        else:
            pass
        return _order_data_field
