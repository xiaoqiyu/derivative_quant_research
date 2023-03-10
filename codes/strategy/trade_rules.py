#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:16
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : trade_rules.py

from codes.backtester import Factor
import numpy as np


def ma_var_1(factor: Factor = None, params: dict = {}) -> int:
    curr_short = factor.ma10[-1]
    last_short = factor.ma10[-2]

    curr_long = factor.ma20[-1]
    last_long = factor.ma20[-2]

    if last_short < last_long and curr_short > curr_long:
        return 1
    if last_short > last_long and curr_short < curr_long:
        return -1
    return 0


def ma_var_2(factor: Factor = None, params: dict = {}) -> int:
    curr_short = factor.ma10[-1]
    last_short = factor.ma10[-2]

    curr_long = factor.ma60[-1]
    last_long = factor.ma60[-2]

    if last_short < last_long and curr_short > curr_long:
        return 1
    if last_short > last_long and curr_short < curr_long:
        return -1
    return 0


def ma_var_3(factor: Factor = None, params: dict = {}) -> int:
    curr_short = factor.ma20[-1]
    last_short = factor.ma20[-2]

    curr_long = factor.ma120[-1]
    last_long = factor.ma120[-2]

    if last_short < last_long and curr_short > curr_long:
        return 1
    if last_short > last_long and curr_short < curr_long:
        return -1
    return 0


def ma_var_4(factor: Factor = None, params: dict = {}) -> int:
    if len(factor.vwap_ls_diff) > 2:
        try:
            curr_ls_diff = factor.vwap_ls_diff[-1]
            last_ls_diff = factor.vwap_ls_diff[-2]
        except Exception as ex:
            print(ex)
        if last_ls_diff < 0 and curr_ls_diff > 0:
            print('ma long signal:', last_ls_diff, curr_ls_diff, factor.update_time[-1])
            return 1
        elif last_ls_diff > 0 and curr_ls_diff < 0:
            print('ma short signal', last_ls_diff, curr_ls_diff, factor.update_time[-1])
            return -1
        else:
            return 0
    return 0


def clf_var_1(factor: Factor = None, params: dict = {}) -> int:
    _coef = params.get('coef')
    _intercept = params.get('intercept')
    label_prob = []
    curr_factor = factor.get_factor()
    for idx, item in enumerate(_coef):
        _score = 0
        for k, v in item.items():
            _val = curr_factor.get(k) or 0.
            if _val != _val:
                return 0
            _score += v * _val
        _score += _intercept[idx]
        label_prob.append(_score)
    if label_prob[0] > label_prob[-1]:
        return -1
    elif label_prob[-1] > label_prob[0]:
        return 1
    return 0


def reg_var_1(factor: Factor = None, params: dict = {}) -> int:
    _coef = params.get('coef') or {}
    _intercept = params.get('intercept') or 0.0
    curr_factor = factor.get_factor()
    _score = 0
    for k, v in _coef.items():
        _val = curr_factor.get(k) or np.nan
        if _val != _val:
            return 0
        _score += v * _val
        _score += _intercept
    if _score > 0.0008:
        return 1
    elif _score < -0.0008:
        return -1
    else:
        return 0
    return 0


def spread_rule(factor: Factor = None, params: dict = {}) -> int:
    if factor.spread[-1] >= 1:
        if factor.mid_price[-1] - factor.last_price[-1] > 0.5:
            return 1
        elif factor.mid_price[-1] - factor.mid_price[-2] < -0.5:
            return -1
        else:
            return 0
    else:
        return 0


def ma_rule(factor: Factor = None, params: dict = {}) -> int:
    # _lst1 = ma_var_1(factor)
    # _lst2 = ma_var_3(factor)
    # _lst3 = ma_var_3(factor)
    # if _lst1[0] + _lst2[0] + _lst3[0] >= 1:
    #     return -1
    # elif _lst1[-1] + _lst2[-1] + _lst3[-1] >= 1:
    #     return 1
    # else:
    #     return 0
    return ma_var_4(factor)


def dual_thrust(factor: Factor = None, params: dict = {}) -> int:
    daily_cache = params.get('daily_cache')
    _dual = params.get('dual')
    _range = max(daily_cache.get('hh') - daily_cache.get('lc'), daily_cache.get('hc') - daily_cache.get('ll'))
    _k1 = _dual.get('k1')
    _k2 = _dual.get('k2')
    _open_price = daily_cache.get('openPrice')
    if factor.last_price[-1] > _open_price + _k1 * _range and factor.last_price[-2] < _open_price + _k1 * _range:
        print("-----------last-2:{0}, last-1:{1}, benchmark:{2}, update_time:{3}".format(factor.last_price[-2],
                                                                                         factor.last_price[-1],
                                                                                         _open_price + _k1 * _range,
                                                                                         factor.update_time[-1]))
        return 1
    elif factor.last_price[-1] < _open_price - _k2 * _range and factor.last_price[-2] > _open_price - _k2 * _range:
        print("-----------last-2:{0}, last-1:{1}, benchmark:{2}, update_time:{3}".format(factor.last_price[-2],
                                                                                         factor.last_price[-1],
                                                                                         _open_price - _k2 * _range,
                                                                                         factor.update_time[-1]))
        return -1
    return 0


# def clf_rule(factor: Factor = None, params: dict = {}) -> list:
#     return clf_var_1(factor, params)

def revert_rule(factor: Factor = None, params: dict = {}) -> int:
    return 1


def reg_rule(factor: Factor = None, params: dict = {}) -> list:
    return reg_var_1(factor, params)


if __name__ == "__main__":
    ma_rule()
