#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 14:23
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : factor_calculation.py


import os
import torch
import copy
from torch.utils.data import DataLoader
import sys
import numpy as np
import pandas as pd
# import talib as ta
import uqer
from uqer import DataAPI
from datetime import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.research.data_process.data_fetcher import DataFetcher
from codes.utils.helper import timeit
from codes.utils.define import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_oir(bid_price: list = [], bid_vol: list = [], ask_price: list = [], ask_vol: list = [],
            n_rows: int = 0) -> tuple:
    """
    calculate order imbalance features
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


@timeit
def calculate_raw_features(data_fetcher: DataFetcher = None, product_id: str = '', instrument_id: str = '',
                           start_date: str = '', end_date: str = ''):
    data_fetcher.get_instrument_mkt(product_ids=[product_id], start_date=start_date, end_date=end_date)
    _instruments = list(set(data_fetcher.instrument_cache['ticker']))
    data_fetcher.get_instrument_contract(instrument_ids=_instruments, product_ids=[product_id])

    tick_mkt = data_fetcher.load_tick_data(start_date=start_date, end_date=end_date, instrument_ids=_instruments,
                                           main_con_flag=1, filter_start_time=None, filter_end_time=None,
                                           if_filter=True)
    tick_mkt = tick_mkt.set_index('InstrumentID').join(
        data_fetcher.contract_cache[['ticker', 'contMultNum']].set_index('ticker')).reset_index()
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
    # FIXME ta lib not work for python 3.11,should build the lib
    # dif, dea, macd = ta.MACD(tick_mkt['LastPrice'], fastperiod=12, slowperiod=26, signalperiod=9)

    tick_mkt['open_close_ratio'] = open_close_ratio
    tick_mkt['price_spread'] = (tick_mkt['BidPrice1'] - tick_mkt['AskPrice1']) / (
            tick_mkt['BidPrice1'] + tick_mkt['AskPrice1'] / 2)
    tick_mkt['buy_sell_spread'] = abs(tick_mkt['BidPrice1'] - tick_mkt['AskPrice1'])
    tick_mkt['oi'] = lst_oi
    tick_mkt['oir'] = lst_oir
    tick_mkt['aoi'] = lst_aoi
    tick_mkt['slope'] = _lst_slope
    tick_mkt['cos'] = _lst_cos
    # FIXME ta lib not work for python 3.11,should build the lib
    # tick_mkt['macd'] = macd
    # tick_mkt['dif'] = dif
    # tick_mkt['dea'] = dea
    tick_mkt['bs_tag'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: 1 if list(x)[-1] > list(x)[0] else -1)
    tick_mkt['bs_vol'] = tick_mkt['bs_tag'] * tick_mkt['Volume']
    return tick_mkt


def cal_derived_featuers():
    # calculate the derived features on cross-section level and time series level(by lag windows)
    pass


@timeit
def gen_train_test_features(data_fetcher: DataFetcher = None, param_model=None, product_id: str = '', freq: str = '60S',
                            missing_threshold: int = 20,
                            train_start_date: str = '',
                            train_end_date: str = '2021-07-05',
                            test_start_date: str = '',
                            test_end_date: str = ''):
    df = calculate_raw_features(data_fetcher=data_fetcher, product_id=product_id, start_date=train_start_date,
                                end_date=test_end_date)

    # TODO read feature from cache for testing
    # df = pd.read_csv(os.path.abspath(os.path.join(__file__, "../feature_sample.csv")))

    # FIXME hardcode for testing features
    df = df[TEST_FEATURES]
    df.columns = RENAME_FEATURES
    df = df.dropna()

    train_end_dt_str = '{0} 14:00:00'.format(train_end_date)
    train_df = df[df.UpdateTime <= train_end_dt_str]
    test_df = df[df.UpdateTime > train_end_dt_str]

    # TODO add logic to drop abnormal data,e.g. mean +- 3std
    std_model = param_model.std_model or StandardScaler()
    transform_features = copy.deepcopy(RENAME_FEATURES)
    transform_features.remove(DT_COL_NAME)
    transform_features.remove(LABEL)

    if train_df.shape[0] > 0:
        std_train_df = pd.DataFrame(std_model.fit_transform(train_df[transform_features]), columns=transform_features)
        std_train_df[DT_COL_NAME] = train_df[DT_COL_NAME]
        std_train_df[LABEL] = train_df[LABEL]
        del train_df
        std_train_df = std_train_df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        train_data_loader, bins, _ = get_dataloader(df=std_train_df, freq=freq,
                                                    missing_threshold=missing_threshold,
                                                    dt_col_name=DT_COL_NAME,
                                                    if_filtered=True, if_train=True, bins=None)
        param_model.update_model(std_model=std_model, bins=bins)
        param_model.dump_model()
    else:
        train_data_loader = None
        bins = param_model.bins

    if test_df.shape[0] > 0:
        std_test_df = pd.DataFrame(std_model.transform(test_df[transform_features]), columns=transform_features)
        std_test_df[DT_COL_NAME] = list(test_df[DT_COL_NAME])
        std_test_df[LABEL] = list(test_df[LABEL])
        del test_df
        std_test_df = std_test_df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        test_data_loader, bins, _ = get_dataloader(df=std_test_df, freq=freq,
                                                   missing_threshold=missing_threshold,
                                                   dt_col_name=DT_COL_NAME,
                                                   if_filtered=True, if_train=False, bins=bins)
    else:
        test_data_loader = None
    return train_data_loader, test_data_loader


@timeit
def gen_predict_feature_dataset(data_fetcher: DataFetcher = None, param_model=None, product_id: str = '',
                                freq: str = '60S',
                                missing_threshold: int = 20,
                                start_date: str = '',
                                end_date: str = '2021-07-05',
                                ):
    df = calculate_raw_features(data_fetcher=data_fetcher, product_id=product_id, start_date=start_date,
                                end_date=end_date)
    # FIXME hardcode for testing features
    df = df[TEST_FEATURES]
    df.columns = RENAME_FEATURES
    df = df.dropna()

    # train_end_dt_str = '{0} 14:00:00'.format(train_end_date)
    # train_df = df[df.UpdateTime <= train_end_dt_str]
    # test_df = df[df.UpdateTime > train_end_dt_str]

    std_model = param_model.std_model
    transform_features = copy.deepcopy(RENAME_FEATURES)
    dt_col = df[DT_COL_NAME]
    transform_features.remove(DT_COL_NAME)
    transform_features.remove(LABEL)

    if df.shape[0] > 0:
        std_df = pd.DataFrame(std_model.transform(df[transform_features]), columns=transform_features)
        std_df[DT_COL_NAME] = list(df[DT_COL_NAME])
        std_df[LABEL] = list(df[LABEL])
        del df
        std_df = std_df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        data_loader, bins, dt_col = get_dataloader(df=std_df, freq=freq, missing_threshold=missing_threshold,
                                                   dt_col_name=DT_COL_NAME,
                                                   if_filtered=True, if_train=False, bins=param_model.bins)
    else:
        data_loader = None
    return data_loader, bins, dt_col


@timeit
def get_dataloader(df, freq: str = '60S', missing_threshold: int = 20, dt_col_name: str = 'UpdateTime',
                   if_filtered: bool = True, if_train: bool = True, bins=None):
    '''
    pass in dataframe, and return dataloader,padding to SEQUENCE
    :param df:
    :param freq:
    :param missing_threshold:
    :param dt_col_name:
    :param if_filtered:
    :param if_train:
    :return:
    '''
    cols = list(df.columns)
    if dt_col_name not in cols or LABEL not in cols:
        raise ValueError("passed features missing datetime col or label column:{0}".format(cols))
    df.index = pd.to_datetime(df['UpdateTime'])
    # df.index = pd.to_datetime(df['time'])
    # # TODO 1. 需要再filter 掉10：15-10：30；2.不按日处理的话，跨日的第一个sample需要去掉？不然就变成前一个交易日的收盘前的行情预测下一个交易日（夜盘）的开盘走势
    # if if_filtered:
    #     df = pd.concat(
    #         [df.loc[time(9, 30): time(11, 30)], df.loc[time(13, 30): time(15, 0)], df.loc[time(21, 0):time(23, 0)]])

    df_label = df[[LABEL]].resample(freq, label='left').sum().replace(0.0, np.nan).dropna()

    if if_train:
        df_label[LABEL], bins = pd.qcut(df_label['label'], q=3, labels=[0, 1, 2], retbins=True)
    else:
        df_label[LABEL], bins = pd.cut(df_label['label'], bins=bins, labels=[0, 1, 2], retbins=True)

    selected_cols = copy.deepcopy(cols)
    selected_cols.remove(LABEL)

    df = df[selected_cols].join(df_label)
    dt_col_val = list(df[dt_col_name])

    selected_cols.remove(dt_col_name)
    selected_cols.append(LABEL)
    df = df[selected_cols]

    notnull_labels = [idx for idx, item in enumerate(list(df[LABEL].notnull())) if item]
    img = []
    _len = len(notnull_labels)

    _featuers = list(df.values)
    _index = list(df.index)
    ab_cnt = 0
    ret_dt = []
    for i in range(1, _len):
        left, right = notnull_labels[i - 1], notnull_labels[i]
        s_len = right - left
        if SEQUENCE - s_len > missing_threshold:
            continue
        _sample = _featuers[left:right]
        if len(_sample) > SEQUENCE:
            ab_cnt += 1
            continue
        for idx in range(SEQUENCE - s_len):
            # TODO padding here, but when calculate the loss, we dnt handle all the time step, and missing timestaop
            # TODO  is controled, so here not calling pack_padded_sequence first
            _row = [0.0] * INPUT_SIZE
            _row.append(np.nan)
            _sample.append(_row)
        img.append(_sample)
        ret_dt.append(dt_col_val[left])
    _tensor = torch.Tensor(np.array(img)).to(device)
    _shuffle = True if if_train else False
    _data_loader = DataLoader(_tensor, batch_size=BATCH_SIZE, shuffle=_shuffle, drop_last=True)
    return _data_loader, bins, ret_dt


if __name__ == '__main__':
    product_id = 'rb'
    start_date = '2021-07-05'
    end_date = '2021-07-09'
    feature_path = os.path.join(_base_dir,
                                'data\\features\\features_{0}_{1}_{2}.csv'.format(product_id, start_date, end_date))

    uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
    data_fetch = DataFetcher(uqer_client)
    df_features = calculate_raw_features(data_fetcher=data_fetch, product_id=product_id, start_date=start_date,
                                         end_date=end_date)
    df_features.to_csv(feature_path, index=False)

    train_data_loader, test_data_loader = gen_train_test_features(data_fetcher=data_fetch, product_id='rb', freq='60S',
                                                                  missing_threshold=20,
                                                                  train_start_date='2021-07-05',
                                                                  train_end_date='2021-07-08',
                                                                  test_start_date='2021-07-09',
                                                                  test_end_date='2021-07-09')
    # print(train_data_loader.dataset.shape)
