#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 16:23
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : model_evaluation.py


import os
import sys
import pandas as pd
import time
import uqer

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)
from codes.utils.define import *
from codes.utils.logger import Logger
from codes.research.model_process.fut_trend_model import train_all
from codes.research.model_process.fut_trend_model import incremental_train_and_infer
from codes.research.model_process.fut_trend_model import stacking_infer
from codes.research.data_process.data_fetcher import DataFetcher

_log_path = os.path.join(_base_dir, 'data\logs\{0}'.format(os.path.split(__file__)[-1].strip('.py')))
logger = Logger(_log_path, 'INFO', __name__).get_log()

uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)


# TODO add this later
def feature_evalution(product_id='rb', start_date='', end_date='', freq='300S'):
    file_name = 'features_rb_2021-07-05_2021-07-09.csv'
    feature_path = os.path.join(_base_dir, 'data\\features\\{0}'.format(file_name))
    print(feature_path)
    df = pd.read_csv(feature_path, encoding='gbk')
    df = df[TEST_FEATURES]
    cols = list(df.columns)
    df.index = pd.to_datetime(df['UpdateTime'])
    # ['UpdateTime', 'open_close_ratio', 'price_spread', 'aoi', 'wap_log_return']
    df = pd.concat([df.loc[time(9, 30): time(11, 30)], df.loc[time(21, 0):time(23, 0)]])
    df.resample(freq).apply(
        {"open_close_ratio": "mean", "price_spread": "mean", "aoi": "mean", "BS_VOL": "sum"}
    ).dropna().droplevel(level=0, axis=1)


def model_evaluation(start_date='2021-03-01', end_date='2021-12-31', infer_weeks=4, product_id='rb', model_name='rnn'):
    # base model training, delete existing model file, it will train from scratch
    # train_all(model_name='rnn', product_id='rb', start_date='2021-01-04', end_date='2021-02-26', train_base=True)
    # incremental training
    week_start_end = data_fetcher.get_week_start_end(start_date=start_date, end_date=end_date)
    week_num = len(week_start_end)
    if week_num < infer_weeks:
        logger.warn("evaluation period is too short, less than target infer weeks:{0}".format(infer_weeks))
        return
    train_infer_dates = []
    for idx in range(infer_weeks - 1, week_num):
        _start_train_week = idx + 1 - infer_weeks
        _end_train_week = idx - 1
        train_infer_dates.append((week_start_end[_start_train_week][0], week_start_end[_end_train_week][1],
                                  week_start_end[idx][0], week_start_end[idx][1]))
    # dates = [('2021-03-01', '2021-03-05', '2021-03-08', '2021-03-12')]
    for start_date, train_end_date, infer_start_date, end_date in train_infer_dates:
        incremental_train_and_infer(model_name=model_name, product_id=product_id, start_date=start_date,
                                    end_date=end_date,
                                    train_end_date=train_end_date, infer_start_date=infer_start_date)
        # FIXME remove hardcode for debug
        break


if __name__ == '__main__':
    # feature_evalution()
    model_evaluation()
