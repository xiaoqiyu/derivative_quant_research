#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 16:23
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : model_evaluation.py


# TODO get train/test start/end date which will iter in each EPOCH
# TODO in train process, define a model which cache the standardized params and bin params
# TODO use cross_entropy_loss to train and update model, and record the prediction label for test dataset to evaluate the performance of different aspect


import os
import torch
import copy
from torch.utils.data import DataLoader
import sys
import pandas as pd
import time

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)
from codes.utils.define import *


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


if __name__ == '__main__':
    feature_evalution()

