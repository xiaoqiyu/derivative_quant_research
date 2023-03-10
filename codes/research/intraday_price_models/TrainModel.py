#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/22 14:04
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : TrainModel.py


import uqer
import numpy as np
from uqer import DataAPI
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
from editorconfig import get_properties, EditorConfigError
import logging
import os

import codes.utils.define as define
from codes.utils import utils as utils
import pickle
from codes import research as F

logging.basicConfig(filename='logs/{0}.txt'.format(os.path.split(__file__)[-1].split('.')[0]), level=logging.DEBUG)
logger = logging.getLogger()

try:
    _conf_file = utils.get_path([define.CONF_DIR,
                                 define.CONF_FILE_NAME])
    options = get_properties(_conf_file)
except EditorConfigError:
    logging.warning("Error getting EditorConfig propterties", exc_info=True)
else:
    for key, value in options.items():
        # _config = '{0},{1}:{2}'.format(_config, key, value)
        print("{0}:{1}".format(key, value))

uqer_client = uqer.Client(token=options.get('uqer_token'))


def _feature_preprocess(df_ref_factor, df_factor, cols):
    _tmp = {}
    for col in cols:
        _low, _high = df_ref_factor[col].quantile([0.1, 0.9])
        _max = df_ref_factor[col].max()
        _min = df_ref_factor[col].min()
        ref = np.array([_low if item < _low else _high if item > _high else item for item in df_ref_factor[col]])
        _std = ref.std()
        _mean = ref.mean()
        val = [_min if item < _min else _max if item > _max else item for item in df_factor[col]]
        val = [_low if item < _mean - 10 * _std else _high if item > _mean + 10 * _std else item for item in
               val]
        # _tmp[col] = [(item - _low) / (_high - _low) for item in val]
        # _tmp[col] = [(item - _min) / (_max - _min) for item in val]

        # _tmp[col] = [(item - _min) / (_max - _min) for item in df_factor[col] ]
        _tmp[col] = df_factor[col]
        _tmp[define.REG_LABEL_NAME] = df_factor[define.REG_LABEL_NAME]
    return pd.DataFrame(_tmp)


def train_model_reg(predict_windows: list = [20],
                    lag_windows: list = [60],
                    start_date: str = '',
                    end_date: str = '',
                    top_k_features: int = 1,
                    train_days: int = 3,
                    product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]

    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'regmodel_{0}.pkl'.format(product_id))

    try:
        # Load from file
        with open(_model_name, 'rb') as file:
            model2 = pickle.load(file)
            print('model loaded====>', model2.coef_, model2.intercept_, model2.classes_)
    except Exception as ex:
        print('load model:{0} fail with error:{1}'.format(_model_name, ex))
        model2 = _lin_reg_model = LinearRegression()

    acc_scores = {}
    factor_lst = []
    cnt = 0
    related_factors = []

    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date with date:{0}", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)

        cols = list(df_factor.columns)
        # cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #         'trend_1', 'bsvol_volume_1',
        #         'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #         'bsvol_volume_ls_ratio']
        cols = define.train_cols
        import copy
        reg_cols = copy.deepcopy(define.train_cols)
        reg_cols.append('label_1')
        # reg_cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #             'trend_1', 'bsvol_volume_1',
        #             'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #             'bsvol_volume_ls_ratio', 'label_1']

        df_corr = df_factor[reg_cols].corr()
        df_corr.to_csv('corr_{0}.csv'.format(date))
        for item in reg_cols:
            try:
                _similar_factors = list(df_corr[df_corr[item] > 0.07][item].index)
                for _k in _similar_factors:
                    if not ((item, _k) in related_factors or (_k, item) in related_factors) and _k != item:
                        related_factors.append((item, _k))
            except Exception as ex:
                print(ex, date)
        factor_lst.append(df_factor)

    if top_k_features < 0:
        acc_scores = dict(zip(cols, [0.1] * len(cols)))

    for k1, k2 in related_factors:
        if k1 in acc_scores and k2 in acc_scores:
            print("{0}:{1} and {2}:{3} in dict, pop{4}".format(k1, acc_scores.get(k1), k2, acc_scores.get(k2), k2))
            # acc_scores.pop(k2)  # TODO always pop k2, add more logic here if needed

    top_k_features = len(acc_scores) if top_k_features < 0 else top_k_features
    sorted_scores = sorted(acc_scores.items(), reverse=True, key=lambda x: x[1])[-top_k_features:]
    sorted_features = [item[0] for item in sorted_scores]
    logger.info(sorted_scores)
    print("sorted features:", sorted_features)

    df_train = None
    if len(factor_lst) > 1:
        for idx in list(range(1, len(factor_lst))):
            _df_train = _feature_preprocess(factor_lst[idx - 1], factor_lst[idx], sorted_features)
            if idx == 1:
                df_train = copy.deepcopy(_df_train)
            else:
                df_train = df_train.append(_df_train)
    df_train.to_csv('train_factor.csv', index=False)
    # df_train = _resample(factor_lst[0])
    # for df_factor in factor_lst[1:]:
    #     df_train = df_train.append(_resample(df_factor))
    model2.fit(df_train[sorted_features], df_train[define.REG_LABEL_NAME])
    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'regmodel_{0}.pkl'.format(product_id))

    with open(_model_name, 'wb') as file:
        pickle.dump(model2, file)

    del factor_lst

    # for df_factor in factor_lst:
    #     df_train = _resample(df_factor)
    #     model2.fit(df_train[sorted_features], df_train['label_clf_1'])
    logger.info('test for trade date:{0}'.format(test_dates[1]))

    df_test = F.get_factor(trade_date=test_dates[1],
                           predict_windows=predict_windows,
                           lag_windows=lag_windows,
                           instrument_id=test_dates[0],
                           exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')

    # y_pred = model2.predict(model1.transform(df_factor[cols])) #TODO why this??
    # if there might be class imbalance, then  micro is preferable since micro-average will aggregate the contributions
    # of all classes to compute the average metric, while macro-average will compute the metric independently for each
    # class and then take the average(hence treating all classes equally)

    y_true = list(df_test[define.REG_LABEL_NAME])
    y_pred = model2.predict(df_test[sorted_features])
    _update_time = list(df_test['UpdateTime'])
    _update_time_str = [item.split()[-1] for item in _update_time]

    dict_score = {}

    df_pred = pd.DataFrame({'pred': y_pred, 'true': y_true, 'UpdateTime': _update_time_str})

    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'df_pred_{0}.csv'.format(test_dates[1]))
    df_pred.to_csv(_file_name)

    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}, instrument_id:{5}\n'.format(
        np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        metrics.r2_score(y_true, y_pred), test_dates[1], predict_windows, lag_windows, instrument_id)

    _model_evalulate_path = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                         define.TICK_MODEL_DIR,
                                         'model_evaluate_{0}_{1}.txt'.format(test_dates[1], instrument_id))
    with open(_model_evalulate_path, 'a') as f:
        f.write(ret_str)

    coef_param = dict(zip(sorted_features, list(model2.coef_)))
    reg_params = {'coef': coef_param, 'intercept': model2.intercept_}

    _param_file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                    define.TICK_MODEL_DIR,
                                    'reg_params_{1}.json'.format(test_dates[1].replace('-', ''), instrument_id))
    utils.write_json_file(_param_file_name, reg_params)

    # r_ret = [item - y_pred[idx] for idx, item in enumerate(y_true)]
    # plt.plot(r_ret)
    print(ret_str)
    # plt.show()
    return dict_score
