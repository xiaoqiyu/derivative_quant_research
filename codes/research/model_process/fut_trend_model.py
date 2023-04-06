#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 17:39
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : fut_trend_model.py
import pprint

import torch
import torch.nn as nn
import uqer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.utils.logger import Logger
from codes.utils.define import *
from codes.research.model_process.dl_models import RNN
from codes.research.data_process.factor_calculation import gen_train_test_features, gen_predict_feature_dataset
from codes.research.data_process.data_fetcher import DataFetcher
from codes.utils.helper import timeit

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_log_path = os.path.join(_base_dir, 'data\logs\{0}'.format(os.path.split(__file__)[-1].strip('.py')))
logger = Logger(_log_path, 'INFO', __name__).get_log()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)


class ParamModel(object):
    def __init__(self, model_path=''):
        self._path = model_path
        self.std_model = None
        self.bins = None

    def update_model(self, std_model=None, bins=None):
        if std_model is not None:
            self.std_model = std_model
        if bins is not None:
            self.bins = bins

    def dump_model(self, model_path=''):
        if model_path:
            self._path = model_path
        with open(self._path, 'wb') as fout:
            pickle.dump(self, fout)

    def load_model(self, model_path=''):
        if model_path:
            self._path = model_path
        if os.path.exists(self._path):
            with open(self._path, 'rb') as fin:
                return pickle.load(fin)


class RNNModel(object):
    def __init__(self, data_fetcher: DataFetcher = None):
        self.data_fetcher = data_fetcher
        self.data_path = os.path.join(_base_dir, 'data')
        self.model_name = 'rnn'

    @timeit
    def save_torch_checkpoint(self, epoch, model, optimizer, train_loss, test_loss, path):  # path convension is .tar
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, path)

    @timeit
    def load_torch_checkpoint(self, model=None, optimizer=None, path=None):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not model:
            model = RNN().to(device)  # 使用GPU或CPU
        if not optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all rnn parameters
        epoch, train_loss, test_loss = 0, np.inf, np.inf
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            test_loss = checkpoint['test_loss']
        return epoch, model, optimizer, train_loss, test_loss

    def get_train_test_dates(self, start_date: str = '', end_date: str = ''):
        all_trade_dates = self.data_fetcher.get_all_trade_dates(start_date, end_date)
        _len = len(all_trade_dates)
        ret = []
        train_days = int(EPOCH_DAYS * TRAIN_RATIO)

        for i in range(_len - EPOCH_DAYS):
            ret.append([all_trade_dates[i], all_trade_dates[i + train_days - 1], all_trade_dates[i + train_days],
                        all_trade_dates[i + EPOCH_DAYS - 1]])
        return ret

    @timeit
    def train_model(self, product_id: str = 'rb', start_date: str = '', end_date: str = ''):
        logger.info("Start train model for product:{0} from {1} to {2}".format(product_id, start_date, end_date))
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rnn = RNN().to(device)  # 使用GPU或CPU
        optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
        loss_func = nn.CrossEntropyLoss()  # 分类问题
        _rnn_model_path = os.path.join(_base_dir,
                                       'data\models\\tsmodels\\{0}_{1}.tar'.format(self.model_name, product_id))
        # in one epoch not load checkpoint
        # epoch, rnn, optimizer, cache_train_loss, cache_test_loss = self.load_torch_checkpoint(rnn, optimizer,
        #                                                                                       _rnn_model_path)
        mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                   milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
        train_loss = []
        test_loss = []
        # min_test_loss = cache_test_loss
        min_test_loss = np.inf
        _param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels.pkl')
        param_model = ParamModel(_param_model_path)
        param_model = param_model.load_model()

        test_predicts = []
        test_labels = []
        test_epoch = []
        # train_test_dates = self.get_train_test_dates(start_date=start_date, end_date=end_date)
        all_trade_dates = self.data_fetcher.get_all_trade_dates(start_date, end_date)
        train_end_idx = int(len(all_trade_dates) * 0.7)
        _train_end_date = all_trade_dates[train_end_idx]
        _test_start_date = all_trade_dates[train_end_idx + 1]
        logger.info(
            "Get dataloader for train:{0}-{1}, test:{2}-{3}".format(start_date, _train_end_date, _test_start_date,
                                                                    end_date))
        train_data_loader, test_data_loader = gen_train_test_features(data_fetcher=self.data_fetcher,
                                                                      param_model=param_model,
                                                                      product_id=product_id,
                                                                      freq="{0}S".format(SEC_INTERVAL),
                                                                      missing_threshold=MISSING_THRESHOLD,
                                                                      train_start_date=start_date,
                                                                      train_end_date=_train_end_date,
                                                                      test_start_date=_test_start_date,
                                                                      test_end_date=end_date)
        for i in range(EPOCH):
            total_train_loss = []
            rnn.train()  # 进入训练模式
            for step, item in enumerate(train_data_loader):
                # lr = set_lr(optimizer, i, EPOCH, LR)
                nx, ny, nz = item.shape
                blocks = torch.chunk(item, nz, dim=2)
                b_x = torch.cat(blocks[:-1], 2)
                b_y = blocks[-1]
                b_x = b_x.type(torch.FloatTensor).to(device)  #
                b_y = b_y.type(torch.long).to(device)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
                prediction = rnn(b_x)
                #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
                #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
                # loss = loss_func(prediction[:, -1, :], b_y[:, 0, :].view(b_y.size()[0]))  # 计算损失，target要转1-D
                loss = loss_func(prediction, b_y[:, 0, :].view(b_y.size()[0]))
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                # TODO double check the grad norm parameters
                nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=10, norm_type=2)
                optimizer.step()  # apply gradients
                total_train_loss.append(loss.item())
            train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵

            step_test_loss = []
            rnn.eval()  # 进入样本外测试模式
            for step, item in enumerate(test_data_loader):
                nx, ny, nz = item.shape
                blocks = torch.chunk(item, nz, dim=2)
                b_x = torch.cat(blocks[:-1], 2)
                b_y = blocks[-1]
                b_x = b_x.type(torch.FloatTensor).to(device)
                b_y = b_y.type(torch.long).to(device)

                with torch.no_grad():
                    prediction = rnn(b_x)  # rnn output
                loss = loss_func(prediction, b_y[:, 0, :].view(b_y.size()[0]))  # calculate loss
                step_test_loss.append(loss.item())
                targets = rnn.predict(b_x)
                targets_lst = targets.tolist()
                y_lst = b_y[:, 0, :].view(b_y.size()[0]).tolist()
                if len(targets_lst) == len(y_lst):
                    test_predicts.extend(targets_lst)
                    test_labels.extend(y_lst)
                    test_epoch.extend([i] * len(y_lst))
                else:
                    logger.warn("predict:{0} and y label:{1} not the same len".format(len(targets_lst), len(y_lst)))
            _correct_num = len([item for idx, item in enumerate(test_predicts) if test_labels[idx] == item])
            true_0 = len([item for item in test_labels if item == 0])
            true_1 = len([item for item in test_labels if item == 1])
            true_2 = len([item for item in test_labels if item == 2])
            pred_0 = len([item for item in test_predicts if item == 0])
            pred_1 = len([item for item in test_predicts if item == 1])
            pred_2 = len([item for item in test_predicts if item == 2])
            _correct_num0 = len(
                [item for idx, item in enumerate(test_predicts) if test_labels[idx] == item and item == 0])
            _correct_num1 = len(
                [item for idx, item in enumerate(test_predicts) if test_labels[idx] == item and item == 1])
            _correct_num2 = len(
                [item for idx, item in enumerate(test_predicts) if test_labels[idx] == item and item == 2])
            _acc0 = round(float(_correct_num0 / pred_0), 2) if pred_0 else 0.0
            _acc1 = round(float(_correct_num1 / pred_1), 2) if pred_1 else 0.0
            _acc2 = round(float(_correct_num2 / pred_2), 2) if pred_2 else 0.0
            logger.info(
                "Epoch:{0},true:0=>{1},1=>{2},2=>{3};pred:0=>{4},1=>{5},2=>{6}".format(i, true_0, true_1, true_2,
                                                                                       pred_0, pred_1, pred_2))
            logger.info(
                "Accu:0=>{0}, 1=>{1}, 2=>{2}, corr_num0=>{3}, corr_num1=>{4}, corr_num2=>{5}".format(_acc0, _acc1,
                                                                                                     _acc2,
                                                                                                     _correct_num0,
                                                                                                     _correct_num1,
                                                                                                     _correct_num2))
            curr_epoch_test_loss = np.mean(step_test_loss)
            test_loss.append(curr_epoch_test_loss)
            logger.info(
                'Epoch:{0}, mean train loss:{1},test loss:{2},acc :{3}, train loss std is:{4}'.format(i, np.mean(
                    total_train_loss), curr_epoch_test_loss, _correct_num / len(
                    test_predicts), np.std(total_train_loss)))
            if test_loss and curr_epoch_test_loss < min_test_loss:
                logger.info("Save model in epoch:{0} with test_loss:{1}".format(i, curr_epoch_test_loss))
                min_test_loss = curr_epoch_test_loss
                self.save_torch_checkpoint(i, rnn, optimizer, train_loss, min_test_loss, _rnn_model_path)
            logger.info('Epoch: {0}, Current learning rate: {1}'.format(i, mult_step_scheduler.get_lr()))
            mult_step_scheduler.step()  # 学习率更新
        df_predict = pd.DataFrame({'y_true': test_labels, 'y_predict': test_predicts, 'epoch': test_epoch})
        predict_path = os.path.join(_base_dir,
                                    'data\models\\tsmodels\\{0}_predict_{1}.csv'.format(self.model_name, product_id))

        df_predict.to_csv(predict_path, index=False)
        plt.plot(train_loss, color='r')
        plt.plot(test_loss, color='b')
        plt.legend(['train_loss', 'test_loss'])
        train_loss_track_path = os.path.join(_base_dir,
                                             'data\models\\tsmodels\\{0}_train_loss_{1}'.format(self.model_name,
                                                                                                product_id))
        logger.info(
            'Complete train for epoch:{0}, save train result figure to :{1}'.format(i, train_loss_track_path))
        plt.savefig(train_loss_track_path)


class LRModel(object):
    def __init__(self, data_fetcher: DataFetcher = None):
        self.data_fetcher = data_fetcher
        self.data_path = os.path.join(_base_dir, 'data')

    def train_model(self):
        pass


# currently  only has RNN model
def stacking_infer(product_id='rb', x=None, start_date='2021-07-01', end_date='2021-07-31'):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn_model = RNNModel()
    _rnn_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\rnn_{0}.tar'.format(product_id))
    epoch, model, optimizer, train_loss, test_loss = rnn_model.load_torch_checkpoint(path=_rnn_model_path)
    # model.to_device(device)

    _param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels.pkl')
    param_model = ParamModel(_param_model_path)
    param_model = param_model.load_model()
    model.to(device)
    if x is not None:
        x.to(device)
        y = model.predict(x)
    else:
        data_loader, bins, dt_cols = gen_predict_feature_dataset(data_fetcher=data_fetcher, param_model=param_model,
                                                                 product_id=product_id,
                                                                 freq="{0}S".format(SEC_INTERVAL),
                                                                 missing_threshold=MISSING_THRESHOLD,
                                                                 start_date=start_date, end_date=end_date,
                                                                 )
        dt_cols = [item.split() for item in dt_cols]
        dt_cols = ['{0}_{1}'.format(item[0], item[1].split('.')[0]) for item in dt_cols]
        y = model.predict(data_loader.dataset[:, :, :-1])
    return dict(zip(dt_cols, y.tolist()))


def train_all(model_name='rnn', product_id='rb', start_date='2021-07-01', end_date='2021-07-15'):
    if model_name == 'rnn':
        ts_model = RNNModel(data_fetcher)
        ts_model.train_model(product_id=product_id, start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    # train_all(model_name='rnn', product_id='rb', start_date='2021-07-01', end_date='2021-07-31')
    # x = torch.randn(5, 120, 15, device=device)
    y = stacking_infer(product_id='rb', x=None, start_date='2021-07-01', end_date='2021-07-02')
    pprint.pprint(y)

