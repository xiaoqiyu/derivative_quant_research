#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 17:39
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : fut_trend_model.py
import math
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
import configparser

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
print(device)
uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)


class ParamModel(object):
    def __init__(self, model_path=''):
        self._path = model_path
        self.std_model = None
        self.bins = None

    def update_model(self, std_model=None, bins=None):
        if std_model is not None:
            logger.info("update param model with var:{0}, means_{1}".format(std_model.var_, std_model.mean_))
            self.std_model = std_model
        if bins is not None:
            logger.info("update param model with bins:{0}".format(bins))
            self.bins = bins

    def dump_model(self, model_path=''):
        if model_path:
            self._path = model_path
        logger.info("saving param model to path:{0}".format(self._path))
        with open(self._path, 'wb') as fout:
            pickle.dump(self, fout)

    def load_model(self, model_path=''):
        if model_path:
            self._path = model_path
        logger.info("loading param model from path:{0}".format(self._path))
        if os.path.exists(self._path):
            with open(self._path, 'rb') as fin:
                _obj = pickle.load(fin)
                self.std_model = _obj.std_model
                self.bins = _obj.bins


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

    def save_model(self, epoch=None, model=None, optimizer=None, train_loss=None, test_loss=None, path=''):
        if path.endswith('tar'):
            self.save_torch_checkpoint(epoch, model, optimizer, train_loss, test_loss, path)
        elif path.endswith('pt'):
            # model_scripted = torch.jit.script(model)  # Export to TorchScript
            # model_scripted.save(path)  # Save
            model.save(path)

    def load_model(self, model=None, optimizer=None, path=None):
        if path.endswith('tar'):
            return self.load_torch_checkpoint(model, optimizer, path)
        elif path.endswith('pt'):
            # path: xx.pt
            # model = torch.jit.load(path)
            # model.eval()
            return torch.load(path)

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
    def train_model(self, product_id: str = 'rb', start_date: str = '', end_date: str = '', train_end_date: str = ''):
        logger.info("Start train model for product:{0} from {1} to {2}".format(product_id, start_date, end_date))
        rnn = RNN().to(device)  # 使用GPU或CPU
        optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
        loss_func = nn.CrossEntropyLoss()  # 分类问题
        _rnn_model_path = os.path.join(_base_dir,
                                       'data\models\\tsmodels\\{0}_{1}.tar'.format(self.model_name, product_id))
        _rnn_model_path_pt = os.path.join(_base_dir,
                                          'data\models\\tsmodels\\{0}_{1}.pt'.format(self.model_name, product_id))
        min_val_loss = np.inf
        first_model = True
        # in one epoch not load checkpoint
        # if not train_base:  # incremental training when there is already a base model
        EPOCH = 50
        learning_rate_gamma = 0.1
        if os.path.exists(_rnn_model_path):
            epoch, rnn, optimizer, cache_train_loss, cache_test_loss = self.load_torch_checkpoint(rnn, optimizer,
                                                                                                  _rnn_model_path)
            optimizer.param_groups[0]['lr'] = 0.00001
            learning_rate_gamma = 1
            logger.info("loading from existing model with train loss:{0} and test_loss:{1}".format(cache_train_loss,
                                                                                                   cache_test_loss))
            min_val_loss = cache_test_loss
            first_model = False

        # if not train_base:
        #     EPOCH = 5
        #     learning_rate_gamma = 1
        # else:
        #     EPOCH = 15
        #     learning_rate_gamma = 0.1

        mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                   milestones=[EPOCH // 2, EPOCH // 4 * 3],
                                                                   gamma=learning_rate_gamma)
        train_loss = []
        val_loss = []
        # min_test_loss = cache_test_loss
        # min_test_loss = np.inf if train_base else cache_test_loss
        _param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels_{0}.pkl'.format(product_id))
        param_model = ParamModel(_param_model_path)
        # TODO not load model here, for training, it will call fit_transform
        # param_model.load_model()

        val_predicts = []
        val_labels = []
        val_epoch = []
        # train_test_dates = self.get_train_test_dates(start_date=start_date, end_date=end_date)
        all_trade_dates = self.data_fetcher.get_all_trade_dates(start_date, end_date)
        # train_end_idx = int(len(all_trade_dates) * 0.7) if train_base else all_trade_dates.index(train_end_date)
        train_end_idx = all_trade_dates.index(train_end_date)
        _train_end_date = all_trade_dates[train_end_idx]
        _val_start_date = all_trade_dates[train_end_idx + 1]
        logger.info(
            "Get dataloader for train:{0}-{1}, valid:{2}-{3}".format(start_date, _train_end_date, _val_start_date,
                                                                     end_date))
        train_data_loader, val_data_loader = gen_train_test_features(data_fetcher=self.data_fetcher,
                                                                      param_model=param_model,
                                                                      product_id=product_id,
                                                                      freq="{0}S".format(SEC_INTERVAL),
                                                                      missing_threshold=MISSING_THRESHOLD,
                                                                      train_start_date=start_date,
                                                                      train_end_date=_train_end_date,
                                                                      test_start_date=_val_start_date,
                                                                      test_end_date=end_date)
        logger.info('train data loader size:{0},val data loader size:{1}'.format(train_data_loader.dataset.size(),
                                                                                 val_data_loader.dataset.size()))

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

            if i % 10 != 0:
                continue
            step_val_loss = []
            rnn.eval()  # 进入样本外测试模式

            curr_val_predicts = []
            curr_val_label = []
            with torch.no_grad():
                prediction = rnn(val_data_loader.dataset[:, :, :-1])
                val_predicts_epoch = rnn.predict(val_data_loader.dataset[:, :, :-1])
                val_labels_epoch = val_data_loader.dataset[:, 0, -1]
                val_labels_epoch = val_labels_epoch.type(torch.long).to(device)
                if prediction.size()[0] == val_labels_epoch.size()[0]:
                    loss = loss_func(prediction, val_labels_epoch)  # calculate loss
                    step_val_loss.append(loss.item())
                else:
                    logger.warn(
                        "predict:{0} and y label:{1} not the same len".format(len(val_predicts), len(val_labels)))
                curr_val_predicts.extend(val_predicts_epoch.tolist())
                curr_val_label.extend(val_labels_epoch.tolist())
                val_predicts.extend(curr_val_predicts)
                val_labels.extend(curr_val_label)
                val_epoch.extend([i] * val_labels_epoch.size()[0])
            _correct_num = len([item for idx, item in enumerate(curr_val_predicts) if curr_val_label[idx] == item])
            true_0 = len([item for item in curr_val_label if item == 0])
            true_1 = len([item for item in curr_val_label if item == 1])
            true_2 = len([item for item in curr_val_label if item == 2])
            pred_0 = len([item for item in curr_val_predicts if item == 0])
            pred_1 = len([item for item in curr_val_predicts if item == 1])
            pred_2 = len([item for item in curr_val_predicts if item == 2])
            _correct_num0 = len(
                [item for idx, item in enumerate(curr_val_predicts) if curr_val_label[idx] == item and item == 0])
            _correct_num1 = len(
                [item for idx, item in enumerate(curr_val_predicts) if curr_val_label[idx] == item and item == 1])
            _correct_num2 = len(
                [item for idx, item in enumerate(curr_val_predicts) if curr_val_label[idx] == item and item == 2])
            _acc0 = round(float(_correct_num0 / pred_0), 2) if pred_0 else 0.0
            _acc1 = round(float(_correct_num1 / pred_1), 2) if pred_1 else 0.0
            _acc2 = round(float(_correct_num2 / pred_2), 2) if pred_2 else 0.0
            _train_loss = round(float(np.mean(total_train_loss)), 4)
            _acc = round(float(_correct_num / len(curr_val_predicts)), 4)
            curr_epoch_val_loss = round(float(np.mean(step_val_loss)), 4)
            val_loss.append(curr_epoch_val_loss)
            logger.info(
                "Epoch:{0},true 0/1/2:({1}/{2}/{3}),pred 0/1/2:({4}/{5}/{6}),acc 0/1/2:({7}/{8}/{9}),train loss:{10}, "
                "valid loss:{11}, acc:{12},learning rate:{13}".format(
                    i, true_0, true_1, true_2,
                    pred_0, pred_1, pred_2, _acc0, _acc1, _acc2, _train_loss, curr_epoch_val_loss, _acc,
                    mult_step_scheduler.get_lr()))
            # TODO remove hardcode, save model(parameter) for better valid loss and accuracy > 0.5
            # if first_model or (val_loss and curr_epoch_val_loss < min_val_loss and _acc > 0.5):
            if first_model or (val_loss and curr_epoch_val_loss < min_val_loss and _acc > 0.3):
                logger.info("Epoch:{0} Save model with val_loss:{1} and prev val loss:{2}, accu:{3}".format(i,
                                                                                                            curr_epoch_val_loss,
                                                                                                            min_val_loss,
                                                                                                            _acc))
                min_val_loss = curr_epoch_val_loss
                # self.save_torch_checkpoint(i, rnn, optimizer, train_loss, min_test_loss, _rnn_model_path)
                self.save_model(i, rnn, optimizer, train_loss, min_val_loss, _rnn_model_path)
                self.save_model(i, rnn, optimizer, train_loss, min_val_loss, _rnn_model_path_pt)

            # logger.info('Epoch: {0}, Current learning rate: {1}'.format(i, mult_step_scheduler.get_lr()))
            mult_step_scheduler.step()  # 学习率更新
        df_predict = pd.DataFrame({'y_true': val_labels, 'y_predict': val_predicts, 'epoch': val_epoch})
        predict_path = os.path.join(_base_dir,
                                    'data\models\\tsmodels\\{0}_predict_{1}.csv'.format(self.model_name, product_id))

        df_predict.to_csv(predict_path, index=False)

        # TODO  not plot for results
        plt.plot(train_loss, color='r')
        plt.plot(val_loss, color='b')
        plt.legend(['train_loss', 'val_loss'])
        train_loss_track_path = os.path.join(_base_dir,
                                             'data\models\\tsmodels\\{0}_train_loss_{1}_{2}_{3}'.format(
                                                 self.model_name,
                                                 product_id, start_date, end_date))
        logger.info(
            'Complete train for epoch:{0}, save train result figure to :{1}'.format(i, train_loss_track_path))
        plt.savefig(train_loss_track_path)


class LRModel(object):
    def __init__(self, data_fetcher: DataFetcher = None):
        self.data_fetcher = data_fetcher
        self.data_path = os.path.join(_base_dir, 'data')

    def train_model(self):
        pass


def torch_infer(product_id='rb', x=None):
    _rnn_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\rnn_{0}.pt'.format(product_id))
    model = torch.load(_rnn_model_path).to(device)
    if x is not None:
        x = x.to(device)
        y = model.predict(x)
    return y


# currently  only has RNN model
def test_model(product_id='rb', x=None, start_date='2021-07-01', end_date='2021-07-31'):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn_model = RNNModel()
    _rnn_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\rnn_{0}.tar'.format(product_id))
    epoch, model, optimizer, train_loss, test_loss = rnn_model.load_torch_checkpoint(path=_rnn_model_path)

    _param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels_{0}.pkl'.format(product_id))
    param_model = ParamModel(_param_model_path)
    param_model.load_model()
    model.to(device)
    if x is not None:
        x.to(device)
        y_pred = model.predict(x)
    else:
        data_loader, bins, dt_cols = gen_predict_feature_dataset(data_fetcher=data_fetcher, param_model=param_model,
                                                                 product_id=product_id,
                                                                 freq="{0}S".format(SEC_INTERVAL),
                                                                 missing_threshold=MISSING_THRESHOLD,
                                                                 start_date=start_date, end_date=end_date,
                                                                 )
        dt_cols = [item.split() for item in dt_cols]
        dt_cols = ['{0}_{1}'.format(item[0], item[1].split('.')[0]) for item in dt_cols]
        y_pred = model.predict(data_loader.dataset[:, :, :-1])
        y_true = data_loader.dataset[:, 0, -1]

        curr_test_predicts = []
        curr_test_label = []
        curr_test_predicts.extend(y_pred.tolist())
        curr_test_label.extend(y_true.tolist())
        _correct_num = len([item for idx, item in enumerate(curr_test_predicts) if curr_test_label[idx] == item])
        true_0 = len([item for item in curr_test_label if item == 0])
        true_1 = len([item for item in curr_test_label if item == 1])
        true_2 = len([item for item in curr_test_label if item == 2])
        pred_0 = len([item for item in curr_test_predicts if item == 0])
        pred_1 = len([item for item in curr_test_predicts if item == 1])
        pred_2 = len([item for item in curr_test_predicts if item == 2])
        _correct_num0 = len(
            [item for idx, item in enumerate(curr_test_predicts) if curr_test_label[idx] == item and item == 0])
        _correct_num1 = len(
            [item for idx, item in enumerate(curr_test_predicts) if curr_test_label[idx] == item and item == 1])
        _correct_num2 = len(
            [item for idx, item in enumerate(curr_test_predicts) if curr_test_label[idx] == item and item == 2])
        _acc0 = round(float(_correct_num0 / pred_0), 2) if pred_0 else 0.0
        _acc1 = round(float(_correct_num1 / pred_1), 2) if pred_1 else 0.0
        _acc2 = round(float(_correct_num2 / pred_2), 2) if pred_2 else 0.0
        _acc = round(float(_correct_num / len(curr_test_predicts)), 4)
        logger.info(
            "******test results*****: Start data:{0},true 0/1/2:({1}/{2}/{3}),pred 0/1/2:({4}/{5}/{6}),"
            "acc 0/1/2:({7}/{8}/{9}),acc:{10}".format(
                start_date, true_0, true_1, true_2,
                pred_0, pred_1, pred_2, _acc0, _acc1, _acc2, _acc, ))

    curr_signal_map = {}
    _signal_path = os.path.join(_base_dir, 'data\models\\tsmodels\\signal_{0}.csv'.format(product_id))
    if os.path.exists(_signal_path):
        _df_signal = pd.read_csv(_signal_path)
        curr_signal_map = dict(zip(_df_signal['datetime'], _df_signal['signal']))
    new_signal_lst = y_pred.tolist()
    new_signal_map = dict(zip(dt_cols, new_signal_lst))
    new_signal_map.update(curr_signal_map)

    pd.DataFrame({'datetime': list(new_signal_map.keys()), 'signal': list(new_signal_map.values())}).to_csv(
        _signal_path, index=False)
    return dict(zip(dt_cols, y_pred.tolist()))


def train_all(model_name='rnn', product_id='rb', start_date='2021-07-01', end_date='2021-07-15', train_end_date='',
              train_base=True):
    if model_name == 'rnn':
        ts_model = RNNModel(data_fetcher)
        ts_model.train_model(product_id=product_id, start_date=start_date, end_date=end_date, train_end_date='',
                             train_base=True)


def incremental_train_and_infer(model_name='rnn', product_id='rb', start_date='2021-07-01', end_date='2021-07-15',
                                train_end_date='', infer_start_date=''):
    if model_name == 'rnn':
        ts_model = RNNModel(data_fetcher)
        ts_model.train_model(product_id=product_id, start_date=start_date, end_date=end_date,
                             train_end_date=train_end_date, train_base=False)
        logger.info("start infer from {0} to {1}".format(infer_start_date, end_date))
        test_model(product_id=product_id, start_date=infer_start_date, end_date=end_date)


def save_param_model(product_id='rb'):
    _param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels.pkl')
    param_model = ParamModel(_param_model_path)
    param_model.load_model()
    print(param_model.bins)
    # config = configparser.ConfigParser()
    # config[product_id]['bins'] = param_model.bins
    out_param_model_path = os.path.join(_base_dir, 'data\models\\tsmodels\\tsmodels_{}.txt'.format(product_id))
    # with open(out_param_model_path, 'w') as configfile:
    #     config.write(configfile)
    bins_str = ''
    for item in param_model.bins:
        bins_str = '{},{}'.format(bins_str, round(item, 10))
    mean_str = ''
    for item in param_model.std_model.mean_:
        mean_str = '{},{}'.format(mean_str, round(item, 10))

    std_str = ''
    for item in param_model.std_model.var_:
        std_str = '{},{}'.format(std_str, round(math.sqrt(item), 10))
    with open(out_param_model_path, 'w') as out_file:
        out_file.writelines('{}\n'.format(bins_str.strip(',')))
        out_file.writelines('{}\n'.format(mean_str.strip(',')))
        out_file.writelines('{}\n'.format(std_str.strip(',')))


if __name__ == '__main__':
    # train base, delete existing model file, it will train from scratch
    train_all(model_name='rnn', product_id='rb', start_date='2021-01-04', end_date='2021-01-15', train_base=True)

    dates = [('2021-03-01', '2021-03-05', '2021-03-08', '2021-03-12')]
    for start_date, train_end_date, infer_start_date, end_date in dates:
        incremental_train_and_infer(model_name='rnn', product_id='rb', start_date=start_date, end_date=end_date,
                                    train_end_date=train_end_date, infer_start_date=infer_start_date)

    x = torch.from_numpy(np.random.randn(3 * 120 * 6).reshape(3, 120, 6)).to(torch.float32)
    y = torch_infer('rb', x)
    print(y.shape)
    print(y)
    # save_param_model()
