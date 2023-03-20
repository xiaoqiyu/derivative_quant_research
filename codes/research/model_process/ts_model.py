#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 17:39
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : ts_model.py

import torch
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable
from torch.nn import utils as nn_utils
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from datetime import time

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.utils.logger import Logger
from codes.utils.define import *
from codes.research.model_process.dl_models import lstm
from codes.research.data_process.factor_calculation import gen_train_test_features

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = Logger().get_log()


def train_lstm():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = load_features()[['open_close_ratio', 'price_spread', 'aoi', 'label']]
    # train_val_df = df[df.dataDate < test_date]
    # test_df = df[df.dataDate == test_date]

    rnn = lstm().to(device)  # 使用GPU或CPU
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
    loss_func = nn.CrossEntropyLoss()  # 分类问题
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    best_accuracy = 0.0
    train_end_dt = [()]
    for i in range(EPOCH):
        total_train_loss = []
        rnn.train()  # 进入训练模式
        # train_loader, valid_loader = get_dataloader(train_ratio=0.5, df=train_val_df)
        train_loader, valid_loader = get_dataloader(train_ratio=0.5, df=df)
        for step, item in enumerate(train_loader):
            # lr = set_lr(optimizer, i, EPOCH, LR)
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
            b_x = b_x.type(torch.FloatTensor).to(device)  #
            b_y = b_y.type(torch.long).to(device)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
            prediction = rnn(b_x)  # rnn output    # prediction (4, 72, 2)
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y[:, -1, :].view(b_y.size()[0]))  # 计算损失，target要转1-D
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵

        step_valid_loss = []
        total_num = 0
        correct_num = 0
        # best_score = 0.0

        rnn.eval()
        for step, item in enumerate(valid_loader):
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.long).to(device)

            if VALID_CRITERIER == 'cross_entropy_loss':
                with torch.no_grad():
                    prediction = rnn(b_x)  # rnn output
                #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
                #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
                loss = loss_func(prediction[:, -1, :], b_y[:, -1, :].view(b_y.size()[0]))  # calculate loss
                step_valid_loss.append(loss.item())

            elif VALID_CRITERIER == 'accuracy':
                targets = rnn.predict(b_x)
                # correct_tensor = (targets == b_y[:, -1, :].view(b_y.size()[0])).tolist()
                correct_num += sum((targets == b_y[:, -1, :].view(b_y.size()[0])).tolist())
                total_num += targets.shape[0]
            else:
                logger.warn("Invalid valid_criterier: {0}".format(VALID_CRITERIER))
        if VALID_CRITERIER == 'cross_entropy_loss':
            logger.info('Epoch:{0}, mean train loss:{1},std train loss:{2}, valid loss is:{3}'.format(i, np.mean(
                total_train_loss), np.std(total_train_loss), valid_loss))
            valid_loss.append(np.mean(step_valid_loss))
            if valid_loss and valid_loss[-1] < min_valid_loss:
                logger.info("Save model in epoch:{0} with valid_loss:{1}".format(i, valid_loss))
                torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                            'valid_loss': valid_loss},
                           'LSTM_{0}.model'.format(security_id))  # 保存字典对象，里面'model'的value是模型
                torch.save(optimizer, 'LSTM_{0}.optim'.format(security_id))  # 保存优化器
                min_valid_loss = valid_loss[-1]
        elif VALID_CRITERIER == 'accuracy':
            _accuracy = float(correct_num / total_num)
            logger.info('Epoch:{0}, mean train loss:{1},std train loss:{2}, accucury is:{3}'.format(i, np.mean(
                total_train_loss), np.std(total_train_loss), _accuracy))
            if _accuracy > best_accuracy:
                best_accuracy = _accuracy
                logger.info(
                    "Save model in epoch:{0} with valid_accuracy:{1}, and best accuracy:{2}".format(i, _accuracy,
                                                                                                    best_accuracy))
                torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                            'valid_accuracy': _accuracy},
                           'LSTM_{0}.model'.format(security_id))  # 保存字典对象，里面'model'的value是模型
                torch.save(optimizer, 'LSTM_{0}.optim'.format(security_id))  # 保存优化器

        else:
            logger.warn("Invalid valid_criterier: {0}".format(VALID_CRITERIER))

        logger.info('Epoch: {0}, Current learning rate: {1}'.format(i, mult_step_scheduler.get_lr()))
        mult_step_scheduler.step()  # 学习率更新
    plt.plot(train_loss, color='r')
    plt.plot(valid_loss, color='b')
    plt.legend(['train_loss', 'valid_loss'])
    train_loss_track_path = 'train_loss_{0}.jpg'.format(security_id)
    # plt.show()
    logger.info('Save train result figure to :{0}'.format(train_loss_track_path))
    plt.savefig(train_loss_track_path)
    print('end')
