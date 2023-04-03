#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 17:32
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : dl_models.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.utils.define import *


# 定义LSTM的结构
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.linear = nn.Linear(in_features=INPUT_SIZE, out_features=RNN_INPUT_SIZE, bias=True)
        self.rnn = nn.LSTM(
            input_size=RNN_INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        # self.rnn = nn.RNN(
        #     input_size=RNN_INPUT_SIZE,
        #     hidden_size=HIDDEN_SIZE,
        #     num_layers=LAYERS,
        #     dropout=DROP_RATE,
        #     batch_first=True
        #
        # )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, NUM_LABEL)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        l_out = self.linear(x)
        r_out, (h_s, h_c) = self.rnn(l_out)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output

    def predict(self, x):
        _distribution = F.softmax(self.forward(x)[:, -1, :])
        return torch.argmax(_distribution, dim=1)


class LrModel(nn.Module):
    pass