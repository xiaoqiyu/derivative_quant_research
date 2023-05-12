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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROP_RATE)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)

    def forward(self, X):
        return self.fc2(self.relu(self.fc1(X)))

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)


# 定义LSTM的结构
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.fc = nn.Linear(in_features=INPUT_SIZE, out_features=RNN_INPUT_SIZE, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        # self.rnn = nn.LSTM(
        #     input_size=RNN_INPUT_SIZE,
        #     hidden_size=HIDDEN_SIZE,
        #     num_layers=LAYERS,
        #     dropout=DROP_RATE,
        #     batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
        #     # 为False，输入输出数据格式是(seq_len, batch, feature)，
        # )
        self.rnn = nn.RNN(
            input_size=RNN_INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True
        )
        # self.hidden_out = nn.Linear(HIDDEN_SIZE, NUM_LABEL)  # 最后一个时序的输出接一个全连接层
        # output 本来的shape 为（batch_size*sequence*hidden_size),做了一个flattern(),变成 hidden_size*sequence
        self.mlp = MLP(HIDDEN_SIZE * SEQUENCE, HIDDEN_SIZE, NUM_LABEL)
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        l_out = self.fc(x)
        # for nn.RNN, _ is h_c, for lstm, _ is (h_s,h_c)
        r_out, _ = self.rnn(l_out)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        r_out = nn.Flatten()(r_out)
        y = self.mlp(r_out)
        return y

    def predict(self, x):
        _distribution = F.softmax(self.forward(x))
        return torch.argmax(_distribution, dim=1)


class LrModel(nn.Module):
    pass

if __name__ == "__main__":
    my_model = RNN()
    inputs = torch.rand(2,120,15)
    print(my_model)

    outputs = my_model(inputs)
    print(outputs.size())
