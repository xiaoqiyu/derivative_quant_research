#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 14:50
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : torch_script_demo.py


import torch
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../../..")))
sys.path.append(_base_dir)

from codes.utils.define import *


# min demo
# class MyModule(torch.jit.ScriptModule):
#     def __init__(self, N, M):
#         super(MyModule, self).__init__()
#         self.weight = torch.nn.Parameter(torch.rand(N, M))
#
#     @torch.jit.script_method
#     def forward(self, input):
#         if input.sum() > 0:
#             output = self.weight.mv(input)
#         else:
#             output = self.weight + input
#         return output
#
#
# my_model = MyModule(10, 20)
# print(my_model)
# input = torch.rand(20)
# output = my_model(input)
# print(output)
# my_model.save("model.pt")

class MLP(torch.jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROP_RATE)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)

    @torch.jit.script_method
    def forward(self, X):
        print('shape x', X.size())
        return self.fc2(self.relu(self.fc1(X)))

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)


# 定义LSTM的结构
class RNN(torch.jit.ScriptModule):
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

    @torch.jit.script_method
    def forward(self, x):
        l_out = self.fc(x)
        r_out, _ = self.rnn(l_out)
        r_out = r_out.flatten(start_dim=1, end_dim=-1)
        # r_out = nn.Flatten()(r_out)
        # return r_out
        y = self.mlp(r_out)
        return y

    @torch.jit.script_method
    def predict(self, x):
        _distribution = F.softmax(self.forward(x))
        return torch.argmax(_distribution, dim=1)


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
rnn.train()
b_x = torch.rand(2, 120, 15)
prediction = rnn(b_x)
loss = loss_func(prediction, torch.ones(2, dtype=torch.long))
optimizer.zero_grad()  # clear gradients for this training step
loss.backward()  # backpropagation, compute gradients
optimizer.step()  # apply gradients
print(loss.item())
#
rnn.eval()
inputs = torch.rand(2, 120, 15)
print(rnn)
outputs = rnn(inputs)
pred = rnn.predict(inputs)
print(outputs.size())
print(pred)
# input = torch.rand(20)
# output = my_model(input)
# print(output)
rnn.save("rnn.pt")
