#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 9:55
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : torch_demo.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

# data = [[1,2],[3,4]]
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# LSTM demo
# sequence=5, batch=3, num_layer=2, input_size=20, hidden_size=20
# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5])
# c = torch.tensor([6])
# output = pack_sequence([a, b, c])
# print(output)


# import torch
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#
#
# # Sequences
# a = torch.tensor([1, 2])
# b = torch.tensor([3, 4, 5])
# c = torch.tensor([6, 7, 8, 9])
# print('a:', a)
# print('b:', b)
# print('c:', c)
#
# # Settings
# seq_lens = [len(a), len(b), len(c)]
# max_len = max(seq_lens)
#
#
# # Zero padding
# a = F.pad(a, (0, max_len-len(a)))
# b = F.pad(b, (0, max_len-len(b)))
# c = F.pad(c, (0, max_len-len(c)))
#
#
# # Merge the sequences
# seq = torch.cat((a, b, c), 0).view(-1, max_len)
# print('Sequence:', seq)
#
#
# # Pack
# packed_seq = pack_padded_sequence(seq, seq_lens, batch_first=True, enforce_sorted=False)
# print('Pack:', packed_seq)
#
#
# # Unpack
# unpacked_seq, unpacked_lens = pad_packed_sequence(packed_seq, batch_first=True)
# print('Unpack:', unpacked_seq)
# print('length:', unpacked_lens)
#
#
# # Reduction
# a = unpacked_seq[0][:unpacked_lens[0]]
# b = unpacked_seq[1][:unpacked_lens[1]]
# c = unpacked_seq[2][:unpacked_lens[2]]
# print('Recutions:')
# print('a:', a)
# print('b:', b)
# print('c:', c)

# batch_size = 3
# max_len = 6
# embedding_size = 8
# hidden_size = 16
# vocal_size = 5
#
# input_seq = [[3, 5, 12, 7, 2], [4, 11, 14], [18, 7, 3, 8, 5, 4]]
# lengths = [5, 3, 6]
# embedding = torch.nn.Embedding(vocal_size, embedding_size, padding_idx=0)
# gru = torch.nn.GRU(2, hidden_size)
#
# input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
# lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
#
# PAD_TOKEN = 0
#
#
# def pad_seq(seq, seq_len, max_len):
#     seq = seq
#     seq += [PAD_TOKEN for _ in range(max_len - seq_len)]
#     return seq
#
#
# pad_seqs = []
# for i, j in zip(input_seq, lengths):
#     pad_seqs.append(pad_seq(i, len(i), max_len))
#
# pad_seqs = [[[1,2],[3,4],[6,7]],
#             [[2,3,],[4,5],[0,0]]]
# pad_seqs = torch.tensor(pad_seqs)
# print(pad_seqs.shape)
# # embeded = embedding(pad_seqs)
# lengths = [3,2]
# pack = torch.nn.utils.rnn.pack_padded_sequence(pad_seqs, lengths, batch_first=True)
# print(pack.data)
#
# pad_outputs, _=gru(pack)
# print(pad_outputs)
#
# # unpack=torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
# # print(unpack)


# examples for pack_padded_sequence
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import utils as nn_utils
#
# batch_size = 2
# max_lenght = 2
# hidden_size = 2
# n_layers = 1
# tensor_in = torch.FloatTensor([[1, 2, 3], [1, 0, 0]]).resize_(2, 3, 1)
# tensor_in = Variable(tensor_in)
# seq_lengths = [3, 1]
#
# pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
# print('packed', pack)
#
# rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
# h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
#
# out, _ = rnn(pack, h0)
# print('out', out)
#
# unpacked = nn_utils.rnn.pad_packed_sequence(out)
# print('unpacked', unpacked)

# import torch
# import torchvision
#
# # An instance of your model.
# model = torchvision.models.resnet18()
#
# # An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)
#
# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# output = traced_script_module(torch.rand(1, 3, 224, 224))
# traced_script_module.save("traced_resnet_model.pt")
#
# print(output)


# class MyModule(torch.nn.Module):
#     def __init__(self, N, M):
#         super(MyModule, self).__init__()
#         self.weight = torch.nn.Parameter(torch.rand(N, M))
#
#     def forward(self, input):
#         if input.sum() > 0:
#           output = self.weight.mv(input)
#         else:
#           output = self.weight + input
#         return output
#
# my_module = MyModule(10,20)
# sm = torch.jit.script(my_module)

import torch
class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method
    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output


my_model = MyModule(10, 20)
print(my_model)
input = torch.rand(20)
output = my_model(input)
print(output)
# my_model.save("model.pt")

