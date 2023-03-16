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

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])
c = torch.tensor([6])
output = pack_sequence([a, b, c])
print(output)


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