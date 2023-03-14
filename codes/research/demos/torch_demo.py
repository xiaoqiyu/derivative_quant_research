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
import numpy as np

data = [[1,2],[3,4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
