#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : draft.py
# @Author: kiki
# @Date  : 2023/12/30
# @Desc  :
# @Contact : rpyxqi@gmail.com

import re

ret = re.findall(r'\w{3}', 'ab12|adb')
import matplotlib.pyplot as plt
plt.sa

print(ret)

import pandas as pd
import numpy as np

_array_date = pd.date_range(start=pd.to_datetime('7/1/2023'), end=pd.to_datetime('12/1/2023'))
n = _array_date.size
df = pd.DataFrame({'date': _array_date, 'a': np.random.random(n), 'b': np.random.random(n)})
# print(df)
print(df[df.a.isnull()])
# print(df.notnull())

# print(df[(df.a>0)&(df.date>pd.to_datetime('10/1/2023'))])
# print(df[df['date'].isin([pd.to_datetime('10/2/2023'), pd.to_datetime('10/5/2023')])])
# print(df[2:5:2])
# print(df.groupby('date').describe())
# print(df.groupby('date').agg([np.sum, np.mean]))
# print(df.groupby('date').max())
# print(df.groupby('date')['a'].agg([np.mean]))
print(df[['a', 'b']].rolling(5).apply(lambda x: list(x)[-1]))

from typing import Optional


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

a =input("input:")
print(a)
a=1
assert  a in[0,1]