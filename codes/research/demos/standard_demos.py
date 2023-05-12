#!/user/bin/env python
#coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : standard_demos.py
#@time   : 2023-05-13 01:05:54
'''

from sklearn.preprocessing import StandardScaler

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))


