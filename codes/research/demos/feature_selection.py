#!/user/bin/env python
#coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : feature_selection.py
#@time   : 2023-09-27 12:07:50
'''
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

x, y = load_digits(return_X_y=True)
print(x.shape)
x_new = SelectKBest(chi2, k=20).fit_transform(x,y)
print(x_new.shape)
