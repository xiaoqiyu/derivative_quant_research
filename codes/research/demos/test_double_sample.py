#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_double_sample.py
# @Author: kiki
# @Date  : 2023/12/29
# @Desc  :
# @Contact : rpyxqi@gmail.com

import statsmodels.api as sm
import numpy as np

data = sm.datasets.get_rdataset("Duncan", "carData")
y = data.data['income']
y = np.vstack([y, y]).reshape(-1, 1)
x = data.data['education']
x = np.vstack([x, x]).reshape(-1, 1)
x = sm.add_constant(x)

print(x.size, y.size)
model = sm.OLS(y, x)
ret = model.fit()
print(ret.summary())
