#!/user/bin/env python
# coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : pd_demo.py
#@time   : 2023-11-16 09:35:03
'''
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('G://features//000001_202101.csv')
# df['time'] = pd.to_datetime(df['start_time'])
# df['time'] = df['time'] + pd.Timedelta(days=1)
# df['time'].max() - df['time'].min()

df['date'] = [item.split(' ')[0] for item in df['start_time']]
df1 = df.loc[:10, :]
df2 = df.iloc[10:20, :]
# df = df[(df.Q>0)| (df.MI >0)&(df.ADV>0)]
# df[df['date'].isin(['20210104'])]
# df[df['date'] =='20210104']
# df[df['date'].str.contains('20210104')]
# df[df['date'].str.startswith('20210104')]
# df[df['date'].str.endswith('20210104')]
# df[2:8:2]
# df.loc[:,'ADV']
# df.iloc[]只能使用整数索引，不能使用标签索引，通过整数索引切边进行筛选时，前闭后开
# df.groupby('date').describe()
# df.groupby('date').agg(np.sum)
# df.groupby('date').max()

# ret = pd.cut(df['ADV'], bins=5, retbins=True)

print(df.shape)
