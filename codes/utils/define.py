#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:04
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : define.py

# strategy constants
# 定义策略回测相关的常量
NO_SIGNAL = -1
LONG_OPEN = 0
LONG_CLOSE = 1
SHORT_OPEN = 2
SHORT_CLOSE = 3
LONG = 4
SHORT = 5
STOP = 6
PLT_START = 3
PLT_END = -10
TICK_SIZE = 41400
TICK = 1

# OpenVolume/CloseVolume has too many missing value, for rb in 202107,about 5% missing value
# TEST_FEATURES = ['UpdateTime', 'open_close_ratio', 'price_spread', 'aoi', 'LastPrice', 'OpenInterest',
#                  'InterestDiff', 'Turnover', 'Volume', 'wap', 'log_return',
#                  'buy_sell_spread', 'slope', 'cos', 'bs_tag', 'bs_vol', 'wap_log_return', ]
# RENAME_FEATURES = ['UpdateTime', 'open_close_ratio', 'price_spread', 'aoi', 'LastPrice',
#                    'OpenInterest', 'InterestDiff', 'Turnover', 'Volume', 'wap', 'log_return',
#                    'buy_sell_spread', 'slope', 'cos', 'bs_tag', 'bs_vol', 'label']

TEST_FEATURES = ['UpdateTime',  # factor 1:UpdateTime
                 'price_spread',  # factor 7；_curr_spread
                 'LastPrice',  # factor 4:_curr_last
                 'InterestDiff',  # factor 18,_curr_interest
                 'Volume',  # factor 17:_curr_vol
                 'wap',  # factor 8, _curr_mid
                 'log_return',  # factor 10, _log_return
                 'wap_log_return']
RENAME_FEATURES = ['UpdateTime',  # factor 1:UpdateTime
                   'price_spread',  # factor 7；_curr_spread
                   'LastPrice',  # factor 4:_curr_last
                   'InterestDiff',  # factor 18,_curr_interest
                   'Volume',  # factor 17:_curr_vol
                   'wap',  # factor 8, _curr_mid
                   'log_return',  # factor 10, _log_return
                   'label']

# model constants
# 定义模型训练相关的常量
INPUT_SIZE = len(TEST_FEATURES) - 2  # 定义输入的特征数
RNN_INPUT_SIZE = 128
HIDDEN_SIZE = 128  # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 512  # batch
EPOCH = 15  # 学习次数
LR = 0.001  # 学习率
SEC_INTERVAL = 60  # FREQ 是相应的，interval 为60， 则 FREQ为“60S"
SEQUENCE = SEC_INTERVAL * 2  # 步长,时间序列长度,因为数据频率是500ms, 一秒有两个数据记录
DROP_RATE = 0.2  # drop out概率
LAYERS = 4  # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'  # 模型名字
VALID_CRITERIER = 'cross_entropy_loss'
NUM_LABEL = 3  # 模型输出的分类类别数量： 三类： 0， 1， 2 分别为空，无，多信号
LABEL = 'label'

DT_COL_NAME = 'UpdateTime'
MISSING_THRESHOLD = 20  # 在每个时间序列样本中，如果确实的时间步数据大于该参数，则抛弃这个样本，跟SEQUENCE有关
EPOCH_DAYS = 10
TRAIN_RATIO = 0.7

cols = ["InstrumentID", "LastPrice", "OpenPrice", "HighestPrice", "LowestPrice", "Volume", "Turnover", "OpenInterest",
        "UpperLimitPrice", "LowerLimitPrice", "UpdateTime",
        "UpdateMillisec",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2", "AskPrice2", "AskVolume2",
        "BidPrice3", "BidVolume3",
        "AskPrice3", "AskVolume3", "BidPrice4", "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5",
        "AskPrice5", "AskVolume5"]

selected_cols = ['InstrumentID', 'UpdateTime', 'Turnover', 'Volume', 'LastPrice', 'AskPrice1', 'AskVolume1',
                 'BidPrice1', 'BidVolume1']

# 历史行情（csv) columns
# ['市场代码', '合约代码', '时间', '最新', '持仓', '增仓', '成交额', '成交量', '开仓', '平仓','成交类型', '方向', '买一价',
# '卖一价', '买一量', '卖一量']
tb_cols = ["Exchange", "InstrumentID", "UpdateTime", "LastPrice", "OpenInterest", "InterestDiff", "Turnover",
           "Volume", "OpenVolume", "CloseVolume", "TransactionType", "Direction", "BidPrice1", "AskPrice1",
           "BidVolume1",
           "AskVolume1"]

# skip_raw_cols = ['Exchange', 'InstrumentID', 'LastPrice', 'OpenInterest', 'InterestDiff',
#                  'Turnover', 'Volume', 'OpenVolume', 'CloseVolume', 'TransactionType', 'Direction',
#                  'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'vwap', 'vol_short', 'vol_long',
#                  'turnover_short', 'turnover_long', 'vwap_short', 'vwap_long', 'bs_vol', 'bs_vol_long',
#                  'bs_vol_short', 'bs_vol_diff', 'bs_tag', 'wap',
#                  'trenddiff', 'trend_long', 'dif', 'dea', 'turnover_ls_ratio',
#                  'vwap_ls_ratio', 'aoi', 'oi']
# remove from VIF, to be added with other calculation


# skip_raw_cols = ['Exchange', 'InstrumentID']

# skip_raw_cols_normalized = ['Exchange', 'InstrumentID', 'OpenInterest', 'InterestDiff',
#                             'Turnover', 'Volume', 'OpenVolume', 'CloseVolume', 'TransactionType', 'Direction',
#                             'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'vwap', 'vol_short', 'vol_long',
#                             'turnover_short', 'turnover_long', 'vwap_short', 'vwap_long', 'bs_vol', 'bs_vol_long',
#                             'bs_vol_short', 'bs_vol_diff', 'bs_tag']

skip_raw_cols = ['Exchange', 'InstrumentID', 'TransactionType', 'Direction', 'BidPrice1',
                 'bs_tag',
                 'LastPrice', 'InterestDiff', 'OpenInterest', 'Turnover', 'Volume', 'OpenVolume', 'CloseVolume',
                 'AskPrice1',
                 # norm value
                 'AskVolume1', 'BidVolume1', 'vwap', 'wap', 'volume_ls_diff', 'turnover_ls_diff', 'bs_vol_ls_diff',
                 'norm_turnover',
                 # norm value
                 'norm_openvolume', 'norm_closevolume', 'norm_turnover_ls_diff', ]  # norm value

# normalized_cols = [('LastPrice', 'LastPrice'), ('wap', 'LastPrice'), ('OpenInterest', 'OpenInterest'),
#                    ('vwap', 'LastPrice'), ('Volume', 'Volume'), ('BidPrice1', 'LastPrice'), ('AskPrice1', 'LastPrice')]


BASE_DIR = 'option_future_research'
RESULT_DIR = 'models'
CONF_DIR = 'conf'
FACTOR_DIR = 'features'
CONF_FILE_NAME = '.editorconfig'
STRATEGY_CONF_NAME = 'strategy.ini'
TICK_MODEL_DIR = 'tickmodels'
DAILY_MODEL_DIR = 'daily_models'
# TICK_MKT_DIR = 'C:\projects\l2mkt\FutAC_TickKZ_PanKou_Daily_'
TICK_MKT_DIR = 'D:\\future\FutAC_TickKZ_PanKou_Daily_'
FACTOR_DIR = 'features'
CACHE_DIR = 'data'
BT_DIR = 't0backtest'
daily_cache_name = 'data/future_20210101_20210804.csv'
MKT_MISSING_SKIP = 0.3

exchange_map = {'XZCE': 'zc', 'XSGE': 'sc', 'XSIE': 'ine', 'XDCE': 'dc'}

normalized_vals = {'LastPrice': 0, 'Volume': 0, 'OpenInterest': 0}
normalized_refs = [('wap', 'LastPrice')]
# normalized_refs = [('LastPrice', 'LastPrice'), ('OpenInterest', 'OpenInterest'), ('InterestDiff', 'Volume'),
#                    ('Turnover', 'Turnover'), ('Volume', 'Volume'), ('OpenVolume', 'Volume'), ('CloseVolume', 'Volume'),
#                    ('BidVolume1', 'Volume'), ('AskVolume1', 'Volume'), ('vwap', 'LastPrice'), ('wap', 'LastPrice'),
#                    ('bs_tag', 'Volume'),
#                    ('volume_ls_diff', 'Volume'), ('turnover_ls_diff', 'Turnover')]

CLF_LABEL_NAME = 'label_clf_1'
REG_LABEL_NAME = 'label_1'

# train_cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
#                 'trend_1', 'bsvol_volume_1',
#                 'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
#                 'bsvol_volume_ls_ratio']

# train_cols = ['slope']
# train_cols = ['cos', 'macd', 'aoi', 'trend_0', 'log_return_0', 'bsvol_volume_0', 'trend_1',
#               'bsvol_volume_1', 'trend_ls_ratio', 'volume_ls_ratio',  'log_return']
# train_cols = ['log_return_0', 'log_return_1', 'slope']
# train_cols = ['cos']
train_cols = ['log_return', 'log_return_0', 'wap_log_return']
