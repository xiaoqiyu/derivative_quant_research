#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:02
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Account.py

class Account(object):
    def __init__(self, init_margin=100000, risk_ratio=0.3, init_capital=None):
        self.transaction = list()
        self.fee = 0.0
        # self.risk_ratio = [0.0]
        self.occupied_margin = init_margin
        self.risk_ratio = risk_ratio
        self.capital = init_capital or self.occupied_margin / self.risk_ratio
        self.live_risk_ratio = self.risk_ratio
        # self.market_value = init_cash
        # self.available_margin = list()
        # self.trade_market_values = [init_cash]
        # self.settle_market_values = [init_cash]

    def add_transaction(self, val=None):
        if val is None:
            val = []
        self.transaction.append(val)

    def cache_transaction(self):
        pass

    def update_account(self, transaction_fee, update_margin):
        self.fee += transaction_fee
        self.occupied_margin = update_margin
        self.live_risk_ratio = self.occupied_margin / (self.capital/self.risk_ratio)
        return self.fee, self.occupied_margin, self.live_risk_ratio

    # def update_risk_ratio(self, val):
    #     self.risk_ratio.append(val)

    # def update_occupied_margin(self, val):
    #     self.occupied_margin.append(val)

    # def update_market_value(self, trade_val, settle_val, fee):
    #     _trade_val = self.trade_market_values[-1]
    #     # print("old mkt val", _trade_val)
    #     self.trade_market_values.append(_trade_val - fee + trade_val)
    #     # print("new mkt val", self.trade_market_values[-1])
    #     _settle_val = self.settle_market_values[-1]
    #     # print("old mkt val 1", _settle_val)
    #     self.settle_market_values.append(_settle_val - fee + settle_val)
    #     # print("new mkt val 1", self.settle_market_values[-1])

    def update_available_margin(self, val):
        self.available_margin.append(val)

    def get_trade_market_values(self):
        return self.trade_market_values

    def get_settle_market_values(self):
        return self.settle_market_values
