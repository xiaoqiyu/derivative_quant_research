#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 14:19
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : sta_reference.py

import uqer
from uqer import DataAPI

uqer_client = uqer.Client(token="6aa0df8d4eec296e0c25fac407b332449112aad6c717b1ada315560e9aa0a311")


def fetch_data(start_date='', end_date='', product_id=[]):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
   