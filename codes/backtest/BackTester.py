#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:05
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : BackTester.py

import os
import sys

_base_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(_base_dir)

import time
from codes.backtest.Factor import Factor
from codes.backtest.Position import Position
from codes.backtest.Account import Account
from codes.strategy.MinSignal import MinSignal
from codes.strategy.ClfSignal import ClfSignal
from codes.research.data_process.data_fetcher import DataFetcher
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from codes.utils import helper
from codes.utils.define import *
from codes.utils.utils import *
from codes.utils.logger import Logger
from copy import deepcopy
import configparser

uqer_client = uqer.Client(token="e4ebad68acaaa94195c29ec63d67b77244e60e70f67a869585e14a7fe3eb8934")
data_fetcher = DataFetcher(uqer_client)
_log_path = os.path.join(_base_dir, 'data\logs\{0}'.format(os.path.split(__file__)[-1].strip('.py')))
logger = Logger(_log_path, 'INFO', __name__).get_log()


def get_fill_ret(order=[], tick=1, mkt=[]):
    _order_type, _price, _lot = order
    bid_price1, ask_price1, bid_vol1, ask_vol1 = mkt[-5:-1]
    _last_price = mkt[3]
    if _order_type == LONG:
        if _lot <= ask_vol1 and (not _price or _price >= ask_price1):
            return [ask_price1, _lot]
        else:
            return [0, 0]
    if _order_type == SHORT:
        if _lot <= bid_vol1 and (not _price or _price <= bid_price1):
            return [bid_price1, _lot]
        else:
            return [0, 0]
    return [0, 0]


def stop_profit_loss(risk_input: dict = {}, risk_conf: dict = {}) -> tuple:
    '''
     # the range in the past 1 min is laess than 2(N) ticks, stop profit/loss will be downgraded,then tend to close the position
    #easier; if not, then the oppisite side, try to earn more in the order, at the same time, it may lose more, so it should
    #improve the strategy/signal
    :param risk_input:
    :param risk_conf:
    :return:
    '''
    _last_price = risk_input.get('last_price').get_lst()[-120:]
    _range = max(_last_price) - min(_last_price)
    _stop_lower = risk_conf.get('stop_lower')
    _stop_upper = risk_conf.get('stop_upper')
    if _range < 2:
        return _stop_lower, _stop_lower
    else:
        return _stop_upper, _stop_upper


def backtest_quick(data_fetcher: DataFetcher = None, product_id: str = 'rb', trade_date: str = '2021-07-01',
                   options: dict = {}) -> tuple:
    # Init contract static and daily data
    data_fetcher.get_instrument_mkt(product_ids=[product_id], start_date=trade_date, end_date=trade_date)
    _instruments = list(set(data_fetcher.instrument_cache['ticker']))
    data_fetcher.get_instrument_contract(instrument_ids=_instruments, product_ids=[product_id])
    instrument_id = list(data_fetcher.instrument_cache[data_fetcher.instrument_cache.mainCon == 1]['ticker'])[0]
    tick_mkt = data_fetcher.load_tick_data(start_date=trade_date, end_date=trade_date, instrument_ids=_instruments,
                                           main_con_flag=1, filter_start_time=None, filter_end_time=None,
                                           if_filter=True)
    multiplier = list(data_fetcher.contract_cache[data_fetcher.contract_cache.ticker == instrument_id]['contMultNum'])[
        0]
    tick_mkt = tick_mkt.set_index('InstrumentID').join(
        data_fetcher.contract_cache[['ticker', 'contMultNum']].set_index('ticker')).reset_index()


    # init factor, position account signal
    factor = Factor(product_id=product_id, instrument_id=instrument_id, trade_date=trade_date)
    position = Position()
    account = Account()
    signal = MinSignal(factor=factor, position=position, instrument_id=instrument_id, trade_date=trade_date,
                       product_id=product_id)

    # Init backtest params
    open_fee = float(options.get('open_fee') or 3.0)
    close_fee = float(options.get('close_fee') or 2.0)
    fee = open_fee + close_fee
    # start_timestamp = options.get('start_timestamp') or '09:05:00'
    # start_datetime = '{0} {1}'.format(trade_date, start_timestamp)
    # end_timestamp = options.get('end_timestamp') or '22:50:00'
    # end_datetime = '{0} {1}'.format(trade_date, end_timestamp)
    # delay_sec = int(options.get('delay_sec')) or 5
    total_return = 0.0
    close_price = 0.0  # not the true close price, to close the position
    _text_lst = ['long_open', 'long_close', 'short_open', 'short_close']
    update_factor_time = 0.0
    get_signal_time = 0.0
    options.update({'instrument_id': instrument_id})
    options.update({'multiplier': multiplier})
    options.update({'trade_date': trade_date})
    options.update({'stop_profit': 10.0})
    options.update({'stop_loss': 10.0})
    options.update({'risk_duration': 30})
    options.update({'vol_limit': 5})

    # 逐tick backtest开始
    values = tick_mkt.values
    total_tick_num = len(values)
    ab_cnt = 0
    close_timestamp = None
    for idx, item in enumerate(values):
        _last = item[3]
        _update_time = item[2]
        _mul_num = item[-1]
        # TODO remove hardcode for the threshold
        if float(ab_cnt / total_tick_num) > 0.25:
            logger.warn("Error for mkt/tick, ignore the trade date:{0}".format(trade_date))
            return (total_return, account.fee, 0, account.transaction)
        try:
            curr_factor = factor.update_factor(item, idx=idx, multiplier=item[-1], lag_long=SEQUENCE,
                                               lag_short=SEC_INTERVAL)
        except Exception as ex:
            # logger.warn(
            #     "Ignore current item, Update factor error for item:{0}, last_factor:{1}, and error:{2} ".format(item,
            #                                                                                                     factor.last_factor,
            #                                                                                                     ex))
            ab_cnt += 1
            continue

        close_price = _last
        close_timestamp = _update_time
        # options.update({'factor': curr_factor})

        # Get signal
        _signal = signal(params=options)
        if _signal.signal_type == LONG_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[LONG, _signal.price, _signal.vol], tick=1, mkt=item)
            logger.info("Long Open with update time:{0},filled price:{1}, filled lot:{2}------".format(_update_time,
                                                                                                       _fill_price,
                                                                                                       _fill_lot))
            if _fill_lot > 0:
                dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_OPEN], _last, _fill_price, _fill_lot, _fill_price,
                     open_fee * _fill_lot,
                     _update_time.split(' ')[-1],
                     _update_time.split(' ')[-1],
                     0.0, 0.0])
                position.update_position(instrument_id=instrument_id, long_short=LONG, price=_fill_price,
                                         timestamp=_update_time, vol=_fill_lot, order_type=LONG_OPEN)
                account.update_fee(open_fee * _fill_lot)

        elif _signal.signal_type == SHORT_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[SHORT, _signal.price, _signal.vol], tick=1, mkt=item)
            logger.info("Short Open with update time:{0},filled price:{1}, filled lot:{2}------".format(_update_time,
                                                                                                        _fill_price,
                                                                                                        _fill_lot))
            if _fill_lot > 0:
                dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                account.add_transaction(
                    [idx, instrument_id, _text_lst[SHORT_OPEN], _last, _fill_price, _fill_lot, _fill_price,
                     open_fee * _fill_lot,
                     _update_time.split(' ')[-1],
                     _update_time.split(' ')[-1],
                     0.0, 0.0])
                position.update_position(instrument_id=instrument_id, long_short=SHORT, price=_fill_price,
                                         timestamp=_update_time,
                                         vol=_fill_lot, order_type=SHORT_OPEN)
                account.update_fee(open_fee * _fill_lot)

        elif _signal.signal_type == LONG_CLOSE:
            logger.info("Long Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, define.LONG)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[SHORT, _signal.price, _signal.vol], tick=1,
                                                      mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    curr_return = (((_fill_price - _pos[1]) * _mul_num) - fee) * _fill_lot
                    total_return += curr_return
                    account.add_transaction(
                        [idx, instrument_id, _text_lst[LONG_CLOSE], _last, _fill_price, _fill_lot, _pos[1],
                         close_fee * _fill_lot,
                         _pos[2].split(' ')[-1],
                         _update_time.split(' ')[-1],
                         holding_time,
                         curr_return])
                    position.update_position(instrument_id=instrument_id, long_short=SHORT, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=LONG_CLOSE)
                    account.update_fee(close_fee * _fill_lot)
        elif _signal.signal_type == SHORT_CLOSE:
            logger.info("Short Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, SHORT)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[LONG, _signal.price, _signal.vol], tick=1, mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    curr_return = ((_pos[1] - _fill_price) * _mul_num - fee) * _fill_lot
                    total_return += curr_return
                    account.add_transaction(
                        [idx, instrument_id, _text_lst[SHORT_CLOSE], _last, _fill_price, _fill_lot, _pos[1],
                         close_fee * _fill_lot,
                         _pos[2].split(' ')[-1],
                         _update_time,
                         holding_time,
                         curr_return])
                    position.update_position(instrument_id=instrument_id, long_short=LONG, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=SHORT_CLOSE)
                    account.update_fee(close_fee * _fill_lot)
        else:  # NO_SIGNAL
            pass

    logger.info("Factor/tick error num:{0} out of:{1}".format(ab_cnt, total_tick_num))
    _pos = position.get_position(instrument_id)
    total_return_risk = total_return
    total_risk = 0.0

    # 收盘强平
    if _pos:
        _tmp_pos = deepcopy(_pos)
        for item in _tmp_pos:
            if item[0] == LONG:
                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick down
                _return = ((close_price - define.TICK - item[1]) * item[-1] - fee) * _mul_num
                total_return += _return
                total_risk += item[1] * item[-1]
                logger.info('final long close with return:{0},total return after:{1} for trade_date:{2}'.format(_return,
                                                                                                                total_return,
                                                                                                                trade_date))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_CLOSE], close_price, close_price - define.TICK, item[-1],
                     item[1],
                     close_fee * item[-1],
                     item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_fee * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.LONG_CLOSE)
            else:

                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick up, tick hardcode
                _return = ((item[1] - close_price - define.TICK) * _mul_num - fee) * item[-1]
                total_risk += item[1] * item[-1]
                total_return += _return
                logger.info('final short close with return:{0},total return after:{1}'.format(_return, total_return))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.SHORT_CLOSE], close_price,
                     close_price + define.TICK, item[-1], item[1],
                     close_fee * item[-1], item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_fee * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.LONG, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.SHORT_CLOSE)
    # 结果统计
    long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open = 0, 0, 0, 0, 0, 0
    total_holding_time = 0.0
    max_holding_time = -np.inf
    min_holding_time = np.inf
    for item in account.transaction:
        if item[2] == 'long_open':
            long_open += 1
        elif item[2] == 'short_open':
            short_open += 1
        elif item[2] == 'long_close':
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:  # short close
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1
    average_holding_time = total_holding_time / (long_open + short_open) if long_open + short_open > 0 else 0.0
    precision = (correct_long_open + correct_short_open) / (
            long_open + short_open) if long_open + short_open > 0 else 0.0
    print("******************back test models for date:{0}*********************".format(trade_date))
    print('trade date', trade_date)
    print(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    print('total return:', total_return)
    print('total fee:', account.fee)
    print('precision:', precision)
    print('update factor time:', update_factor_time)
    print('get signal time:', get_signal_time)
    print("average_holding_time:", average_holding_time)
    print("max_holding_time:", max_holding_time)
    print("min_holding_time:", min_holding_time)
    print("******************back test models for date:{0}*********************".format(trade_date))

    _backtest_path = os.path.join(_base_dir, 'data\\backtest\\backtest_{0}.csv'.format(product_id))
    try:
        result_df = pd.read_csv(_backtest_path)
    except Exception as ex:
        result_df = pd.DataFrame(
            {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return_final': [],
             'total_return_unclose': [],
             'total_fee': [],
             'unclosed_value': [], 'precision': [], 'long_open': [], 'short_open': [],
             'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
             'average_holding_time': [], 'max_holding_time': [], 'min_holding_time': []
             })

    result_df = result_df.append(
        {'trade_date': trade_date, 'product_id': product_id, 'instrument_id': instrument_id,
         'total_return_final': total_return, 'total_return_unclose': total_return_risk,
         'total_fee': account.fee,
         'unclosed_value': total_risk, 'precision': precision,
         'long_open': long_open, 'short_open': short_open, 'correct_long_open': correct_long_open,
         'wrong_long_open': wrong_long_open, 'correct_short_open': correct_short_open,
         'wrong_short_open': wrong_short_open, 'average_holding_time': average_holding_time,
         'max_holding_time': max_holding_time, 'min_holding_time': min_holding_time
         }, ignore_index=True)

    result_df.to_csv(_backtest_path, index=False)

    # 画图，开平仓信号，问题处理一下
    # if plot_mkt:
    #     _idx_lst = list(range(len(factor.last_price.get_lst())))
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.last_price[define.PLT_START:define.PLT_END])
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.vwap[define.PLT_START:define.PLT_END])
    #     ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.turning[define.PLT_START:define.PLT_END])
    #     logger.info(np.array(factor.last_price).std())
    #     ax1.grid(True)
    #     ax1.set_title('{0}_{1}'.format(instrument_id, trade_date))
    #     xtick_labels = [item[:-3] for item in factor.update_time]
    #     ax1.set_xticks(_idx_lst[::3600])
    #     min_lst = []
    #     ax1.set_xticklabels(xtick_labels[::3600])
    #     # ax1.set_xticks(factor.update_time[::3600])
    #     # ax1.set_xticks(x_idx, xtick_labels, rotation=60, FontSize=6)
    #     for item in account.transaction:
    #         _t_lst = ['lo', 'lc', 'so', 'sc']
    #         ax1.text(item[0], item[3], s='{0}'.format(item[2]))
    #
    #     ax2 = ax1.twinx()
    #     ax2.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.trend_short[define.PLT_START:define.PLT_END], 'r')
    #     plt.show()
    ret = (total_return, account.fee, precision, account.transaction)
    return ret


def backtesting(product_id: str = 'm', trade_date: str = '20210401', signal_name: str = 'RegSignal',
                result_fname_digest: str = '', options: dict = {},
                plot_mkt: bool = True) -> tuple:
    '''

    :param product_id: product id to be tested, lower case
    :param trade_date: backtest trade, 'yyyymmdd'
    :param signal_name: signal type
    :param result_fname_digest:  file name to data test result, which the digest of back test params
    :param options: other params to be included in the back test
    :param plot_mkt: whether to plot and save jpg of the mkt and trade signal
    :return: (total return, total fee, precision, list of transaction)
    '''

    # load back test config
    backtesting_config = ''
    if not options:
        try:
            _strategy_conf = get_path([CONF_DIR, STRATEGY_CONF_NAME])
            config = configparser.ConfigParser()
            config.read(_strategy_conf)
            signal_lst = []
            for k, v in config['strategy'].items():
                if k == 'signal_names':
                    options.update({k: v})
                    signal_lst = v.split(',')
                elif k == 'start_timestamp' or k == 'end_timestamp':
                    options.update({k: v})
                else:
                    options.update({k: float(v)})
                for sig_name in signal_lst:
                    options[sig_name] = {}
                    for k1, v1 in config[sig_name].items():
                        options[sig_name].update({k1: float(v1)})
        except Exception as ex:
            logger.info(ex)

    for key, value in options.items():
        backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)

    # get contract and daily data
    instrument_id_df = get_instrument_ids(start_date=trade_date, end_date=trade_date, product_id=product_id)
    instrument_id, trade_date, exchange_cd = instrument_id_df.values[0]
    _mul_num = get_mul_num(instrument_id=instrument_id)

    # Load depth markets
    _tick_mkt_path = os.path.join(define.TICK_MKT_DIR, define.exchange_map.get(exchange_cd),
                                  '{0}_{1}.csv'.format(instrument_id, trade_date.replace('-', '')))
    tick_mkt = pd.read_csv(_tick_mkt_path, encoding='gbk')
    tick_mkt.columns = define.tb_cols
    logging.info('trade_date:{0}, instrument_id:{1}, shape:{2}'.format(trade_date, instrument_id, tick_mkt.shape))
    values = tick_mkt.values

    # init factor, position account signal
    factor = Factor(product_id=product_id, instrument_id=instrument_id, trade_date=trade_date)
    position = Position()
    account = Account()
    signal = ClfSignal(factor=factor, position=position, instrument_id=instrument_id,
                       trade_date=trade_date, product_id=product_id)

    # Init backtest params
    open_fee = float(options.get('open_fee')) or 3.0
    close_t0_fee = float(options.get('close_t0_fee')) or 0.0
    fee = open_fee + close_t0_fee
    start_timestamp = options.get('start_timestamp') or '09:05:00'
    start_datetime = '{0} {1}'.format(trade_date, start_timestamp)
    end_timestamp = options.get('end_timestamp') or '22:50:00'
    # end_datetime = '{0} {1}'.format(trade_date, end_timestamp)
    # delay_sec = int(options.get('delay_sec')) or 5
    total_return = 0.0
    close_price = 0.0  # not the true close price, to close the position
    _text_lst = ['long_open', 'long_close', 'short_open', 'short_close']
    update_factor_time = 0.0
    get_signal_time = 0.0
    signal_delay = 0
    conf_signal_delay = int(options.get('signal_delay'))

    # robust handling, skip dates with records missing more than threshold(e.g. 0.3 here)
    if len(values) < define.TICK_SIZE * define.MKT_MISSING_SKIP:
        logging.warning("miss mkt for trade_date:{0} with len:{1} and tick size:{2}".format(trade_date, len(values),
                                                                                            define.TICK_SIZE
                                                                                            ))
        return
    _size = len(values)
    logger.info('size is:', _size)

    dt_last_trans_time = None
    try:
        dt_last_trans_time = datetime.strptime(start_datetime, '%H:%M:%S')
    except Exception as ex:
        dt_last_trans_time = datetime.strptime(start_datetime.split('.')[0], '%Y-%m-%d %H:%M:%S')
    else:
        pass

    signal_num = 0
    logger.info("**************Start Ticks with size:{0} and trade_date:{1}****************".format(_size, trade_date))
    for idx, item in enumerate(values):
        _last = item[3]
        _update_time = item[2]
        start_ts = time.time()
        curr_factor = factor.update_factor(item, idx=idx, multiplier=_mul_num,
                                           lag_long=int(options.get('long_windows')),
                                           lag_short=int(options.get('short_windows')))
        end_ts = time.time()
        update_factor_time += (end_ts - start_ts)

        # FIXME double check
        if not is_trade(start_timestamp, end_timestamp, _update_time):
            # logger.info(_update_time, 'not trade time--------------')
            continue

        if signal_delay % conf_signal_delay != 0:
            signal_delay += 1
            continue
        close_price = _last
        close_timestamp = _update_time
        start_ts = time.time()

        options.update({'instrument_id': instrument_id})
        options.update({'multiplier': _mul_num})
        options.update({'fee': fee})
        options.update({'factor': curr_factor})
        options.update({'trade_date': trade_date})
        s1 = time.time()
        options.update({'stop_lower': 2.0})
        options.update({'stop_upper': 20.0})
        stop_profit, stop_loss = stop_profit_loss({'last_price': factor.last_price}, options)
        options.update({'stop_profit': stop_profit, 'stop_loss': stop_loss})

        # Get signal
        _signal = signal(params=options)
        if _signal.signal_type != define.NO_SIGNAL:
            signal_num += 1
            logger.info('Get Signal=>', _signal.signal_type, 'vol=>', _signal.vol, 'price=>', _signal.price,
                        'direction=>',
                        _signal.direction, 'update time=>', _update_time)

        e1 = time.time()
        end_ts = time.time()
        get_signal_time += (end_ts - start_ts)
        _pos = position.get_position(instrument_id)
        _last_trans_time = _pos[-1][2] if _pos else start_datetime
        dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
        _trans_gap_time = (dt_curr_time - dt_last_trans_time).seconds
        s2 = time.time()
        # handle signal, update account and position
        if _signal.signal_type == define.LONG_OPEN:
            logger.info("Long Open with update time:{0}------".format(_update_time))
            _fill_price, _fill_lot = get_fill_ret(order=[define.LONG, _signal.price, _signal.vol], tick=1, mkt=item)
            if _fill_lot > 0:
                dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_OPEN], _last, _fill_price, _fill_lot, _fill_price,
                     open_fee * _fill_lot,
                     _update_time.split(' ')[-1],
                     _update_time.split(' ')[-1],
                     0.0, 0.0])
                position.update_position(instrument_id=instrument_id, long_short=define.LONG, price=_fill_price,
                                         timestamp=_update_time, vol=_fill_lot, order_type=define.LONG_OPEN)
                account.update_fee(open_fee * _fill_lot)

        elif _signal.signal_type == define.SHORT_OPEN:
            logger.info("Short Open with update time:{0}------".format(_update_time))
            _fill_price, _fill_lot = get_fill_ret(order=[define.SHORT, _signal.price, _signal.vol], tick=1, mkt=item)
            if _fill_lot > 0:
                dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.SHORT_OPEN], _last, _fill_price, _fill_lot, _fill_price,
                     open_fee * _fill_lot,
                     _update_time.split(' ')[-1],
                     _update_time.split(' ')[-1],
                     0.0, 0.0])
                position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=_fill_price,
                                         timestamp=_update_time,
                                         vol=_fill_lot, order_type=define.SHORT_OPEN)
                account.update_fee(open_fee * _fill_lot)

        elif _signal.signal_type == define.LONG_CLOSE:
            logger.info("Long Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, define.LONG)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[define.SHORT, _signal.price, _signal.vol], tick=1,
                                                      mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    # logger.info('transaction time=>', dt_last_trans_time)
                    curr_return = (((_fill_price - _pos[1]) * _mul_num) - fee) * _fill_lot
                    total_return += curr_return

                    account.add_transaction(
                        [idx, instrument_id, _text_lst[define.LONG_CLOSE], _last, _fill_price, _fill_lot, _pos[1],
                         close_t0_fee * _fill_lot,
                         _pos[2].split(' ')[-1],
                         _update_time.split(' ')[-1],
                         holding_time,
                         curr_return])
                    position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=define.LONG_CLOSE)
                    account.update_fee(close_t0_fee * _fill_lot)
        elif _signal.signal_type == define.SHORT_CLOSE:
            logger.info("Short Close with update time:{0}------".format(_update_time))
            _pos = position.get_position_side(instrument_id, define.SHORT)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[define.LONG, _signal.price, _signal.vol], tick=1, mkt=item)
                if _fill_lot > 0:
                    dt_last_trans_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    curr_return = ((_pos[1] - _fill_price) * _mul_num - fee) * _fill_lot
                    total_return += curr_return
                    account.add_transaction(
                        [idx, instrument_id, _text_lst[define.SHORT_CLOSE], _last, _fill_price, _fill_lot, _pos[1],
                         close_t0_fee * _fill_lot,
                         _pos[2].split(' ')[-1],
                         _update_time,
                         holding_time,
                         curr_return])
                    position.update_position(instrument_id=instrument_id, long_short=define.LONG, price=_fill_price,
                                             timestamp=_update_time, vol=_fill_lot, order_type=define.SHORT_CLOSE)
                    account.update_fee(close_t0_fee * _fill_lot)
        else:  # NO_SIGNAL
            pass
        e2 = time.time()
        # logger.info('handle signal time:', idx, idx/_size, e2-s2)
        signal_delay += 1
    _pos = position.get_position(instrument_id)
    total_return_risk = total_return
    total_risk = 0.0
    logger.info("***************complete tick with signal num:{0} *************************".format(signal_num))
    if _pos:
        _tmp_pos = deepcopy(_pos)
        for item in _tmp_pos:
            if item[0] == define.LONG:
                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick down
                _return = ((close_price - define.TICK - item[1]) * item[-1] - fee) * _mul_num
                total_return += _return
                total_risk += item[1] * item[-1]
                logger.info('final long close with return:{0},total return after:{1} for trade_date:{2}'.format(_return,
                                                                                                                total_return,
                                                                                                                trade_date))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_CLOSE], close_price, close_price - define.TICK, item[-1],
                     item[1],
                     close_t0_fee * item[-1],
                     item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_t0_fee * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.LONG_CLOSE)
            else:

                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick up, tick hardcode
                _return = ((item[1] - close_price - define.TICK) * _mul_num - fee) * item[-1]
                total_risk += item[1] * item[-1]
                total_return += _return
                logger.info('final short close with return:{0},total return after:{1}'.format(_return, total_return))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.SHORT_CLOSE], close_price,
                     close_price + define.TICK, item[-1], item[1],
                     close_t0_fee * item[-1], item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_t0_fee * item[-1])
                position.update_position(instrument_id=instrument_id, long_short=define.LONG, price=close_price,
                                         timestamp=close_timestamp,
                                         vol=item[-1], order_type=define.SHORT_CLOSE)

    if options.get('cache_factor') == '1':
        logging.info('Start data factor')
        factor.cache_factor()
        logging.info('Complete data factor')
    else:
        logging.info('Stip data factor')

    if plot_mkt:
        _idx_lst = list(range(len(factor.last_price)))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.last_price[define.PLT_START:define.PLT_END])
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.vwap[define.PLT_START:define.PLT_END])
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.turning[define.PLT_START:define.PLT_END])
        logger.info(np.array(factor.last_price).std())
        ax1.grid(True)
        ax1.set_title('{0}_{1}'.format(instrument_id, trade_date))
        xtick_labels = [item[:-3] for item in factor.update_time]
        ax1.set_xticks(_idx_lst[::3600])
        min_lst = []
        ax1.set_xticklabels(xtick_labels[::3600])
        # ax1.set_xticks(factor.update_time[::3600])
        # ax1.set_xticks(x_idx, xtick_labels, rotation=60, FontSize=6)
        for item in account.transaction:
            _t_lst = ['lo', 'lc', 'so', 'sc']
            ax1.text(item[0], item[3], s='{0}'.format(item[2]))

        ax2 = ax1.twinx()
        ax2.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.trend_short[define.PLT_START:define.PLT_END], 'r')

        _ret_path = get_path([define.RESULT_DIR, define.BT_DIR, '{0}_{1}.jpg'.format(instrument_id, trade_date)])
        plt.savefig(_ret_path)
    long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open = 0, 0, 0, 0, 0, 0
    total_fee = 0.0
    total_holding_time = 0.0
    max_holding_time = -np.inf
    min_holding_time = np.inf
    for item in account.transaction:
        if item[2] == 'long_open':
            long_open += 1
        elif item[2] == 'short_open':
            short_open += 1
        elif item[2] == 'long_close':
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:  # short close
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1
    average_holding_time = total_holding_time / (long_open + short_open) if long_open + short_open > 0 else 0.0
    logger.info("******************back test models for date:{0}*********************".format(trade_date))
    logger.info('trade date', trade_date)
    logger.info(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    logger.info('total return:', total_return)
    logger.info('total fee:', account.fee)
    logger.info('total risk:', total_risk)
    logger.info('update factor time:', update_factor_time)
    logger.info('get signal time:', get_signal_time)
    logger.info("average_holding_time:", average_holding_time)
    logger.info("max_holding_time:", max_holding_time)
    logger.info("min_holding_time:", min_holding_time)
    logger.info("******************back test models for date:{0}*********************".format(trade_date))

    precision = (correct_long_open + correct_short_open) / (
            long_open + short_open) if long_open + short_open > 0 else 0.0
    # f = open("models/results_{0}.txt".format(product_id), "a")
    # _key = '{0}_{1}'.format(trade_date, instrument_id)
    # result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    # f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    _ret_path = get_path([define.RESULT_DIR, define.BT_DIR, '{0}.csv'.format(result_fname_digest)])
    try:
        result_df = pd.read_csv(_ret_path)
    except Exception as ex:
        result_df = pd.DataFrame(
            {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return_final': [],
             'total_return_unclose': [],
             'total_fee': [],
             'unclosed_value': [], 'precision': [], 'long_open': [], 'short_open': [],
             'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
             'average_holding_time': [], 'max_holding_time': [], 'min_holding_time': []
             })

    result_df = result_df.append(
        {'trade_date': trade_date, 'product_id': product_id, 'instrument_id': instrument_id,
         'total_return_final': total_return, 'total_return_unclose': total_return_risk,
         'total_fee': account.fee,
         'unclosed_value': total_risk, 'precision': precision,
         'long_open': long_open, 'short_open': short_open, 'correct_long_open': correct_long_open,
         'wrong_long_open': wrong_long_open, 'correct_short_open': correct_short_open,
         'wrong_short_open': wrong_short_open, 'average_holding_time': average_holding_time,
         'max_holding_time': max_holding_time, 'min_holding_time': min_holding_time
         }, ignore_index=True)

    result_df.to_csv(_ret_path, index=False)
    ret = (total_return, account.fee, precision, account.transaction)
    return ret


if __name__ == '__main__':
    # backtesting()
    import pprint

    print(backtest_quick(data_fetcher=data_fetcher, product_id='rb', trade_date='2021-03-08'))