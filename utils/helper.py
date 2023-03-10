import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..", "../mi_models")))
import configparser
import numpy as np
import pandas as pd
import json
import hashlib
import talib as ta
import re
from typing import List, Tuple, Union
from datetime import time
import datetime
from utils.logger import Logger

log_path = os.path.abspath(
    os.path.join(__file__, "../..", "../mi_models/data/logs/{}".format(os.path.split(__file__)[-1].strip('py'))))
logger = Logger(log_path, 'INFO', __name__).get_log()


def get_config(overwrite_config_path=None):
    '''
    apply configparser to parse the config file, by default type is str
    :param overwrite_config_path: str, path of the config file to overwrite the default path
    :return:
    '''
    config = configparser.ConfigParser()
    config_path = overwrite_config_path or os.path.join(get_parent_dir(), 'conf', 'conf')
    config.read(config_path)
    logger.debug('Reading config file from:{0}'.format(config_path))
    return config


def set_config(config_path='', update_sections={}):
    if not config_path or not update_sections:
        logger.info('Nothing to update to config')
        return
    config = configparser.ConfigParser()
    config_path = config_path or os.path.join(get_parent_dir(), 'conf', 'conf')
    config.read(config_path)
    logger.debug('Reading config file from:{0}'.format(config_path))
    sections = config.sections()
    for section, param in update_sections.items():
        if section not in sections:
            config.add_section(section)
        if param:
            for key, val in param.items():
                config.set(section, key, str(val))
    config.write((open(config_path, 'w')))


def format_float(val=None):
    return np.nan if not val else float(val)


def handle_none(df):
    df.fillna(value=np.nan, inplace=True)
    df.replace(to_replace=np.NaN, value=0, inplace=True)
    df.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)


def trans_none_2_nan(tmpdata):
    if (tmpdata == None):
        tmpdata = np.nan
    return tmpdata


def clear_none(tmpFactors):
    tmpFactors.fillna(value=np.nan, inplace=True)
    return tmpFactors


def clear_nan_inf(tmpfactors):
    tmpfactors.replace(to_replace=np.NaN, value=0, inplace=True)
    tmpfactors.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
    return tmpfactors


def df_to_payload(df=None):
    index_lst = df.index
    list_of_dicts = df.to_dict('record')
    record_num = len(index_lst)
    for i in range(record_num):
        list_of_dicts[i].update({'index': index_lst[i]})
    return list_of_dicts


def get_hash_key(key=None):
    if not key:
        return key

    if isinstance(key, list):
        key = ','.join(key)
    m1 = hashlib.md5()
    m1.update(key.encode('utf-8'))
    return m1.hexdigest()


def format_sec_code(val=None):
    sec_code = val if isinstance(val, str) else str(val)
    if len(sec_code) == 6:
        return sec_code
    else:
        return '0' * (6 - len(sec_code)) + sec_code


def list_files(abs_path=None, ref_path=None):
    if not (abs_path or ref_path):
        return []
    path = abs_path or os.path.join(get_parent_dir(), ref_path)
    if os.path.exists(path):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            all_files = os.listdir(path)
            return ['{0}/{1}'.format(path, item) for item in all_files]
    return []


# TODO double check the logic, for the first period/window_len - 1 , there is no values, whether to drop it or filled it
# fill with the previous series regarding the window_len/period
def adjusted_sma(inputs=[], period=10, filled=True):
    ret = list(ta.SMA(np.array(list(inputs), dtype=float), timeperiod=period))
    if not filled:
        return ret
    # fill the na values regarding of windows_len
    fixed_len = period - 1 if period < len(inputs) else len(inputs)
    for i in range(fixed_len):
        ret[i] = sum(inputs[:i + 1]) / (i + 1)
    return ret


def get_parent_dir(file=None):
    _file = file or __file__
    curr_path = os.path.abspath(_file)
    parent_path = os.path.abspath(os.path.dirname(curr_path) + os.path.sep)
    return os.path.dirname(parent_path)


def write_json_file(file_path='', data=None):
    if not data:
        return
    with open(file_path, 'w') as outfile:
        j_data = json.dumps(data)
        outfile.write(j_data)


def load_json_file(filepath=''):
    with open(filepath) as infile:
        contents = infile.read()
        return json.loads(contents)


def gen_file_name(
        code: Union[int, str],
        date: Union[int, str, pd.Timestamp, datetime.date],
        style: str,
) -> str:
    """根据 code 和 date 生成符合格式的文件名

    Args:
        code (int, str): 六位股票代码 (譬如. 1 或者 '000001')
        date (str, pd.Timestamp, datetime.date): 数据对应日期
        style (str): 数据类型 (目前支持 'stock_trade', 'stock_order', 'stock_orderqueue' 和 'stock_snapshot')

    Returns:
        str: 符合格式的文件名
    """
    code = str(code).zfill(6)
    if code[0] == "6":
        postfix = "SSE"
    else:
        postfix = "SZE"
    if isinstance(date, int):
        date = str(date)
    date = pd.Timestamp(date).strftime("%Y%m%d")
    # FIXME HACK FOR THE FILE NAME, double check
    # return f"{style}_{code}.{postfix}_{date}.csv.gz"
    return f"{style}_{code}.{postfix}_{date}.csv.gz"


def resample(
        origin_df: pd.DataFrame,
        freq: str = "60S",
        data_alignment: str = "right",
        style: str = "stock_trade",
        price_column: str = "TRADE_PRICE",
        vol_column: str = "TRADE_QTY",
        amount_volumn: str = "TRADE_AMOUNT",
        cancel_column: str = "EXEC_TYPE",
        date_column: str = "TRADE_DATE",
        time_column: str = "TRADE_TIME",
        code_column: str = "SECURITY_ID",
) -> pd.DataFrame:
    """对输入数据进行降采样操作, 只支持逐笔成交数据

    Args:
        origin_df (pd.DataFrame): 需要包含日期和股票代码
        freq (str, optional): 降采样的频率, 默认为 "60S"
        data_alignment(str, option): 降采样以起始时间为准，还是结束时间为准，默认为 'right'
        style (str, optional): 数据类型，默认为 "stock_trade"
        price_column (str): 价格列，默认为 "TRADE_PRICE"
        vol_column (str): 交易量, 默认为 "TRADE_QTY"
        amount_volumn (str): 交易额，默认为 "TRADE_AMOUNT"
        cancel_column (str, optional): 撤单成交列，只对逐笔成交有效, 默认为 "EXEC_TYPE"
        date_column (str, optional): 日期列，默认为 "TRADE_DATE"
        time_column (str, optional): 时间列，默认为 "TRADE_TIME"
        code_column (str, optional): 股票代码列，默认为 :"SECURITY_ID"
        if_delayed (bool, optional): 是否惰性计算，默认为 False

    Returns:
        pd.DataFrame: 降采样后的数据
    """

    def _resample_ss(grp, freq, data_alignment, time_column):
        # 重复时间戳处理
        grp = grp.set_index(time_column).sort_index()
        # 交易时间过滤
        # grp = grp.loc[time(9, 25): time(11, 30)].append(
        #     grp.loc[time(13, 0): time(15, 0)]
        # )
        grp = pd.concat([grp.loc[time(9, 25): time(11, 30)], grp.loc[time(13, 0): time(15, 0)]])
        # 降采样
        if data_alignment == "left":
            return (
                grp.resample(freq)
                .apply(
                    {"BS_VOL": "sum"}
                )
                .dropna()
                # .droplevel(level=0, axis=0)
                .rename(columns={"BS_VOL": "bs_vol"})
            )
        else:
            return (
                grp.resample(freq)
                .apply(
                    {"BS_VOL": "sum"}
                )
                .dropna()
                .shift(1)
                .dropna()
                # .droplevel(level=0, axis=0)
                .rename(columns={"BS_VOL": "bs_vol"})
            )

    def _resample(grp, freq, data_alignment, time_column):
        # 重复时间戳处理
        grp = grp.set_index(time_column).sort_index()
        # 交易时间过滤
        # grp = grp.loc[time(9, 25): time(11, 30)].append(
        #     grp.loc[time(13, 0): time(15, 0)]
        # )
        grp = pd.concat([grp.loc[time(9, 25): time(11, 30)], grp.loc[time(13, 0): time(15, 0)]])
        # 降采样
        if data_alignment == "left":
            return (
                grp.resample(freq)
                .apply(
                    {"TRADE_PRICE": "ohlc", "TRADE_QTY": "sum", "TRADE_AMOUNT": "sum", "BS_VOL": "sum"}
                )
                .dropna()
                .droplevel(level=0, axis=1)
                .rename(columns={"TRADE_QTY": "vol", "TRADE_AMOUNT": "amount", "BS_VOL": "bs_vol"})
            )
        else:
            return (
                grp.resample(freq)
                .apply(
                    {"TRADE_PRICE": "ohlc", "TRADE_QTY": "sum", "TRADE_AMOUNT": "sum", "BS_VOL": "sum"}
                )
                .dropna()
                .shift(1)
                .dropna()
                .droplevel(level=0, axis=1)
                .rename(columns={"TRADE_QTY": "vol", "TRADE_AMOUNT": "amount", "BS_VOL": "bs_vol"})
            )

    if not isinstance(origin_df, pd.DataFrame):
        raise ValueError("[ERROR]\t不支持的数据格式")
    if origin_df.empty:
        print("[WARNING]\t输入数据为空，返回空 dataframe")
        return pd.DataFrame()
    if style == "stock_snapshot":
        return origin_df.groupby('TRADE_DATE').apply(
            _resample_ss,
            freq=freq,
            data_alignment=data_alignment,
            time_column=time_column,
        ).droplevel(0, axis=0)
    # 对逐笔成交进行兼容处理，深交所的逐笔成交同时包括了撤单成交信息
    if style == "stock_trade":
        origin_df = origin_df.loc[origin_df[cancel_column] != "4"]
    return origin_df.groupby(date_column).apply(
        _resample,
        freq=freq,
        data_alignment=data_alignment,
        time_column=time_column,
    )


def load_data(
        code: Union[int, str],
        start: Union[int, str, pd.Timestamp, datetime.datetime],
        end: Union[int, str, pd.Timestamp, datetime.datetime],
        style: str,
        base_dir: str,
        date_column: str = "TRADE_DATE",
        start_time: time = None,
        end_time: time = None,
        if_filter: bool = True,
) -> pd.DataFrame:
    """读取指定时间段指定股票或股票列表的数据

    Args:
        code (Union[int, str, List, Tuple]): 股票六位代码或者股票代码列表
        start (Union[int, str, pd.Timestamp, datetime.datetime]): 起始时间
        end (Union[int, str, pd.Timestamp, datetime.datetime]): 结束时间
        style (str): 数据类型 (目前支持 'stock_trade', 'stock_order', 'stock_orderqueue' 和 'stock_snapshot')
        base_dir (str): 默认数据路径，该路径下为 'stock_trade', 'stock_order' 等文件夹
        date_column (str, optional): 为了方便处理，为数据加入日期列，默认日期列名为 'TRADE_DATE'
        if_filter (bool, option): 是否对非交易时间进行过滤，默认为 True

    Returns:
        pd.DataFrame: 以日期为索引的 DataFrame
    """
    # 股票代码格式化处理
    if isinstance(code, int):
        code = str(code).zfill(6)
    if isinstance(code, str):
        pattern = re.compile(r"\d+")
        match = pattern.match(code)
        if not match:
            raise ValueError("[ERROR]\t股票 {} 格式不支持".format(code))
        else:
            code = match.group()

    # 日期序列生成
    start = pd.Timestamp(str(start))
    end = pd.Timestamp(str(end))
    dates_range = pd.date_range(start, end)

    # 对不同数据格式的时间列兼容处理
    if style in ["stock_snapshot", "stock_orderqueue"]:
        time_column = "QUOT_TIME"
    elif style == "stock_order":
        time_column = "ORDER_TIME"
    elif style == "stock_trade":
        time_column = "TRADE_TIME"
    else:
        raise ValueError(
            f"[ERROR]\t Currently data style {style} has not been supported!"
        )

    # 返回数据
    df_result = pd.DataFrame()
    # 数据读取
    for date in dates_range:
        date = date.strftime("%Y%m%d")
        # 非交易日过滤
        if not os.path.exists(
                os.path.join(base_dir, style, date)
        ):  # faster than os.listdir
            continue
        # 文件名
        file_name = gen_file_name(code, date, style)
        # 全路径文件名
        abs_file_name = os.path.join(base_dir, style, date, file_name)
        # 停牌股或其他原因导致数据缺失的股票跳过
        if not os.path.exists(abs_file_name):
            continue
        df = pd.read_csv(abs_file_name, dtype={time_column: str})
        # 兼容性处理
        if "TRADE_BS_FLAG" in df.columns:
            df["TRADE_BS_FLAG"] = df["TRADE_BS_FLAG"].astype(str)
        if "EXEC_TYPE" in df.columns:
            df["EXEC_TYPE"] = df["EXEC_TYPE"].astype(str)
        if "INSTRUMENT_STATUS" in df.columns:
            df["INSTRUMENT_STATUS"] = df["INSTRUMENT_STATUS"].astype(str)
        # 重复数据处理
        if style == "stock_snapshot":  # 对快照而言，保留最新数据
            df = df.drop_duplicates(subset=[time_column], keep="last")
        elif style == "stock_trade":  # 对逐笔成交数据做处理
            # EXEC_TYPE is cancel type, for sz, the trade data contains the cancel trades(EXEC_TYPE="4")
            df = df[df.EXEC_TYPE != '4']

            trade_price_lst = list(df['TRADE_PRICE'])
            trade_vol_lst = list(df['TRADE_QTY'])
            n = len(trade_price_lst)
            bs_vol = [trade_vol_lst[0]]
            # since TRADE_BS_FLAG is missing, derived from the trade price, if price up, then it's buy;
            # if price down, then it's sell; if price not change, then it is same as the previous trade
            for i in range(1, n):
                if trade_price_lst[i] > trade_price_lst[i - 1]:
                    bs_vol.append(trade_vol_lst[i])
                elif trade_price_lst[i] < trade_price_lst[i - 1]:
                    bs_vol.append(trade_vol_lst[i] * -1)
                else:

                    bs_vol.append(trade_vol_lst[i] * np.sign(bs_vol[i - 1]))
            df['BS_VOL'] = bs_vol
        else:  # 对逐笔数据而言，去重,#TODO only works for snapshot
            df = df.drop_duplicates(subset=["BID_APPL_SEQ_NUM", "OFFER_APPL_SEQ_NUM"])
        df[date_column] = df[time_column].iloc[0][:8]
        df[time_column] = pd.to_datetime(df[time_column], format="%Y%m%d%H%M%S%f")
        df = df.set_index(time_column)
        if if_filter:
            if (not start_time) or (not end_time):
                # df = df.loc[time(9, 25): time(11, 30)].append(
                #     df.loc[time(13, 0): time(15, 0)]
                # )
                df = pd.concat([df.loc[time(9, 30): time(11, 30)], df.loc[time(13, 0): time(15, 0)]])
            elif not start_time:
                if not isinstance(start_time, time):
                    raise ValueError(
                        "[ERROR]\t{} has an unsupported type!".format(start_time)
                    )
                df = df.loc[time(9, 30): time(11, 30)].append(
                    df.loc[time(13, 0): end_time]
                )
            elif not end_time:
                if not isinstance(end_time, time):
                    raise ValueError(
                        "[ERROR]\t{} has an unsupported type!".format(end_time)
                    )
                df = df.loc[start_time: time(11, 30)].append(
                    df.loc[time(13, 0): time(15, 0)]
                )
            else:
                df = df.loc[start_time: time(11, 30)].append(
                    df.loc[time(13, 0): end_time]
                )
        # df_result = df_result.append(df.reset_index())
        df_result = pd.concat([df_result, df.reset_index()])

    return df_result





if __name__ == '__main__':
    print(get_parent_dir())
    print(gen_file_name(code='000001', date='20210104', style='stock_trade'))
