import hashlib
from datetime import datetime, timedelta
from tushare.util.dateu import trade_cal
import talib

########################################################################################################################
#
#
#   无状态公共函数
#
#
#
########################################################################################################################


def today():
    """

    @return:
    """
    tm_now = datetime.now()
    td = tm_now.strftime("%Y-%m-%d")
    return td


def n_days_ago(n_days):
    """

    @param n_days:
    @return:
    """
    tm_now = datetime.now()
    delta = timedelta(days=n_days)
    tm_start = tm_now - delta
    ago = tm_start.strftime("%Y-%m-%d")
    return ago


def n_trade_days_ago(n_trade_days, end_dt=today()):
    """
    获取从end_dt向前的N个交易日开始的日期
    @param n_trade_days:从start_dt开始往前面推几个交易日。
    @param end_dt:往前推算交易日的开始日期，格式类似"2019-02-02"
    @return:
    """
    trade_days = trade_cal()
    last_idx = trade_days[trade_days.calendarDate == end_dt].index.values[0]

    df = trade_days[trade_days.isOpen == 1]
    start_date = df[df.index <= last_idx].tail(n_trade_days).head(1).iat[0, 0]
    return start_date


def get_trade_days_between(start_dt, end_dt):
    """
    找到两个日期中间，包含开始和结束日期内的交易日
    @param start_dt:
    @param end_dt:
    @return:
    """
    trade_days = trade_cal()
    n = trade_days[(trade_days.calendarDate<=end_dt)&(trade_days.calendarDate>=start_dt)&(trade_days.isOpen==1)].shape[0]
    return n


def get_md5(s):
    """

    @param s:
    @return:
    """
    md = hashlib.md5()
    md.update(s.encode('utf-8'))
    return md.hexdigest()


########################################################################################################################
#
#
#   指标计算函数
#
#
#
########################################################################################################################


def MA(df, window, field, new_field):
    """

    @param df:
    @param window:
    @param field:
    @param new_field:
    @return:
    """
    df[new_field] = df[field].rolling(window=window).mean()
    return df


def MACD(df, field_name='close', quick_n=12, slow_n=26, dem_n=9):
    """

    @param df:
    @param field_name:
    @param quick_n:
    @param slow_n:
    @param dem_n:
    @return:
    """
    diff, macdsignal, macd_bar = talib.MACD(df[field_name], fastperiod=quick_n, slowperiod=slow_n, signalperiod=dem_n)
    return diff, macdsignal, macd_bar
