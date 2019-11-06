import math
from itertools import groupby
from operator import itemgetter

import talib
from futu import *
from futu import RET_OK
from pandas import DataFrame
from tushare.util.dateu import trade_cal

from fearquantlib.config import QuantConfig, timeConvTable

_config = QuantConfig()


class KL_Period(object):
    KL_60 = "KL_60"
    KL_30 = "KL_30"
    KL_15 = "KL_15"


K_LINE_TYPE = {
    KL_Period.KL_60: KLType.K_60M,
    KL_Period.KL_30: KLType.K_30M,
    KL_Period.KL_15: KLType.K_15M,
}


class RG_AreaTagFieldNameExt(object):
    BAR_WAVE_TAG = 'tag'
    RG_TAG = 'rg_tag'


class RG_AreaTag(object):
    GREEN = 'g'
    RED = 'r'


class WaveType(object):
    RED_PEAK = 2  # 红柱高峰
    RED_VALLEY = 1  # 红柱峰底

    GREEN_PEAK = -2  # 绿柱波峰
    GREEN_VALLEY = -1  # 绿柱波底，乳沟深V的尖


def MA(df, window, field, new_field):
    """

    :param df:
    :param window:
    :param field:
    :param new_field:
    :return:
    """
    df[new_field] = df[field].rolling(window=window).mean()
    return df


def MACD(df, field_name='close', quick_n=12, slow_n=26, dem_n=9):
    """
    
    :param df:
    :param field_name:
    :param quick_n:
    :param slow_n:
    :param dem_n:
    :return:
    """
    diff, macdsignal, macd_bar = talib.MACD(df[field_name], fastperiod=quick_n, slowperiod=slow_n, signalperiod=dem_n)
    return diff, macdsignal, macd_bar


def __find_successive_bar_areas(df: DataFrame, field='bar'):
    """
    这个地方不管宽度，只管找连续的区域
    改进的寻找连续区域算法；
    还有一种算法思路，由于红色柱子>0, 绿色柱子<0, 只要找到 x[n]*x[n+1]<0的点然后做分组即可。
    :param raw_df:
    :param field:
    :return:
    """
    successive_areas = []
    # 第一步：把连续的同一颜色区域的index都放入一个数组
    arrays = [df[df[field] >= 0].index.array, df[df[field] <= 0].index.array]
    for arr in arrays:
        successive_area = __do_find_successive_areas(arr)
        successive_areas.append(successive_area)

    return successive_areas[0], successive_areas[1]  # 分别是红色和绿色的区间


def __do_find_successive_areas(arr):
    """
    值连续升序或者连续降序的段
    :param arr: 下标index
    :return:
    """
    successive_area = []
    for k, g in groupby(enumerate(arr), lambda iv: iv[0] - iv[1]):
        index_group = list(map(itemgetter(1), g))
        successive_area.append((min(index_group), max(index_group)))

    return successive_area


def today():
    """

    :return:
    """
    tm_now = datetime.now()
    td = tm_now.strftime("%Y-%m-%d")
    return td


def n_days_ago(n_days):
    """
    :param n_days:
    :return:
    """
    tm_now = datetime.now()
    delta = timedelta(days=n_days)
    tm_start = tm_now - delta
    ago = tm_start.strftime("%Y-%m-%d")
    return ago


def n_trade_days_ago(n_trade_days, end_dt=today()):
    """
    获取从end_dt向前的N个交易日开始的日期
    :param n_trade_days: 从start_dt开始往前面推几个交易日。
    :param start_dt: 往前推算交易日的开始日期，格式类似"2019-02-02"
    :return:
    """
    trade_days = trade_cal()
    last_idx = trade_days[trade_days.calendarDate == end_dt].index.values[0]

    df = trade_days[trade_days.isOpen == 1]
    start_date = df[df.index <= last_idx].tail(n_trade_days).head(1).iat[0, 0]
    return start_date


def prepare_csv_data(code_list, start_date=None, end_date=None, n_days=_config.n_days_bar_fetch, timeUnitDelta=0):
    """

    :param code_list: 股票列表
    :return:{code:[周期1，小周期2, 小周期3.。。]，}
    """
    if start_date is None:
        start_date = n_trade_days_ago(n_days)
    if end_date is None:
        end_date = today()

    code_data_path = {}
    quote_ctx = OpenQuoteContext(host=_config.futu_api_ip, port=_config.futu_api_port)
    for code in code_list:
        files = []
        for k in _config.periods:
            ktype = K_LINE_TYPE[k]
            ret, df, page_req_key = quote_ctx.request_history_kline(code, start=start_date, end=end_date,
                                                                    ktype=ktype,
                                                                    fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE,
                                                                            KL_FIELD.HIGH, KL_FIELD.LOW],
                                                                    max_count=1000)
            if ret != RET_OK:
                logger.error(df)
                return None

            csv_file_name = df_file_name(code, ktype)
            if timeUnitDelta >= 0:
                df.to_csv(csv_file_name)
            else:
                df[:timeUnitDelta // timeConvTable[k]].to_csv(csv_file_name)
            files.append(csv_file_name)
            time.sleep(3.1)  # 频率限制
        code_data_path[code] = files

    quote_ctx.close()

    return code_data_path


def get_df_of_code(code, start_date, end_date, ktype=K_LINE_TYPE[KL_Period.KL_60]):
    """

    :param code:
    :param ktype:
    :param n_days:
    :return:
    """
    quote_ctx = OpenQuoteContext(host=_config.futu_api_ip, port=_config.futu_api_port)
    ret, df, page_req_key = quote_ctx.request_history_kline(code, start=start_date, end=end_date,
                                                            ktype=ktype,
                                                            fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE, KL_FIELD.HIGH,
                                                                    KL_FIELD.LOW],
                                                            max_count=1000)
    quote_ctx.close()
    return df


def df_file_name(stock_code, ktype):
    """

    :param stock_code:
    :param ktype:
    :return:
    """
    prefix = _config.dev_model
    return f'data/{prefix}{stock_code}_{ktype}.csv'


def compute_df_bar(code_list):
    """
    计算60,30,15分钟的指标，存盘
    :param df:
    :return:
    """
    for code in code_list:
        for k in _config.periods:  # 这里面的周期从大到小排列，并且可在配置里enable/disable
            ktype = K_LINE_TYPE[k]
            csv_file_name = df_file_name(code, ktype)
            df = pd.read_csv(csv_file_name, index_col=0)
            df = __do_compute_df_bar(df, k)  # K 是配置文件里的["KL_60","KL_30", "KL_15"] 之一
            # TODO 这里还要把尾部，从最后一个1/-1之后的强行选出来一个顶、底。尾部一般由于数据少没有被打上tag，就要特殊处理
            df.to_csv(csv_file_name)


def __do_compute_df_bar(df, k_period):
    diff, dem, bar = MACD(df)
    df['macd_bar'] = bar  # macd
    df = MA(df, 5, 'close', 'ma5')
    df = MA(df, 10, 'close', 'ma10')
    df['em_bar'] = (df['ma5'] - df['ma10']).apply(lambda val: round(val, 2))  # 均线
    __add_df_tags(df, "macd_bar", k_period)  # 顶部、谷底、连续区域打标r/g
    __add_df_tags(df, "em_bar", k_period)  # r/g
    return df


def __add_df_tags(df: DataFrame, field, k_peroid):
    """
    field 的连续区域以及顶底
    :param df:
    :return:
    """
    moutain_min_width = _config.periods_config[k_peroid]['moutain_min_width']
    tag_field_name = __ext_field(field)
    red_areas, blue_areas = __find_successive_bar_areas(df, field)
    df_blue = __do_bar_wave_tag(df, field, blue_areas, moutain_min_width=moutain_min_width)
    df_blue[tag_field_name] *= -1  # 因为计算都变为正值，所以绿柱子要乘以-1
    df_red = __do_bar_wave_tag(df, field, red_areas, moutain_min_width=moutain_min_width)
    df[tag_field_name] = df_red[tag_field_name] | df_blue[tag_field_name]

    # 连续的区域用r, g区分，方便后续计算
    color_tag_field = __ext_field(field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    df[color_tag_field] = RG_AreaTag.GREEN  # 先设置绿色很重要，因为可以把宽度太短的连续区域过滤掉

    for s, e in red_areas:  # 过滤掉红色过小的区域，这里还要注意一点，如果可被过滤的短区域存在最后，也是要保留成红色的
        if e + 1 == df.shape[0]:
            df.loc[s:e, color_tag_field] = RG_AreaTag.RED
        elif e - s + 1 >= moutain_min_width:
            df.loc[s:e, color_tag_field] = RG_AreaTag.RED


def __ext_field(field_name, ext=RG_AreaTagFieldNameExt.BAR_WAVE_TAG):
    """

    :param field_name:
    :return:
    """
    return f'_{field_name}_{ext}'


def __do_bar_wave_tag(raw_df: DataFrame, field, successive_bar_area, moutain_min_width=5):
    """
    这里找波峰和波谷，找谷底的目的是为了测量波峰/谷的斜率
    # TODO 试一下FFT寻找波谷波峰

    :param raw_df:
    :param field:
    :param successive_bar_area: 想同样色柱子区域, [tuple(start, end)]
    :param moutain_min_width: 作为一个山峰最小的宽度，否则忽略
    :return: 打了tag 的df副本
    """
    df = raw_df[[field]].copy()
    tag_field = __ext_field(field)
    df[tag_field] = 0  # 初始化为0
    df[field] = df[field].abs()  # 变成正值处理
    for start, end in successive_bar_area:  # 找到s:e这一段里的所有波峰
        sub_area_list = [(start, end)]

        for s, e in sub_area_list:  # 产生的破碎的连续区间加入这个list里继续迭代直到为空
            if e - s + 1 < moutain_min_width:  # 山峰的宽度太窄，可以忽略
                continue
            # 找到最大柱子，在df上打标
            max_row_index = df.iloc[s:e + 1][field].idxmax(axis=0)  # 寻找规定的行范围的某列最大值的索引
            # 先不急于设置为波峰，因为还需要判断宽度是否符合要求
            # 从这根最大柱子向两侧扫描，直到波谷
            # max_row_index先向左侧扫描
            i, j = s, e
            for i in range(max_row_index, s + 1, -1):  # 向左侧扫描, 下标是(s,s+1,   [s+2, max_row_index])
                if df.at[i, field] >= df.at[i - 1, field] or df.at[i, field] >= df.at[i - 2, field]:
                    if i == s + 2:
                        i = s
                        break
                    else:
                        continue
                else:
                    break  # i 就是左侧波谷

            # 从min_index向右侧扫描
            for j in range(max_row_index, e - 1):  # 下标范围是[arr_min_index, len(arr)-2]
                if df.at[j, field] >= df.at[j + 1, field] or df.at[j, field] >= df.at[j + 2, field]:
                    if j == e - 2:
                        j = e
                    else:
                        continue
                else:
                    break  # j 就是右侧波谷

            # =========================================================
            # 现在连续的波段被分成了3段[s, i][i, j][j, e]
            # max_row_index 为波峰；i为波谷；j为波谷；'
            if j - i + 1 >= moutain_min_width:
                df.at[max_row_index, tag_field] = WaveType.RED_PEAK  # 打tag

                # 在下一个阶段中评估波峰波谷的变化度（是否是深V？）
                # 一段连续的区间里可以产生多个波峰，但是波谷可能是重合的，这就要评估是否是深V，合并波峰
                df.at[i, tag_field] = WaveType.RED_VALLEY
                df.at[j, tag_field] = WaveType.RED_VALLEY

            # 剩下两段加入sub_area_list继续迭代
            if i - s + 1 >= moutain_min_width:
                sub_area_list.append((s, i))
            if e - j + 1 >= moutain_min_width:  # j为啥不能为0呢？如果为0 说明循环进不去,由此推倒出极值点位于最左侧开始的2个位置，这个宽度不足以参与下一个遍历。
                sub_area_list.append((j, e))

        # 这里是一个连续区间处理完毕
        # 还需要对波谷、波峰进行合并，如果不是深V那么就合并掉
        # TODO

    return df


def bottom_divergence_cnt(df: DataFrame, bar_field, value_field, start_time_key=None):
    """
    field字段出现连续背离的个数,也既多重背离个数。
    背离必须是连续的。

    方法是：找出最后一段绿色bar_field（为了计算方便负值转为正值），value_field顶点的值，形成两个array
        找到bar_field中连续的下降段个数S_1，value_field中连续的下降段个数S_2。最后返回max(S_1,S_2)
        为什么选最大而不是最小呢？其实最小也可以，但里面涉及到一些模糊的东西，把阈值设大，然后还要
        辅助人工交易，如果取了min过于严格会误杀很多。

    :param df:
    :param bar_field: bar的field名字
    :param value_field:  价格
    :return: 没有背离为0，
    """
    rg_tag_name = __ext_field(bar_field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    field_tag_name = __ext_field(bar_field, ext=RG_AreaTagFieldNameExt.BAR_WAVE_TAG)
    if start_time_key is not None:
        dftemp = df[(df.time_key >= start_time_key) & (df[rg_tag_name] == RG_AreaTag.GREEN)].copy().reset_index(
            drop=True)
    else:
        dftemp = __get_last_successive_rg_area(df, rg_tag_name, area=RG_AreaTag.GREEN)  # 获得最后一段连续绿色区域
    # 这一段连续区域里包含了被同化的不同色，需要对这部分对应的值进行处理，等于0是个办法
    # 对应于底背离，应该是bar_field>0, 但是 颜色标记为G的那些，因为颜色全都是G，因此只需要
    # 把bar_field>0的全都设置为0即可
    dftemp.loc[dftemp[bar_field] > 0, bar_field] = 0
    # TODO 对最后一段进行打tag，要做一定的预测行为?绿峰一直在增长但背离的情况
    bar_array = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK][bar_field].abs().array  # 最后一段连续绿色区域的bar的顶点
    val_array = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK][value_field].array
    # # 然后找出来最大长度的区
    bar_desc = __max_successive_series_len(bar_array)
    val_desc = __max_successive_series_len(val_array)
    cnt = max(bar_desc, val_desc) - 1
    return max(0, cnt)  # 防止小于0


def __max_successive_series_len(arr):
    """
    寻找最大连续子序列，子序列的下标必须是相连的
    例如， 1,2,3 返回3
    Parameters
    ----------
    arr

    Returns
    -------

    """
    max_area_len = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[j - 1]:
                max_area_len = max(j - i + 1, max_area_len)
            else:
                i = j

    return max_area_len


def bar_green_wave_cnt(df: DataFrame, bar_field='macd_bar', start_time_key=None):
    """
    在一段连续的绿柱子区间，当前的波峰是第几个
    方法是：从当前时间开始找到前面第一段连续绿柱，然后计算绿柱区间有几个波峰；
    如果当前是红柱但是没超过设置的最大宽度，可以忽略这段红柱
    :param df:
    :param field:
    :return:  波峰个数, 默认1
    """
    if df.shape[0] == 0:
        return 0
    field_tag_name = __ext_field(bar_field, ext=RG_AreaTagFieldNameExt.BAR_WAVE_TAG)
    rg_tag_name = __ext_field(bar_field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    if start_time_key is not None:
        dftemp = df[(df.time_key >= start_time_key) & (df[rg_tag_name] == RG_AreaTag.GREEN)].copy().reset_index(
            drop=True)
    else:
        dftemp = __get_last_successive_rg_area(df, rg_tag_name, area=RG_AreaTag.GREEN)  # 获得最后一段连续绿色区域
    wave_cnt = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK].shape[0]
    # TODO 这个地方有点问题，对于最后一段区域需要进一步处理，做一定预测。当前的GREEN_TOP在实时中不一定被打标
    return wave_cnt


def get_current_ma_distance(df: DataFrame):
    """
    计算(ma(5)-ma(10))/close，保留2位小数
    :param df:
    :return: 0.0131 ， 0.0323， 保留3位小数
    """
    close_price = df.at[df.shape[0] - 1, 'close']
    ma_gap = df.at[df.shape[0] - 1, 'em_bar']
    return round(abs(ma_gap / close_price), 3)


def __get_last_successive_rg_area(df: DataFrame, rg_field_name, area=RG_AreaTag.GREEN):
    """
    获取最后一段颜色为area的连续区域
    :param df:
    :param rg_field_name:
    :param area:
    :return: df
    """
    # TODO 全红或者全绿需要处理
    if df[df[rg_field_name] != area].shape[0] == 0:
        return df.copy().reset_index(drop=True)
    else:
        last_idx = df[df[rg_field_name] != area].tail(1).index[0]  # 红绿只抹平毛刺小区域，但是计算的实际值需要取出来之后做进一步处理
        dftemp = df[df.index > last_idx].copy().reset_index(drop=True)
        return dftemp


def resonance_cnt(df1: DataFrame, df2: DataFrame, field, start_time_key=None):
    """
    2个周期的共振次数
    方法是：选最后一段连续绿色区域，找出波数w1和w2,然后返会min(s1,s2)
    :param df1:
    :param df2:
    :param field:
    :return:
    """
    rg_tag_name = __ext_field(field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    if start_time_key is not None:
        area1 = df1[(df1.time_key >= start_time_key) & df1[rg_tag_name] == RG_AreaTag.GREEN].copy().reset_index(
            drop=True)
        area2 = df2[(df2.time_key >= start_time_key) & df2[rg_tag_name] == RG_AreaTag.GREEN].copy().reset_index(
            drop=True)
    else:
        area1 = __get_last_successive_rg_area(df1, rg_tag_name, area=RG_AreaTag.GREEN)
        area2 = __get_last_successive_rg_area(df2, rg_tag_name, area=RG_AreaTag.GREEN)
    wave_1 = bar_green_wave_cnt(area1, field)
    wave_2 = bar_green_wave_cnt(area2, field)
    return max(0, min(wave_1, wave_2) - 1)  # 2个波形成1个共振


def is_macd_bar_reduce(df: DataFrame, field='macd_bar', k_period=_config.periods[0], max_reduce_bar_distance=4):
    """
    macd 绿柱子第一根减少出现，不能减少太剧烈，前面的绿色柱子不能太少
    前提是：最后一段柱子必须是绿色的
    :param df:
    :param field:
    :return: 减少返回True, 否则False, 第二个参数是最后一根绿色柱子出现的日期
    """
    field_rg_tag_name = __ext_field(field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    field_tag = __ext_field(field, ext=RG_AreaTagFieldNameExt.BAR_WAVE_TAG)
    last_idx = df[df[field_rg_tag_name] == RG_AreaTag.RED].tail(1).index[0]  # 最后一个红柱的index
    if last_idx + 1 == df.shape[0]:  # 红柱子是最后一个，没有出绿柱
        return False, None
    else:
        green_bar_len = df[last_idx + 1:].shape[0]
        if green_bar_len > math.ceil(_config.periods_config[k_period]['moutain_min_width'] / 2):
            cur_bar_len = df.iloc[-1][field]  # 当前计算出的长度
            pre_bar_1_len = df.iloc[-2][field]  # 理论上缩短的第一根（但是也许是最长的那一根）
            # 如果pre_bar_1是一个绿色峰顶，那么就没有必要和他左侧的继续进行比较了
            if df.at[df.shape[0] - 2, field_tag] == WaveType.GREEN_PEAK:
                pre_bar_2_len = float('-inf')
            else:
                pre_bar_2_len = df.iloc[-3][field]  # 理论上最长的那根,也可能是最长那根左侧的

            # 然后计算当前bar和最长bar的距离
            cur_bar_idx = df.tail(1).index[0]
            max_bar_idx = __get_last_possible_max_bar_idx(df, "macd_bar")
            is_reduce = (cur_bar_len > pre_bar_1_len and cur_bar_len > pre_bar_2_len) and \
                        (cur_bar_idx - max_bar_idx <= max_reduce_bar_distance)
            # TODO 这里还需要评估一下到底减少多少幅度/速度是最优的
            return is_reduce, df.iloc[last_idx + 1]['time_key']
        else:  # 最后的绿色柱子太少了
            return False, None


def __get_last_possible_max_bar_idx(df: DataFrame, field):
    """
    一段绿色区域被打标之后，如果新增一根，那么这根由于个数不足，无法判断是不是一个最大值。
    这个函数就是要找到这样的可能最大值的点，如果没有就返回打标的最大值点
    Parameters
    ----------
    df 一段连续的df

    Returns
    -------

    """
    dftemp = df.copy()
    dftemp.loc[dftemp[field] > 0, field] = 0  # 有可能是红色的区域设置为0
    dftemp[field] = dftemp[field].abs()
    # 从这里面找出来绿色峰底，然后有2种情况：1）存在；2）不存在
    # 1）存在，那么找到峰底之后的部分的最大值，然后返回
    # 2）不存在，直接找到最大值index返回
    field_tag = __ext_field(field, ext=RG_AreaTagFieldNameExt.BAR_WAVE_TAG)
    field_rg_tag = __ext_field(field, ext=RG_AreaTagFieldNameExt.RG_TAG)
    df2 = dftemp[dftemp[field_rg_tag] == WaveType.GREEN_VALLEY]
    if df2.shape[0] > 0:
        peak_idx = df2.tail(1).index[0]
        if peak_idx + 1 == dftemp.shape[0]:  # 最后一个是谷底，那么就把前面的峰底返回
            return dftemp[dftemp[field_rg_tag] == WaveType.GREEN_PEAK].index[0]
        else:
            return dftemp[peak_idx + 1:][field].idxmax(axis=0)
    else:
        return dftemp[field].idxmax(axis=0)


if __name__ == '__main__':
    """
    df -> df 格式化统一 -> macd_bar, em5, em10 -> macd_bar, em_bar -> 
    macd_bar 判别, macd_wave_scan em_bar_wave_scan -> 按权重评分 
    """
    # STOCK_CODE = 'SZ.002405'
    # prepare_csv_data([STOCK_CODE], n_days=90)
    # compute_df_bar([STOCK_CODE])
    # fname = df_file_name(STOCK_CODE, KLType.K_60M)
    # df60 = pd.read_csv(fname, index_col=0)
    # red_areas, blue_areas = find_successive_bar_areas(df60, 'macd_bar')
    # df_new = do_bar_wave_tag(df60, 'macd_bar', red_areas)
    # print(df_new.index)

    code = "SH.600703"
    df = get_df_of_code(code, "2019-09-20", "2019-10-21", KLType.K_30M)
    df15 = __do_compute_df_bar(df)
    ct = bottom_divergence_cnt(df15[:-4], "macd_bar", "close")
    print(ct)
    gct = bar_green_wave_cnt(df15[:-4])
    print(gct)
