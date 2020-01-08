import math
from itertools import groupby
from operator import itemgetter
from pandas import DataFrame
from fearquantlib.utils import MA, MACD


class RGAreaTagFieldName(object):
    BAR_WAVE_TAG_NAME = 'tag'
    RG_TAG_NAME = 'rg_tag'


class RGAreaTagValue(object):
    GREEN = 'g'
    RED = 'r'


class WaveType(object):
    RED_PEAK = 2  # 红柱高峰
    RED_VALLEY = 1  # 红柱峰底

    GREEN_PEAK = -2  # 绿柱波峰
    GREEN_VALLEY = -1  # 绿柱波底，乳沟深V的尖


def __find_successive_bar_areas(df: DataFrame, field='bar'):
    """
    这个地方不管宽度，只管找连续的区域
    @param df:
    @param field:
    @return:
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
    @param arr: 下标index
    @return:
    """
    successive_area = []
    for k, g in groupby(enumerate(arr), lambda iv: iv[0] - iv[1]):
        index_group = list(map(itemgetter(1), g))
        successive_area.append((min(index_group), max(index_group)))

    return successive_area


def __ext_field(field_name, ext=RGAreaTagFieldName.BAR_WAVE_TAG_NAME):
    """

    @param field_name:
    @param ext:
    @return:
    """
    return f'_{field_name}_{ext}'


def __do_bar_wave_tag(raw_df: DataFrame, field, successive_bar_area, moutain_min_width=5):
    """
    这里找波峰和波谷，找谷底的目的是为了测量波峰/谷的斜率
    这个函数会把负值变成正值处理，如果是绿峰需要对返回值进行处理
    :param raw_df:
    :param field:
    :param successive_bar_area: 想同样色柱子区域, [(start1, end1),(start2, end2),...]
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
                if i != max_row_index:  # 这是为了防止边界上的是最大峰，被覆盖掉
                    df.at[i, tag_field] = WaveType.RED_VALLEY
                if j != max_row_index:
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


def __max_successive_series_len(arr, asc=True, eq=False):
    """
    寻找最大连续子序列，子序列的下标必须是相连的
    例如， 1,2,3 返回3
    @param arr:
    @param asc:
    @param eq:
    @return:
    """
    max_area_len = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if asc and (arr[j] > arr[j - 1]):
                max_area_len = max(j - i + 1, max_area_len)
            elif not asc and (arr[j] < arr[j - 1]):
                max_area_len = max(j - i + 1, max_area_len)
            else:
                i = j

    return max_area_len


def compute_df_bar(df, **kwargs):
    """

    @param df:
    @param kwargs:与df所属周期绑定的配置参数
    @return:
    """
    diff, dem, bar = MACD(df)
    df['macd_bar'] = bar  # macd
    df = MA(df, 5, 'close', 'ma5')
    df = MA(df, 10, 'close', 'ma10')
    df['em_bar'] = (df['ma5'] - df['ma10']).apply(lambda val: round(val, 2))  # 均线距离
    __add_peak_valley_and_successive_area_tag(df, "macd_bar", **kwargs)  # 顶部、谷底、连续区域打标r/g
    __add_peak_valley_and_successive_area_tag(df, "em_bar", **kwargs)  # r/g
    return df


def __add_peak_valley_and_successive_area_tag(df: DataFrame, field, **kwargs):
    """
    field 的连续区域以及顶底
    @param df:
    @param field:
    @param kwargs: 与df所属周期绑定的配置参数
    @return:
    """
    moutain_min_width = kwargs['moutain_min_width']
    tag_field_name = __ext_field(field)
    red_areas, blue_areas = __find_successive_bar_areas(df, field)
    df_blue = __do_bar_wave_tag(df, field, blue_areas, moutain_min_width=moutain_min_width)
    df_blue[tag_field_name] *= -1  # 因为计算都变为正值，所以绿柱子要乘以-1
    df_red = __do_bar_wave_tag(df, field, red_areas, moutain_min_width=moutain_min_width)
    df[tag_field_name] = df_red[tag_field_name] | df_blue[tag_field_name]

    # 连续的区域用r, g区分，方便后续计算
    color_tag_field = __ext_field(field, ext=RGAreaTagFieldName.RG_TAG_NAME)
    df[color_tag_field] = RGAreaTagValue.GREEN  # 先设置绿色很重要，因为可以把宽度太短的连续区域过滤掉

    for s, e in red_areas:  # 过滤掉红色过小的区域，这里还要注意一点，如果可被过滤的短区域存在最后，也是要保留成红色的
        if e + 1 == df.shape[0]:  # 最后是出现了红色，那么无论多长就要保留,原因是可能继续增长，代表了趋势的方向
            df.loc[s:e, color_tag_field] = RGAreaTagValue.RED
        elif e - s + 1 >= moutain_min_width:
            df.loc[s:e, color_tag_field] = RGAreaTagValue.RED


def bottom_divergence_cnt(df: DataFrame, bar_field, value_field, start_time_key=None):
    """
    field字段出现连续背离的个数,也既多重背离个数。
    背离必须是连续的。

    方法是：找出最后一段绿色bar_field（为了计算方便负值转为正值），value_field顶点的值，形成两个array
        找到bar_field中连续的下降山峰（一峰更比一峰高/低）个数S_1，value_field中连续的下降山峰个数S_2。最后返回max(S_1,S_2)
        为什么选最大而不是最小呢？其实最小也可以，但里面涉及到一些模糊的东西，把阈值设大，然后还要
        辅助人工交易，如果取了min过于严格会误杀很多。

    :param df:
    :param bar_field: bar的field名字
    :param value_field:  价格
    :return: 没有背离为0，
    """
    rg_tag_name = __ext_field(bar_field, ext=RGAreaTagFieldName.RG_TAG_NAME) #红绿
    field_tag_name = __ext_field(bar_field, ext=RGAreaTagFieldName.BAR_WAVE_TAG_NAME) #波峰、波谷
    # 一般情况下这个start_time_key属于大周期，比如60分，当60分出现绿色的时候，小周期肯定是提前的。因此这里要加以处理
    if start_time_key is not None:
        dftemp = __get_real_successive_rg_area(df, rg_tag_name, start_time_key, RGAreaTagValue.GREEN)
    else:
        dftemp = __get_last_successive_rg_area(df, rg_tag_name, area=RGAreaTagValue.GREEN)  # 获得最后一段连续绿色区域
    # 这一段连续区域里包含了被同化的不同色，需要对这部分对应的值进行处理，等于0是个办法
    # 对应于底背离，应该是bar_field>0, 但是 颜色标记为G的那些，因为颜色全都是G，因此只需要
    # 把bar_field>0的全都设置为0即可
    dftemp.loc[dftemp[bar_field] > 0, bar_field] = 0
    # TODO 对最后一段进行打tag，要做一定的预测行为?绿峰一直在增长但背离的情况
    bar_array = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK][bar_field].abs().array  # 最后一段连续绿色区域的bar的顶点
    val_array = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK][value_field].array
    # # 然后找出来最大长度的区
    bar_desc = __max_successive_series_len(bar_array, asc=False)
    val_desc = __max_successive_series_len(val_array, asc=False)
    cnt = min(bar_desc, val_desc) - 1  # 背离取最小
    return max(0, cnt)  # 防止小于0


def bar_green_wave_cnt(df: DataFrame, bar_field='macd_bar', start_time_key=None):
    """
    在一段连续的绿柱子区间，当前的波峰是第几个
    方法是：从当前时间开始找到前面第一段连续绿柱，然后计算绿柱区间有几个波峰；
    如果当前是红柱但是没超过设置的最大宽度，可以忽略这段红柱
    @param df:
    @param bar_field:
    @param start_time_key:
    @return:波峰个数, 默认1
    """
    if df.shape[0] == 0:
        return 0
    field_tag_name = __ext_field(bar_field, ext=RGAreaTagFieldName.BAR_WAVE_TAG_NAME)
    rg_tag_name = __ext_field(bar_field, ext=RGAreaTagFieldName.RG_TAG_NAME)
    if start_time_key is not None:
        dftemp = __get_real_successive_rg_area(df, rg_tag_name, start_time_key, area=RGAreaTagValue.GREEN)
    else:
        dftemp = __get_last_successive_rg_area(df, rg_tag_name, area=RGAreaTagValue.GREEN)  # 获得最后一段连续绿色区域
    wave_cnt = dftemp[dftemp[field_tag_name] == WaveType.GREEN_PEAK].shape[0]
    # TODO 这个地方有点问题，对于最后一段区域需要进一步处理，做一定预测。当前的GREEN_TOP在实时中不一定被打标
    return wave_cnt


def get_current_ma_distance(df: DataFrame):
    """
    计算(ma(5)-ma(10))/close，保留2位小数
    @param df:
    @return:0.0131 ， 0.0323， 保留3位小数
    """
    close_price = df.at[df.shape[0] - 1, 'close']
    ma_gap = df.at[df.shape[0] - 1, 'em_bar']
    return round(abs(ma_gap / close_price), 3)


def __get_real_successive_rg_area(df: DataFrame, tag_name, start_time_key, area=RGAreaTagValue.GREEN):
    """
    start_time_key是大周期的green bar开始时间，如果用在小周期上，因为小周期green bar提前于大周期，因此在
    小周期上start_time_key之前也有可能有属于小周期的连续green bar
    @param df:
    @param tag_name:
    @param start_time_key:
    @param area:
    @return:
    """
    # 找到start_time_key的index, 然后取df[:start_time_key], 找到最后一个RG_AreaTag.RED， 这一行的下一个就是连续绿色的真正起始地址
    start_time_key_idx = df[df.time_key == start_time_key].index[0]
    dfbefore = df[:start_time_key_idx]
    other_area = dfbefore[dfbefore[tag_name] != area]
    if other_area.shape[0] != 0:  # 从这个颜色最后一个开始切分，要后面的
        idx = other_area.tail(1).index[0]
        dftemp = df[idx + 1:].copy().reset_index(drop=True)
    else:
        dftemp = df.copy().reset_index(drop=True)

    return dftemp


def __get_last_successive_rg_area(df: DataFrame, rg_field_name, area=RGAreaTagValue.GREEN):
    """
    获取最后一段颜色为area的连续区域
    @param df:
    @param rg_field_name:
    @param area:
    @return:
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
    @param df1:
    @param df2:
    @param field:
    @param start_time_key:
    @return:
    """
    rg_tag_name = __ext_field(field, ext=RGAreaTagFieldName.RG_TAG_NAME)
    if start_time_key is not None:
        area1 = __get_real_successive_rg_area(df1, rg_tag_name, start_time_key, RGAreaTagValue.GREEN)
        area2 = __get_real_successive_rg_area(df2, rg_tag_name, start_time_key, RGAreaTagValue.GREEN)
    else:
        area1 = __get_last_successive_rg_area(df1, rg_tag_name, area=RGAreaTagValue.GREEN)
        area2 = __get_last_successive_rg_area(df2, rg_tag_name, area=RGAreaTagValue.GREEN)
    wave_1 = bar_green_wave_cnt(area1, field)
    wave_2 = bar_green_wave_cnt(area2, field)
    return max(0, min(wave_1, wave_2) - 1)  # 2个波形成1个共振


def is_macd_bar_reduce(df: DataFrame, field='macd_bar', max_reduce_bar_distance=4, **kwargs):
    """
    macd 绿柱子第一根减少出现，不能减少太剧烈，前面的绿色柱子不能太少
    前提是：最后一段柱子必须是绿色的
    @param df:
    @param field:
    @param k_period:
    @param max_reduce_bar_distance:
    @return:减少返回True, 否则False, 第二个参数是最后一根绿色柱子出现的日期
    """
    field_rg_tag_name = __ext_field(field, ext=RGAreaTagFieldName.RG_TAG_NAME)
    field_tag = __ext_field(field, ext=RGAreaTagFieldName.BAR_WAVE_TAG_NAME)
    last_idx = df[df[field_rg_tag_name] == RGAreaTagValue.RED].tail(1).index[0]  # 最后一个红柱的index
    if last_idx + 1 == df.shape[0]:  # 红柱子是最后一个，没有出绿柱
        return False, None
    else:
        green_bar_len = df[last_idx + 1:].shape[0]
        if green_bar_len > math.ceil(kwargs['moutain_min_width'] / 2):
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
    @param df:一段连续的df
    @param field:
    @return:
    """
    dftemp = df.copy()
    dftemp.loc[dftemp[field] > 0, field] = 0  # 有可能是红色的区域设置为0
    dftemp[field] = dftemp[field].abs()
    # 从这里面找出来绿色峰底，然后有2种情况：1）存在；2）不存在
    # 1）存在，那么找到峰底之后的部分的最大值，然后返回
    # 2）不存在，直接找到最大值index返回
    field_tag = __ext_field(field, ext=RGAreaTagFieldName.BAR_WAVE_TAG_NAME)
    field_rg_tag = __ext_field(field, ext=RGAreaTagFieldName.RG_TAG_NAME)
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

