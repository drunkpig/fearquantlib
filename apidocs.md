### 配置文件
配置文件存放了量化时一些条件参数，比如波峰的最小宽度，波谷距离波峰的最小距离阈值等，是一个事关成败的文件。

#### 1. 配置文件位置
配置文件的寻找顺序为
1. 从用户HOME目录下的 .fear-quant/config.ini
2. 运行时项目入口文件平行的目录下寻找config.ini
当找到一个就停止，因此只有一个会起作用

#### 2. 配置文件格式


```ini
[application_env]
dev_model=dev_
futu_api_ip=localhost
futu_api_port=11111


[parameters]
wave_scan_max_gap=3
moutain_min_width = 5
n_days_bar_fetch=30

```
- application_env
    - dev_model 可以是dev_或者real_, 主要区分了数据是用来测试还是线上实时。这些数据会临时落盘，加个前缀容易区分
    - futu_api_ip 富途openapi的服务地址
    - futu_api_port 端口
    
- parameters
    - wave_scan_max_gap 在bar柱子中，一段上升/下降趋势中有一些毛刺，这个值可以实现当前（第N）条bar和第N-wave_scan_max_gap的bar值进行比较，从而进行平滑
    - moutain_min_width 一个红色或者绿色区域的最小宽度
    - n_days_bar_fetch 进行一次计算需要获取前面多少个交易日的数据


### 常量
```python
class KL_Period(object):
    KL_60 = "KL_60" # 60分K
    KL_30 = "KL_30"
    KL_15 = "KL_15"


class WaveType(object): # bar的波峰波谷定义的常量
    RED_PEAK = 2  # 红柱高峰
    RED_VALLEY = 1  # 红柱峰底

    GREEN_PEAK = -2  # 绿柱波峰
    GREEN_VALLEy = -1  # 绿柱波底，乳沟深V的尖

```


### API

#### 技术指标
- MA
- MACD

#### 日期
- today
- n_days_ago
- n_trade_days_ago

#### 数据源获取
- prepare_csv_data 从富途获取code list的多周期原始数据并存在磁盘上
- get_df_of_code 获取一个code, 一个周期的原始数据并保存在磁盘上

#### 数据处理
- df_file_name 根据code和K线类型获得数据文件磁盘位置
- compute_df_bar 对传入的code列表利用准备好的数据(磁盘文件)分别计算出新df, df新生成字段包括：macd_bar, em_bar, macd_bar_rg_tag, em_bar_rg_tag。调用了__add_df_tags

- do_bar_wave_tag 对传入的一段连续区域寻找里面的波峰和波谷
- bottom_divergence_cnt 底部背离了多少次
- bar_green_wave_cnt 当前绿柱连续区域有多少个波峰

- ma_distance 5周期和10周期线的距离占当前收盘价的百分点，返回4位小数
- resonance_cnt 2个df的共振次数
- is_macd_bar_reduce 当前一根绿柱是否变短了

- __find_successive_bar_areas 对于传入的df, 寻找指定字段的 红色/绿色连续区域，并返回找到的红色和绿色区域数组，数组里每个元素是区域的开始和结束下标

- __get_last_successive_rg_area  获得最后一端某字段的g/r连续区域
- __add_df_tags  为某个字段加入r/g标签
- __ext_field 标签扩展字段，例如r/g字段
- __do_find_successive_areas 给一个list，取出其中的连续区域，返回一个tuple list, tuple包含了连续区域开始和结束下标
