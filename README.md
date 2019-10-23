## useful api

### 交易日历
```python
from tushare.util.dateu import trade_cal
trade_cal()

>>
      calendarDate  isOpen
0       1990-12-19       1
1       1990-12-20       1
2       1990-12-21       1
3       1990-12-22       0
4       1990-12-23       0
5       1990-12-24       1
6       1990-12-25       1
7       1990-12-26       1
```


### 判断一个日期是不是交易日
```python
from tushare.util.dateu import is_holiday

is_holiday("2019-09-22") # True， 假期非交易日期
```


### 从某天向前推N个交易日
```python
from macdlib import n_trade_days_ago

date_str = n_trade_days_ago(3, '2019-03-03')# 获取2019-03-03包含当日（如果是交易日）前面3天交易日的开始日期

date_str2 = n_trade_days_ago(5)# 获取最近5个交易日的开始日期
```





--------------

### 几个交互式图表库

1. ployly express
2. pyechart
3. 