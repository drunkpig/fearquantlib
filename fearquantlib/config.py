import os
import sys
from configparser import ConfigParser
import logging

logger = logging.getLogger()


class QuantConfig(object):
    """

    """
    DEV_MODEL = 'dev_'
    futu_api_ip = 'localhost'
    futu_api_port = 11111
    wave_scan_max_gap = 3  # 扫描波峰时最大跳跃间隔
    moutain_min_width = 5
    n_days_bar_fetch = 30  # 向前看多少天的60分数据


"""
配置首先从~/.fear-quant/config.ini读取
其次读取运行时目录下config.ini
如果都不存在报错，如果两个都存在，只使用前者
"""
home_config_file = os.path.expanduser('~/.fear-quant/config.ini')
runtime_config_file = os.path.dirname(os.path.realpath(sys.argv[0]))  # 运行时目录

if os.path.exists(home_config_file):
    config_file = home_config_file
elif os.path.exists(runtime_config_file):
    config_file = runtime_config_file
else:
    logger.error("没有找到配置文件")
    raise Exception("没有找到配置文件")

cfg = ConfigParser()
cfg.read(config_file, encoding="utf-8")
QuantConfig.DEV_MODEL = str(cfg.get("application_env", 'dev_model'))
QuantConfig.futu_api_ip = str(cfg.get("application_env", 'futu_api_ip'))
QuantConfig.futu_api_port = cfg.getint("application_env", 'futu_api_port')

QuantConfig.wave_scan_max_gap = cfg.getint("parameters", 'wave_scan_max_gap')
QuantConfig.moutain_min_width = cfg.getint("parameters", 'moutain_min_width')
QuantConfig.n_days_bar_fetch = cfg.getint("parameters", 'n_days_bar_fetch')
