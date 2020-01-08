import os
import sys
import logging
import json

logger = logging.getLogger()

# 周、天、60分等周期（交易时间）等于多少个5分钟单位的一个换算表
timeConvTable = {
    "KL_60" : 12,
    "KL_30" : 6,
    "KL_15" : 3,
}


class QuantConfig(object):
    """

    """
    def __init__(self, config_file='~/.fear-quant/config.json'):
        self.config_file = os.path.expanduser(config_file)
        logger.info(f"使用配置文件{self.config_file}")
        with open(self.config_file, 'r', encoding='utf-8') as f:
            obj = json.load(f)

            self.periods = []
            self.periods_config = {}
            parameters = obj['parameters']
            self.n_days_bar_fetch = parameters['n_days_bar_fetch']
            periods = parameters['periods']
            for p in periods:
                periods_cfg = parameters[p]
                if periods_cfg['enable']:
                    self.periods.append(p)
                    wave_scan_max_gap=periods_cfg['wave_scan_max_gap']
                    moutain_min_width=periods_cfg['moutain_min_width']
                    self.periods_config[p] = {"wave_scan_max_gap":wave_scan_max_gap, "moutain_min_width":moutain_min_width}


if __name__=="__main__":
    cfg = QuantConfig()
    print(cfg.periods)

