import os
import sys
import logging
import json

logger = logging.getLogger()


class QuantConfig(object):
    """

    """
    def __init__(self):
        config_file = self.__get_config_file()
        with open(config_file, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            env = obj['application_env']
            self.dev_model = env['dev_model']
            self.futu_api_ip = env['futu_api_ip']
            self.futu_api_port = env['futu_api_port']

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

    @staticmethod
    def __get_config_file():
        """
                配置文件采用json格式，
                搜索顺序如下：
                1）命令行第一个参数里找，
                2）~/.fear-quant/config.ini读取
                3）运行时目录下config.ini
                直到遇到第一个就启用这个配置，同时忽略其他配置文件。如果没有发现配置文件则报错退出。
                """
        if len(sys.argv)>=2:
            cmd_config_file = sys.argv[1]
        else:
            cmd_config_file = None
        home_config_file = os.path.expanduser('~/.fear-quant/config.json')
        runtime_config_file = os.path.dirname(os.path.realpath(sys.argv[0]))  # 运行时目录
        runtime_config_file = f"{runtime_config_file}/config.json"

        if cmd_config_file and os.path.exists(cmd_config_file):
            config_file = cmd_config_file
            logger.info("使用命令行配置文件")
        elif os.path.exists(home_config_file):
            config_file = home_config_file
            logger.info("使用家目录下配置文件")
        elif os.path.exists(runtime_config_file):
            config_file = runtime_config_file
            logger.info("使用工程目录下配置文件")
        else:
            logger.error("没有找到配置文件")
            raise Exception("没有找到配置文件")

        return config_file


if __name__=="__main__":
    cfg = QuantConfig()
    print(cfg.periods)

