# -*- coding: utf-8 -*-
# @time    : 2018/9/10 17:15
# @author  : huangyu
# @file    : logger.py

import logging


class Logger(object):
    def __init__(self, log_name='log.txt', log_level='INFO', logger=__name__, handler='ch,fh'):
        '''

        :param log_name: 日志保存文件名
        :param log_level:
        :param logger:
        :param handler: 添加fh（file handler）即保存到文件，ch(console handler)即输出到控制台
        '''

        self.logger = logging.getLogger(logger)
        self.logger.setLevel(log_level)

        fh = logging.FileHandler(log_name)
        fh.setLevel(log_level)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        log_format = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s] [%(filename)s:%(lineno)d]')

        fh.setFormatter(log_format)
        ch.setFormatter(log_format)

        handlers = handler.split(',')
        if not self.logger.handlers:
            if 'fh' in handlers:
                self.logger.addHandler(fh)
            if 'ch' in handlers:
                self.logger.addHandler(ch)

    def get_log(self):
        return self.logger
