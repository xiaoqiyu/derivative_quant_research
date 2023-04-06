#!/user/bin/env python
# coding=utf-8
'''
@project : option_future_research_private
@author  : rpyxqi@gmail.com
#@file   : MinSignal.py
#@time   : 2023-04-07 00:11:25
'''

from .Signal import Signal


class MinSignal(Signal):
    def __init__(self, factor, position, instrument_id, trade_date, product_id):
        super().__init__(factor, position)
        # _reg_param_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
        #                                define.TICK_MODEL_DIR,
        #                                'reg_params_{0}.json'.format(instrument_id))
        # self.reg_params = utils.load_json_file(_reg_param_name)
        self.reg_params = {}
        # TODO load model

    def __call__(self, *args, **kwargs):
        params = kwargs.get('params')  # is options
