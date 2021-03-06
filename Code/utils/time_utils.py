# -*- coding: utf-8 -*-
"""
@author: Eric Tsai <eric492718@gmail.com>
@brief: utils for time

"""

import datetime


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M")
    return now_str


def _timestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str
