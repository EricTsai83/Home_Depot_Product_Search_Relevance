# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
@author: Eric Tsai <eric492718@gmail.com>
@brief: utils for logging

"""

import os
import logging
import logging.handlers


# |等級      | 等級數值  |  輸出函數           |  說明    |
# |----------|----------|--------------------|:--------:|
# |NOTSET    |0         |無對應的輸出函數      |未設定    |
# |DEBUG     |10        |logging.debug()     |除錯       |
# |INFO      |20        |logging.info()      |訊息       |
# |WARNING   |30        |logging.warning()   |警告       |
# |ERROR     |40        |logging.error()     |錯誤       |
# |CRITICAL  |50        |logging.critical()  |嚴重錯誤   |

# ***This module(Logging) will write a log in model.log (which is a log file). When model.log exceeded the capacity limit which is module variable, model.log will rename to model.log.1 and so on. Eventually, if we set backupCount=3 than we will get model.log, model.log.1, and model.log.2. They record the newest to the oldest log.***<br>
#
# **Rotating File Handler (backupCount=3):**<br>
# If the log file exceeds the capacity limit, pass the path to another stage.
#      
#     new record 📃
#              ↓ (write in)
#              ↓
#              model.log → → model.log.1
#                      (rename)        ↓
#                                      ↓ (rename)
#            model.log.3 ← ← model.log.2
#            ↓         (rename)
#     (drop) ↓ 
#           🗑️

# loglevel: 記錄最低等級
def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = '[%(asctime)s] %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
                    filename=os.path.join(logdir, logname),
                    maxBytes=10*1024*1024, # 大小不超過 10 MB，若紀錄檔已超過就會轉換為backup的log檔，
                    backupCount=10)        # 並重新創建一個新的紀錄檔，紀錄新的log
    handler.setFormatter(formatter)

    logger = logging.getLogger('')
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger


# convert notebook.ipynb to a .py file
# !jupytext --to py logging_utils.ipynb


