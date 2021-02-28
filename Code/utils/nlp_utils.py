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
@brief: utils for nlp

"""

import re


def _tokenize(text, token_pattern=" "):
    # token_pattern = r"(?u)\b\w\w+\b"
    # token_pattern = r"\w{1,}"
    # token_pattern = r"\w+"
    # token_pattern = r"[\w']+"
    if token_pattern == " ":
        # just split the text into tokens
        return text.split(" ")
    else:
        token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)  # compile a regular expression pattern into a regular expression object
        group = token_pattern.findall(text)
        return group
