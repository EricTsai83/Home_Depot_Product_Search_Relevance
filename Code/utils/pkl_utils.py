# -*- coding: utf-8 -*-
"""
@author: Eric Tsai <eric492718@gmail.com>
@brief: utils for pickle

"""

import pickle

# pickle.HIGHEST_PROTOCOL will always be the right version for the current Python version. Because this is a binary format, make sure to use 'wb' as the file mode.
def _save(fname, data, protocol=pickle.HIGHEST_PROTOCOL):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
