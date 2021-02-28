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
@brief: utils for numpy

"""

import sys
import numpy as np
from scipy.stats import pearsonr
from collections import Counter
# sys.path是Python的搜尋模組的路徑集，是一個list
# 可以在 python 環境下使用sys.path.append(path)新增相關的路徑，但在退出python環境後自己新增的路徑就會自動消失了
sys.path.append("..") # in order to import config
import config


# +
# # Import matplotlib, numpy and math 
# import matplotlib.pyplot as plt 
# import numpy as np 
# import math 
# -

# ## 1. Sigmoid
# $$y = \frac{1}{1 + e^{-x}}$$

def _sigmoid(score):
    p = 1. / (1. + np.exp(-score))
    return p


# ### 1.1 Sigmoid Graph

# +
# x = np.linspace(-10, 10, 1000) 
# y = 1/(1 + np.exp(-x)) 
# plt.plot(x, y) 
# plt.xlabel("x") 
# plt.ylabel("Sigmoid(X)") 
# plt.show() 
# -

# ## 2. Logit

# $$y = log(\frac{x}{1-x})$$

def _logit(p):
    return np.log(p/(1.-p))


# ### 2.1 Logit Graph

# +
# x = np.linspace(0.0001, 0.9999, 1000) 
# y = np.log(x/(1.-x))
# plt.plot(x, y) 
# plt.xlabel("x") 
# plt.ylabel("Logit(X)") 
# plt.show() 
# -

# ## 3. Softmax

# $$q(i) = P_{W,b}(Y=i|X) = \frac {e^{W_i X + b_i}} {\sum_j e^{W_j X + b_j}} = \frac {d_i} {\sum_j d_j} = \frac {e^{W_i X + b_i - max(W_j X + b_j)}} {\sum_j e^{W_j X + b_j - max(W_j X + b_j)}}$$

def _softmax(score):
    """
    Compute the softmax function for each row of the input x.
    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    score = np.asarray(score, dtype=float)
    score = np.exp(score - np.max(score))
    try:
        score /= np.sum(score, axis=1)[:,np.newaxis]  # 增加原本 array 的維度 array[np.newaxis,:]增加 row 的維度，增加 column 的維度 array[:,np.newaxis]
    except:
        score /= np.sum(score, axis=0)
    return score


# ### 3.1 Softmax Output Check

# +
# x = np.linspace(-10, 10, 10) 
# y = _softmax(x)
# print(f"==========\nInput:\n==========\n{x}")
# print(f"==========\nOutput:\n==========\n{y}")
# print(f"==========\nOutput sum:\n==========\n {sum(y)}")
# -

def _cast_proba_predict(proba):
    N = proba.shape[1]
    w = np.arange(1,N+1)
    pred = proba * w[np.newaxis,:]
    pred = np.sum(pred, axis=1)
    return pred


def _one_hot_label(label, n_classes):
    """
    label(array): digital label represent each class
    n_class(int): number of class
    """
    num = label.shape[0]
    tmp = np.zeros((num, n_classes), dtype=int)
    tmp[np.arange(num),label.astype(int)] = 1
    return tmp


# +
# label=np.array([1,2,0,3,6])

# +
# _one_hot_label(label, n_classes=7)
# -

def _majority_voting(x, weight=None):
    ## apply weight
    if weight is not None:
        assert len(weight) == len(x)
        x = np.repeat(x, weight)
    c = Counter(x)
    value, count = c.most_common()[0]
    return value


def _voter(x, weight=None):
    idx = np.isfinite(x)  # Test element-wise for finiteness (not infinity or not Not a Number).
    if sum(idx) == 0:
        value = config.MISSING_VALUE_NUMERIC
    else:
        if weight is not None:
            value = _majority_voting(x[idx], weight[idx])
        else:
            value = _majority_voting(x[idx])
    return value


# +
# res=np.array([1,3,3,3,6])

# +
# np.isfinite(res)

# +
# _voter(res)

# +
# _voter(res, np.array([6,1,1,1,5]))
# -

def _array_majority_voting(X, weight=None):
    """
    majority voting by column
    note: if a tie, return the smallest result
    """
    # sometime, your function got an 'if' statement, then you can't use np.vectorize for iteration, use '.apply_along_axis'
    y = np.apply_along_axis(_voter, axis=1, arr=X, weight=weight)
    return y


# +
# res=np.array([[1,3,1,5,0],  # 1
#               [0,2,3,0,6],  # 0
#               [1,3,2,3,6],  # 3
#               [1,4,2,3,6],  # tie
#               [1,5,2,3,6]])  # tie

# +
# _array_majority_voting(res, weight=None)
# -

def _mean(x):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        value = float(config.MISSING_VALUE_NUMERIC) # cast it to float to accommodate the np.mean
    else:
        value = np.mean(x[idx]) # this is float!
    return value


# +
# res=np.array([1,3,3,3,6])

# +
# _mean(res)
# -

def _array_mean(X):
    y = np.apply_along_axis(_mean, axis=1, arr=X)
    return y


res=np.array([[1,3,1,5,0],
              [0,2,3,0,6],
              [1,3,2,3,6],
              [1,4,2,3,6],
              [1,5,2,3,6]])


# +
# _array_mean(res)
# -

def _dim(x):
    d = 1 if len(x.shape) == 1 else x.shape[1]
    return d


def _corr(x, y_train):
    if _dim(x) == 1:
        corr = pearsonr(x.flatten(), y_train)[0]
        if str(corr) == "nan":
            corr = 0.
    else:
        corr = 1.
    return corr


# +
# x=np.array([1,3,3,3,6])
# y_train=np.array([1,3,3,3,6])

# +
# _corr(x, y_train)
# -

def _entropy(proba):
    entropy = -np.sum(proba*np.log(proba))
    return entropy

# +
# p1_li = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]
# p2_li = [1-i for i in p1_li]
# prob_li = list(zip(p1_li, p2_li))
# entropy_li = []
# for i, t in prob_li:
#     proba=np.array([i, t])
#     y = _entropy(proba)
#     entropy_li.append(y)

# +
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()

# X = p1_li
# Y = entropy_li

# ax1.plot(X,Y)
# ax1.set_xlabel("p1")
# ax1.set_ylabel("entropy(p1,p2)")

# new_tick_locations = np.array(p2_li)

# def tick_function(X):
#     return ["%.1f" % z for z in X]

# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(X)
# ax2.set_xticklabels(tick_function(new_tick_locations))
# ax2.set_xlabel("P2")


# plt.show()
# -

def _try_divide(x, y, val=0.0):
    """try to divide two numbers"""
    if y != 0.0:
        val = float(x) / y
    return val

# +
# _try_divide(10,2)

# +
# _try_divide(10,0)
# -

# convert notebook.ipynb to a .py file
# !jupytext --to py np_utils.ipynb






