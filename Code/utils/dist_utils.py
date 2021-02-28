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
@author: Eric Tsai <eric492718g@gmail.com>
@brief: utils for distance computation

"""

# +
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import lzma
    import Levenshtein
except:
    pass
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

# 若是用 cmd 運行，會從一開始的 current path (若無更改起始目錄的話)查找 module，但 notebook 預設情況是以notebook的位置當作 current path 找 module
# 故將code資料夾的路徑(也就是上一層的路徑)加入系統路徑
sys.path.append("..")
from utils import np_utils
import config


# -

# ## Edit Distance

# * ### Levenshtein distance 

# The [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between two strings $a,b$ (of length $|a|$ and $|b|$ respectively) is given by $lev(a,b)$where
#
# ${\displaystyle \qquad \operatorname {lev} (a,b)={\begin{cases}|a|&{\text{ if }}|b|=0,\\|b|&{\text{ if }}|a|=0,\\\operatorname {lev} (\operatorname {tail} (a),\operatorname {tail} (b))&{\text{ if }}a[0]=b[0]\\1+\min {\begin{cases}\operatorname {lev} (\operatorname {tail} (a),b)\\\operatorname {lev} (a,\operatorname {tail} (b))\\\operatorname {lev} (\operatorname {tail} (a),\operatorname {tail} (b))\\\end{cases}}&{\text{ otherwise.}}\end{cases}}}$
#
# where the $tail$ of some string $x$ is a string of all but the first character of $x$, and $x[n]$ is the $n$th character of the string $x$, starting with character 0.

# #### A practical example
# ---
# The Levenshtein distance between "kitten" and "sitting" is three, since the following three edits change one into the other, and there is no way to do it with fewer than three edits:<br>
# kitten → sitten (substitution of 's' for 'k')<br>
# sitten → sittin (substitution of 'i' for 'e')<br>
# sittin → sitting (insertion of 'g' at the end)<br>
# We can then convert the difference into a percentage(Levenshtein ratio) using the following formula:<br>
# p = (1 - l/m) × 100<br>
# Where `l` is the levenshtein distance and `m` is the length of the longest of the two words:<br>
# (1 - 3/7) × 100 = 57.14...
# ---

# * ### SequenceMatcher().ratio()

# [Reference](https://docs.python.org/3/library/difflib.html)<br>
# Return a measure of the sequences’ similarity as a float in the range [0, 1].<br>
# Where `T` is the total number of elements in both sequences, and `M` is the number of matches, this is `2.0*M / T`.<br>
# Note that this is 1.0 if the sequences are identical, and 0.0 if they have nothing in common.

def _edit_dist(str1, str2):
    """
    caculate two string distance
    """
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d


def _is_str_match(str1, str2, threshold=1.0):
    """
    add a threshold parameter to decide whether two string are a match
    """
    assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - _edit_dist(str1, str2)) >= threshold


# * ### SequenceMatcher().find_longest_match()

# [Reference](https://docs.python.org/3/library/difflib.html)

def _longest_match_size(str1, str2):
    """
    find the longest matching block, and return the string length
    """
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def _longest_match_ratio(str1, str2):
    """
    find the longest matching block between string1 and string2, 
    and then calculate the string length divide by min(string1, string2)
    """
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return np_utils._try_divide(match.size, min(len(str1), len(str2)) )



# ## Normalized compression distance

# [Idea Reference](https://en.wikipedia.org/wiki/Normalized_compression_distance)
# [Code Reference](https://docs.python.org/3/library/lzma.html)

def _compression_dist(x, y, l_x=None, l_y=None):
    """
    compress data (a bytes object) and decide two string distance by calculating result length
    """
    if x == y:
        return 0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    dist = np_utils._try_divide(min(l_xy,l_yx)-min(l_x,l_y), max(l_x,l_y))
    return dist



# ## Distance Caculate by Vector

# ### Cosine Similarity

# [cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

def _cosine_sim(vec1, vec2):
    try:
        s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]  # convert vector to martix
    except:
        try:
            s = cosine_similarity(vec1, vec2)[0][0]
        except:
            s = config.MISSING_VALUE_NUMERIC
    return s



def _vdiff(vec1, vec2):
    return vec1 - vec2



def _rmse(vec1, vec2):
    vdiff = vec1 - vec2
    rmse = np.sqrt(np.mean(vdiff**2))
    return rmse



# ## Distance Caculate by distribution

# ### Kullback–Leibler Divergence
# [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)<br>
# For discrete probability distributions $P$ and $Q$ defined on the same probability space, $X$, the relative entropy from $Q$ to $P$ is defined to be<br>
# ${\displaystyle D_{\text{KL}}(P\parallel Q)=\sum _{x\in {\mathcal {X}}}P(x)\log \left({\frac {P(x)}{Q(x)}}\right)}$

def _KL(dist1, dist2):
    "Kullback-Leibler Divergence"
    return np.sum(dist1 * np.log(dist1/dist2), axis=1)


# ### Jaccard Similarity Coefficient
# [Jaccard similarity coefficient]('https://en.wikipedia.org/wiki/Jaccard_index')<br>
# Jaccard similarity coefficient is a statistic used for gauging the similarity and diversity of sample sets.<br>
# Given two sets A and B, the Jaccard coefficient is defined as the ratio of the size of the intersection of A and B to the size of the union of A and B.<br>
# ${\displaystyle J(A,B)={{|A\cap B|} \over {|A\cup B|}}={{|A\cap B|} \over {|A|+|B|-|A\cap B|}}}$

def _jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(float(len(A.intersection(B))), len(A.union(B)))


# ### Dice coefficient
# [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)<br>
# Dice coefficient original formula was intended to be applied to discrete data. Given two sets, X and Y, it is defined as<br>
# ${\displaystyle DSC={\frac {2|X\cap Y|}{|X|+|Y|}}}$

def _dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))

# convert notebook.ipynb to a .py file
# !jupytext --to py dist_utils.ipynb
