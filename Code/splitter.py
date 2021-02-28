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
@brief: splitter for Homedepot project

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, cross_validate
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
plt.rcParams["figure.figsize"] = [5, 5]

import config
from utils import pkl_utils


## advanced splitter
class HomedepotSplitter:
    def __init__(self, dfTrain, dfTest, col, n_splits=5, random_state=config.RANDOM_SEED,
                    verbose=False, plot=False, split_param=[0.5, 0.25, 0.5]):
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.plot = plot
        self.split_param = split_param
        self.col = col

    def __str__(self):
        return "HomedepotSplitter"

    def _check_split(self, dfTrain, dfTest, col, suffix="", plot=False):
        """ 
        1. calculate actual training and test data proportion, data is provided by Kaggle competition
        2. calculate item proportion by a specific column. it can display three-item sets like training, test, and intersection item set
        3. plot venn diagram to show the result
        4. return specific column's unique items in train data (I think this step will let the function object be ambiguous, but I let it go.)
        """
        #====================================================================================
        if self.verbose:
            print("-"*50)
        num_train = self.dfTrain.shape[0]
        num_test = self.dfTest.shape[0]
        ratio_train = num_train/(num_train+num_test)
        ratio_test = num_test/(num_train+num_test)

        if self.verbose:
            print("Sample Stats: %.2f (train) | %.2f (test)" % (ratio_train, ratio_test))
        #====================================================================================
        col_train_set = set(np.unique(self.dfTrain[self.col]))
        col_test_set = set(np.unique(self.dfTest[self.col]))
        col_total_set = col_train_set.union(col_test_set)
        col_intersect_set = col_train_set.intersection(col_test_set)  # return col_train_set and col_test_set intersection

        ratio_train = ((len(col_train_set) - len(col_intersect_set)) / len(col_total_set))  # only in train data
        ratio_intersect = len(col_intersect_set) / len(col_total_set)  # set of intersection
        ratio_test = ((len(col_test_set) - len(col_intersect_set)) / len(col_total_set))  # only in test data

        if self.verbose:
            print("%s Stats: %.2f (train) | %.2f (train & test) | %.2f (test)" % (self.col, ratio_train, ratio_intersect, ratio_test))
        #====================================================================================
        if self.plot:
            plt.figure()
            if suffix == "actual":
                venn2([col_train_set, col_test_set], ("train", "test"))
            else:
                venn2([col_train_set, col_test_set], ("train", "valid"))
            fig_file = "%s/%s_%s.pdf"%(config.FIG_DIR, suffix, self.col)
            plt.savefig(fig_file)
            plt.clf()  # Clear figure
        #====================================================================================
        ## SORT it for reproducibility !!!
        col_train_set = sorted(list(col_train_set))  # sorted(): default is sort by A to Z
        return col_train_set

    def _get_df_idx(self, df, col, values):
        '''
        if you get a list and contain item we want to find, 
        use this function to find the index from the data frame
        '''
        return np.where(df[self.col].isin(values))[0]  # note: this method return index in order, not actual index
    
    def get_column_value_set(self, df, col):
        col_value_set = set(np.unique(df[col]))
        return col_value_set
    
    def split(self):
        """
        object: to let your validation data and new training data same as old training data and test data in
        relationship patterns (search_term and product_id intersect set )
        """
        ## original Train and Test Split
        if self.verbose:
            print("\n"+"*"*50)
            print("Original Train and Test Split")
        self._check_split(self.dfTrain, self.dfTest, self.col, "actual")
        col_value_set = self.get_column_value_set(self.dfTrain, self.col)
        col_value_set_li = list(col_value_set)

        ## naive split
        if self.verbose:
            print("\n"+"*"*50)
            print("Naive Split")

        ## show naive split result(split like training and test data proportion which are Kaggle offer)    
        rs = ShuffleSplit(n_splits=1, test_size=0.69, random_state=self.random_state)
        for trainInd, validInd in rs.split(self.dfTrain):
            dfTrain2 = self.dfTrain.iloc[trainInd].copy()
            dfValid = self.dfTrain.iloc[validInd].copy()
            self._check_split(dfTrain2, dfValid, self.col, "naive")

            
        ## split on product_uid & search_term and check which is better
        if self.verbose:
            print("\n"+"*"*50)
            print("Split on product_uid & search_term")

        self.splits = [0]*self.n_splits
        rs = ShuffleSplit(n_splits=self.n_splits, test_size=self.split_param[0], random_state=self.random_state)
        for run, (trInd, vaInd) in enumerate(rs.split(col_value_set_li)):
            if self.verbose:
                print("="*50)
            # let some search term be a common term which is appear in training data and test data
            ntr = int(len(trInd)*self.split_param[1])
            term_train2 = [col_value_set_li[i] for i in trInd[:ntr]]
            term_common = [col_value_set_li[i] for i in trInd[ntr:]]
            term_valid = [col_value_set_li[i] for i in vaInd]

            trainInd = self._get_df_idx(self.dfTrain, self.col, term_train2)
            commonInd = self._get_df_idx(self.dfTrain, self.col, term_common)
            validInd = self._get_df_idx(self.dfTrain, self.col, term_valid)


            # if search term only one value in data, then we need to split that feature without stratification
            # if not, split it with stratification
            dfTrain_index_reset = self.dfTrain.reset_index(drop=True)
            dfTrain_feature_order_map = list(zip(dfTrain_index_reset.index.values, dfTrain_index_reset['search_term'].values))
            group_feature = dfTrain_index_reset.iloc[commonInd].groupby(self.col).size().reset_index(name='size')
            value = group_feature.query("size==1")[self.col].values
            commonInd_value_more_than_one = dfTrain_index_reset.iloc[commonInd][~dfTrain_index_reset.iloc[commonInd][self.col].isin(value)].index.values
            commonInd_value_only_one = dfTrain_index_reset.iloc[commonInd][dfTrain_index_reset.iloc[commonInd][self.col].isin(value)].index.values

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.split_param[2], random_state=run)
            iidx, oidx = list(sss.split(np.zeros(len(commonInd_value_more_than_one)), 
                                        dfTrain_index_reset.iloc[commonInd_value_more_than_one]['search_term'])
                             )[0]

            trainInd = np.hstack((trainInd, commonInd_value_more_than_one[iidx]))  #np.hstack: Stack the arrays horizontally
            validInd = np.hstack((validInd, commonInd_value_more_than_one[oidx]))

            ss = ShuffleSplit(n_splits=1, test_size=self.split_param[2], random_state=run)
            iidx, oidx = list(ss.split(np.zeros(len(commonInd_value_only_one))))[0]
            trainInd = np.hstack((trainInd, commonInd_value_only_one[iidx]))
            validInd = np.hstack((validInd, commonInd_value_only_one[oidx]))


            trainInd = sorted(trainInd)
            validInd = sorted(validInd)

            if self.verbose:
                dfTrain2 = dfTrain_index_reset.iloc[trainInd].copy()
                dfValid = dfTrain_index_reset.iloc[validInd].copy()
                if run == 0:
                    plot = True
                else:
                    plot = False
                self._check_split(dfTrain2, dfValid, self.col, "proposed", plot)
    #             _check_split(dfTrain2, dfValid, "search_term", "proposed", plot)


            self.splits[run] = trainInd, validInd
        
            if self.verbose:
                print("-"*50)
                print("Index for run: %s" % (run+1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        return self
    
    def save(self, fname):
        pkl_utils._save(fname, self.splits)


def main():
    dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
    dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")

    # splits for level1
    splitter = HomedepotSplitter(dfTrain=dfTrain, 
                                 dfTest=dfTest,
                                 col = 'search_term',
                                 n_splits=config.N_RUNS, 
                                 random_state=config.RANDOM_SEED, 
                                 verbose=True,
                                 plot=True,
                                 # tune these params to get a close distribution
                                 split_param=[0.5, 0.25, 0.5],
                                )
    splitter.split()
    splitter.save("%s/splits_level1.pkl"%config.SPLIT_DIR)
    splits_level1 = splitter.splits
    
    ## splits for level2
    splits_level1 = pkl_utils._load("%s/splits_level1.pkl"%config.SPLIT_DIR)
    splits_level2 = [0]*config.N_RUNS
    for run, (trainInd, validInd) in enumerate(splits_level1):
        dfValid = dfTrain.iloc[validInd].copy()
        splitter2 = HomedepotSplitter(dfTrain=dfValid, 
                                      dfTest=dfTest, 
                                      col = 'search_term',
                                      n_splits=1, 
                                      random_state=run, 
                                      verbose=True,
                                      # tune these params to get a close distribution
                                      split_param=[0.5, 0.15, 0.6]
                                     )
        splitter2.split()
        splits_level2[run] = splitter2.splits[0]
    pkl_utils._save("%s/splits_level2.pkl"%config.SPLIT_DIR, splits_level2)

    ## splits for level3
    splits_level2 = pkl_utils._load("%s/splits_level2.pkl"%config.SPLIT_DIR)
    splits_level3 = [0]*config.N_RUNS
    for run, (trainInd, validInd) in enumerate(splits_level2):
        dfValid = dfTrain.iloc[validInd].copy()
        splitter3 = HomedepotSplitter(dfTrain=dfValid, 
                                    dfTest=dfTest,
                                    col = 'search_term',
                                    n_splits=1, 
                                    random_state=run, 
                                    verbose=True,
                                    # tune these params to get a close distribution
                                    split_param=[0.5, 0.15, 0.7])
        splitter3.split()
        splits_level3[run] = splitter3.splits[0]
    pkl_utils._save("%s/splits_level3.pkl"%config.SPLIT_DIR, splits_level3)


if __name__ == "__main__":
    main()