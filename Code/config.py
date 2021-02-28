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

# # Creating a Basic Configuration File

# +
# can return some system information
import platform

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# -

#
# <div class="alert alert-warning" role="alert">
# <strong>Note!</strong> Sometimes, we need to test our process work or not. So we need to test small sample and make sure the programme run smoothly and correctly.
# </div>

# ---------------------- Overall -----------------------
TASK = 'sample'  #'all'
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 200

# size
TRAIN_SIZE = 74067
if TASK == 'sample':
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 166693
VALID_SIZE_MAX = 60000 # 0.7 * TRAIN_SIZE

# +
# ------------------------ PATH ------------------------
# Root directory
ROOT_DIR = '..' 

# Subdirectory

## data
DATA_DIR = f'{ROOT_DIR}/Data'
### clean data
CLEAN_DATA_DIR = f'{DATA_DIR}/Clean'

## Feature
FEAT_DIR = f'{ROOT_DIR}/Feat'
FEAT_FILE_SUFFIX = ".pkl"

## Code
CODE_DIR = f'{ROOT_DIR}/Code'
### feature config
FEAT_CONF_DIR = f'{CODE_DIR}/conf'

## Figure
FIG_DIR = f'{ROOT_DIR}/Fig'

## Log
LOG_DIR = f'{ROOT_DIR}/Log'

## output 
OUTPUT_DIR = f'{ROOT_DIR}/Output'
### submit output
SUBM_DIR = f'{OUTPUT_DIR}/Subm'

## Temporary folder
TMP_DIR = f'{ROOT_DIR}/Tmp'

## Thirdparty folder
THIRDPARTY_DIR = f'{ROOT_DIR}/Thirdparty'

# dictionary
WORD_REPLACER_DATA = "%s/dict/word_replacer.csv"%DATA_DIR

# colors
COLOR_DATA = "%s/dict/color_data.py"%DATA_DIR


# index split
SPLIT_DIR = "%s/split"%DATA_DIR

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = f'{DATA_DIR}/train.csv'
TEST_DATA = f'{DATA_DIR}/test.csv'
ATTR_DATA = f'{DATA_DIR}/attributes.csv'
DESC_DATA = f'{DATA_DIR}/product_descriptions.csv'
SAMPLE_DATA = f'{DATA_DIR}/sample_submission.csv'

ALL_DATA_RAW = f'{CLEAN_DATA_DIR}/all.raw.csv.pkl'
INFO_DATA = f'{CLEAN_DATA_DIR}/info.csv.pkl'
ALL_DATA_LEMMATIZED = f'{CLEAN_DATA_DIR}/all.lemmatized.csv.pkl'
ALL_DATA_LEMMATIZED_STEMMED = f'{CLEAN_DATA_DIR}/all.lemmatized.stemmed.csv.pkl'


# ------------------------ PARAM ------------------------
# missing value
MISSING_VALUE_STRING = 'MISSINGVALUE'   # str type
MISSING_VALUE_NUMERIC = -1.   # float type

# attribute name and value SEPARATOR
ATTR_SEPARATOR = " | "


# correct query with google spelling check dict
# turn this on/off to have two versions of features/models
# which is useful for ensembling, but not used in final submission
GOOGLE_CORRECTING_QUERY = False


# auto correcting query (quite time consuming; not used in final submission)
AUTO_CORRECTING_QUERY = False 


# query expansion (not used in final submission)
QUERY_EXPANSION = False

# stop words
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# cv
N_RUNS = 5
N_FOLDS = 1


# intersect count/match
STR_MATCH_THRESHOLD = 0.85

# ------------------------ OTHER ------------------------
PLATFORM = platform.system()

DATA_PROCESSOR_N_JOBS = 2 if PLATFORM == "Windows" else 6  # my notebook only two core. So sad.

AUTO_SPELLING_CHECKER_N_JOBS = 4 if PLATFORM == "Windows" else 8
# multi processing is not faster
AUTO_SPELLING_CHECKER_N_JOBS = 1

RANDOM_SEED = 2021
# -

# ## Convert notebook to python script

# convert notebook.ipynb to a .py file
# !jupytext --to py config.ipynb




