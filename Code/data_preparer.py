#!/usr/bin/env python
# coding: utf-8

# # Data pre-processing

# In[ ]:


"""
@author: Eric Tsai <eric492718@gmail.com>
@brief: generate raw dataframe data

"""


# ## Table of contents
# * [1. Import libraries](#1.-Import-libraries)
# * [2. Design Working Directory](#2.-Design-Working-Directory)
# * [3. Creating a Dataset](#3.-Creating-a-Dataset)

# ## 1. Import libraries

# In[2]:


import gc

import numpy as np
import pandas as pd

# import config
from utils import pkl_utils
import config

import os


# ## 2. Design Working Directory
# Synchronize Working Directory to config.py

# <div class="alert alert-warning" role="alert">
#   <strong>Note!</strong> Synchronize Working Directory to config.py 
# </div>

# ### EX.
# ---
#     ./
#     ├── A
#     │   ├── B
#     │   └── C
#     ├── D
#     │   └── E
#     └── F
# ---
#     ./
#     ├── EDA
#     ├── Data
#     │   └── Clean (clean data)
#     ├── Feat
#     ├── Code
#     │    └── Conf (feature config)
#     ├── Fig
#     ├── Log    
#     ├── Output
#     └── README.md

# ## 3. Creating a Dataset

# ***Because in the real world, we usually get the whole dataset and split the training and test data by ourseleve.
# So I will like to start with making a dataset, and then create a complete API. It will help me to reproduce analysis (or model) structure easily.***

# In[3]:


# load provided data
dfTrain = pd.read_csv(config.TRAIN_DATA, encoding="ISO-8859-1")
dfTest = pd.read_csv(config.TEST_DATA, encoding="ISO-8859-1")
dfAttr = pd.read_csv(config.ATTR_DATA)
dfDesc = pd.read_csv(config.DESC_DATA)


# In[4]:


print('--------------------------------------------------------------------------------------------------------')
print('Train')
print(f'shape: {dfTrain.shape}, column: {dfTrain.columns.values}')
print('--------------------------------------------------------------------------------------------------------')
print('Test')
print(f'shape: {dfTest.shape}, column: {dfTest.columns.values}')
print('--------------------------------------------------------------------------------------------------------')
print('Attr')
print(f'shape: {dfAttr.shape}, column: {dfAttr.columns.values}')
print('--------------------------------------------------------------------------------------------------------')
print('Desc')
print(f'shape: {dfDesc.shape}, column: {dfDesc.columns.values}')
print('--------------------------------------------------------------------------------------------------------')


# In[5]:


# 
print("Train Mean: %.6f"%np.mean(dfTrain["relevance"]))
print("Train Var: %.6f"%np.var(dfTrain["relevance"]))


# In[6]:


#
dfTest["relevance"] = np.zeros((config.TEST_SIZE))
dfAttr.dropna(how="all", inplace=True)
dfAttr["value"] = dfAttr["value"].astype(str)


# In[7]:


# concat train and test
dfAll = pd.concat((dfTrain, dfTest), ignore_index=True)
del dfTrain
del dfTest
gc.collect()


# In[8]:


# merge product description
dfAll = pd.merge(dfAll, dfDesc, on="product_uid", how="left")


# In[9]:


dfAll.head()


# * ### Check dataset information

# In[10]:


df_column_type = pd.DataFrame(dfAll.dtypes, columns = ['column_type'])
df_Non_Null_Count = pd.DataFrame(dfAll.notnull().sum(), columns = ['Non_Null_Count'])
df_info = pd.concat([df_column_type, df_Non_Null_Count ], axis = 1)

display(df_info)
print('-------------------------------------------------------------')
print(f'total columns: {dfAll.shape[1]}')
print('-------------------------------------------------------------')
temp = pd.DataFrame(dfAll.dtypes, columns = ['dtypes']).groupby('dtypes').size()
temp = pd.DataFrame(temp, columns = ['count'])
temp = temp.reset_index(drop = False)
temp = temp.astype({"dtypes": str})
column_type_count = [(temp['dtypes'][i],temp['count'][i]) for i in range(len(temp))]
print('column type count:')
print(column_type_count)

temp = pd.DataFrame(dfAll.memory_usage(), columns = ['memory_usage'])
temp = temp.reset_index(drop = False)
temp.columns = ['item','memory_usage']
column_memory_usage = [(temp['item'][i],temp['memory_usage'][i]) for i in range(len(temp))]
print('-------------------------------------------------------------')
print('column memory usage (bytes):')
print(column_memory_usage)


# <code style="background:yellow;color:black">***The dataset seems to be fine. No Missing value need to be solve. But I would like to create a process to solve the problem if it is exist.***</code>

# In[11]:


dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
del dfDesc
gc.collect()


# In[12]:


# merge product brand from attributes
dfBrand = dfAttr[dfAttr.name=='MFG Brand Name'][['product_uid', 'value']].rename(columns={'value': 'product_brand'})
dfAll = pd.merge(dfAll, dfBrand, on='product_uid', how='left')
# this command is not necessary, because 'product_brand' is already stored by str type. 
# but do this can remind me to consider difference condition 
dfBrand['product_brand'] = dfBrand['product_brand'].values.astype(str)  # the command is redundant in this case
dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
del dfBrand
gc.collect()


# In[13]:


dfAll.head()


# 前置單一底線 _$(object) ：用於類別內部使用，from M import *並無法直接使用此類的物件。此種用法類似於在class中定義private的method或是attribute。

# If using 'single leading underscore', it means the function only design for this case.<br>
# Engineering view: weak "internal use" indicator. E.g. from M import * does not import objects whose name starts with an underscore.

# <div class="alert alert-success" role="alert">
#   <h4 class="alert-heading">Note!</h4>
# <strong>
# 1. If using 'single leading underscore', it means the function only design for this case.<br>
# 2. Engineering view: weak "internal use" indicator. E.g. from M import * does not import objects whose name starts with an underscore.<br> 
# 3. Sometime, the uncompleted function will add a single leading underscore in the function name.</strong>

# In[14]:


# merge product color from attributes
color_columns = ['Color Family', 'Color/Finish', 'Color', 'Color/Finish Family', 'Fixture Color/Finish']
dfColor = dfAttr[dfAttr.name.isin(color_columns)][["product_uid", "value"]].rename(columns={"value": "product_color"})
dfColor.dropna(how="all", inplace=True)  # the command is redundant in this case
_agg_color = lambda df: " ".join(list(set(df["product_color"])))
dfColor = dfColor.groupby("product_uid").apply(_agg_color)
dfColor = dfColor.reset_index(name="product_color")
dfColor["product_color"] = dfColor["product_color"].values.astype(str)
dfAll = pd.merge(dfAll, dfColor, on="product_uid", how="left")
dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
del dfColor
gc.collect()


# In[15]:


dfAll.head()


# <code style="background:yellow;color:black">***The number of attributes is different from each product, so covert each attributes to the text and use '|' to separate them.***</code>

# In[16]:


# merge product attribute
_agg_attr = lambda df: config.ATTR_SEPARATOR.join(df["name"] + config.ATTR_SEPARATOR + df["value"])
dfAttr = dfAttr.groupby("product_uid").apply(_agg_attr)
dfAttr = dfAttr.reset_index(name="product_attribute_concat")
dfAll = pd.merge(dfAll, dfAttr, on="product_uid", how="left")
dfAll.fillna(config.MISSING_VALUE_STRING, inplace=True)
del dfAttr
gc.collect()


# In[17]:


dfAll.head()


# In[ ]:


# save data
if config.TASK == 'sample':
    dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy() # in this case ".copy" is redundant
pkl_utils._save(config.ALL_DATA_RAW, dfAll)

# info
dfInfo = dfAll[['id','relevance']].copy()
pkl_utils._save(config.INFO_DATA, dfInfo)


# In[ ]:


if os.path.isfile('data_preparer.ipynb'):
    get_ipython().system('jupyter nbconvert --to script data_preparer.ipynb')


# In[ ]:





# In[ ]:




