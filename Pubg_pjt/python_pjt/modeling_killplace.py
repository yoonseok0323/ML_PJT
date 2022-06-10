#!/usr/bin/env python
# coding: utf-8

# ## DataProcessing_Columns
# - 아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지
# - 이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.
# - 데이터 스케일링  
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)


# In[25]:


df = pd.read_csv("Downloads/pubg-finish-placement-prediction/train_V2.csv")


# In[26]:


df = df[['matchId','matchType','Id','groupId','assists','killPlace','kills','matchDuration', 'boosts' ,'damageDealt', 'heals',  'killStreaks', 'longestKill', 'walkDistance','winPlacePerc']]


# # killplace

# In[27]:


# df[(df['kills'] ==0) & (df['killPlace'] >40)]


# In[28]:


df_1 = df[(df['matchId']== '2001300d4f5787') & (df['kills']<3) & (df['killPlace']<50) & (df['winPlacePerc']<0.1)] 
df_1.sort_values('killPlace',ascending=False)


# In[29]:


df[(df['matchId']== '2001300d4f5787') & (df['groupId'] == 'de180c28e7c199')]


# In[30]:


df[(df['matchId']== '2001300d4f5787') & (df['groupId'] == '58bc4104935623')]


# In[31]:


df_1.groupby('groupId').mean()


# In[32]:


df[(df['matchType'] =='solo')&(df['kills'] != 0) & (df['winPlacePerc'])]


# In[34]:


df_2 = df[(df['matchId']== 'e5e181d2da0334') & (df['killPlace']<50) &(df['winPlacePerc'] < 0.3)]

df_2.sort_values('killPlace',ascending=False)


# In[35]:


df[(df['matchId']== 'e5e181d2da0334') & (df['killPlace']<10)]


# In[ ]:




