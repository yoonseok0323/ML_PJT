#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)


# In[2]:


# df = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/train_V2.csv")
df = pd.read_csv("/Users/krc/Desktop/modeling_pjt/pjt_df1.csv",index_col=0)


# In[3]:


# df[df['winPlacePerc'].isna()]
# #2744604 탈주닌자로 예상 drop


# In[4]:


# df.drop(index=2744604, axis=0, inplace = True)
# df.drop(columns='Id',inplace = True)
# df.drop(columns = 'groupId', inplace = True)
# df.drop(columns= 'matchId',inplace = True)
# df


# In[5]:


df


# In[6]:


df.info()


# In[7]:


df['matchType'].unique()


# In[8]:


df.groupby('matchType').mean().sort_values(by='winPlacePerc',ascending=False)


# # Kill

# In[9]:


kill = df[['kills','teamKills','killStreaks','headshotKills','DBNOs','damageDealt','winPlacePerc']]
# 킬 & 데미지"
kill.describe()


# In[10]:


kill.head()


# In[12]:


plt.figure(figsize=(15,15))
sns.heatmap(kill.corr(), linewidths = 1.0, vmax = 1.0,
           square = True, linecolor = "white", annot = True, annot_kws = {"size" : 16})


# In[13]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(x= kill['damageDealt'], y= kill['winPlacePerc'],data=kill)


# In[ ]:





# In[14]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=kill)
plt.show()


# In[15]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killStreaks", y="winPlacePerc", data=kill)
plt.show()


# # Heal

# In[16]:


heal=df[['boosts','heals','revives','winPlacePerc','matchDuration']]
# 회복 아이템
heal.mean()


# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(heal.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# In[19]:


plt.figure(figsize=(15,8))
sns.boxplot(x="boosts", y="winPlacePerc", data=heal)
plt.show()
# 부스트 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[20]:


plt.figure(figsize=(15,8))
sns.boxplot(x="heals", y="winPlacePerc", data=heal)
plt.show()
#heals 힐링 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[21]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()
# 팀플일 경우 부활 -> 3~4번을 넘어가면 장기전으로 이어지므로 의미가 없는 것으로 보인다.


# # Dist

# In[22]:


dist = df[['rideDistance','walkDistance','swimDistance','winPlacePerc']]


# In[23]:


dist


# In[24]:


df[(df['walkDistance'] == 0.0) & (df['winPlacePerc'] != 0.0)]
# 움직이지않고도 winplaceperc가 높은 경우.


# In[25]:


df[(df['walkDistance'] == 0.0) & (df['rideDistance'] == 0.0) & (df['swimDistance'] == 0.0)]


# In[26]:


dist.describe()
#1154m 03 1000
# 06 4백만


# In[27]:



f, axes = plt.subplots(2,2,figsize=(20, 15), sharex = True)
plt.xlim(0, 5000)
sns.distplot(df['rideDistance'],color='skyblue',ax=axes[0,0])
sns.distplot(df['walkDistance'],color='olive',ax=axes[0,1])
sns.distplot(df['swimDistance'],color='red',ax=axes[1,0])


# In[ ]:




