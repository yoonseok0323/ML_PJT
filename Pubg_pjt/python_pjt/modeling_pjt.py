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

# In[34]:


kill = df[['kills','teamKills','roadKills','longestKill','weaponsAcquired','killStreaks','headshotKills','DBNOs','damageDealt','winPlacePerc']]
# 킬 & 데미지"
kill.describe()


# In[35]:


kill


# In[36]:


plt.figure(figsize=(15,15))
sns.heatmap(kill.corr(), linewidths = 1.0, vmax = 1.0,
           square = True, linecolor = "white", annot = True, annot_kws = {"size" : 16})


# In[12]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(x= kill['damageDealt'], y= kill['winPlacePerc'],data=kill)


# In[13]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(x= kill['longestKill'], y= kill['winPlacePerc'],data=kill)


# In[14]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=kill)
plt.show()


# In[15]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killStreaks", y="winPlacePerc", data=kill)
plt.show()


# In[40]:


plt.figure(figsize=(30,15))
sns.boxplot(x="weaponsAcquired", y="winPlacePerc", data=kill)
plt.show()


# In[53]:


kill['weaponsAcquired'].mean()


# In[49]:


wea = kill['weaponsAcquired'].unique()
wea
#236


# In[52]:


wea1=kill['weaponsAcquired'].value_counts()
wea1.head(30)


# # Heal

# In[54]:


heal=df[['boosts','heals','revives','matchDuration','winPlacePerc']]
# 회복 아이템
heal.mean()


# In[55]:


plt.figure(figsize=(10,10))
sns.heatmap(heal.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# In[18]:


plt.figure(figsize=(15,8))
sns.boxplot(x="boosts", y="winPlacePerc", data=heal)
plt.show()
# 부스트 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[56]:


heal['boosts'].value_counts()


# In[57]:


heal['boosts'].mean()


# In[19]:


plt.figure(figsize=(15,8))
sns.boxplot(x="heals", y="winPlacePerc", data=heal)
plt.show()
#heals 힐링 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[64]:


heal['heals'].mean()


# In[65]:


heal['heals'].value_counts()


# In[66]:


hea= heal['heals'].value_counts()
hea.head(30)


# In[20]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()
# 팀플일 경우 부활 -> 3~4번을 넘어가면 장기전으로 이어지므로 의미가 없는 것으로 보인다.


# # Dist

# In[21]:


dist = df[['rideDistance','walkDistance','swimDistance','winPlacePerc']]


# In[22]:


dist


# In[68]:


plt.figure(figsize=(10,10))
sns.heatmap(dist.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# In[ ]:





# In[71]:


dist[(dist['walkDistance'] == 0.0) & (dist['winPlacePerc'] != 0.0)]
# 움직이지않고도 winplaceperc가 높은 경우.


# In[24]:


df[(df['walkDistance'] == 0.0) & (df['rideDistance'] == 0.0) & (df['swimDistance'] == 0.0)]


# In[25]:


dist.describe()
#1154m 03 1000
# 06 4백만


# In[26]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='walkDistance',y='winPlacePerc',data=dist)


# In[27]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='swimDistance',y='winPlacePerc',data=dist)


# In[33]:


dist['swimDistance'].value_counts()


# In[29]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='rideDistance',y='winPlacePerc',data=dist)


# In[ ]:




