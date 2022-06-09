#!/usr/bin/env python
# coding: utf-8

# # PUBG_Modeling_PJT
# - WHAT IS GOAL? -> 사용자가 몇 등을 할 것인 예측하는 것
# 
# - 6/7 ~ 6/8 EDA 작업
# 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)


# In[4]:


# df = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/train_V2.csv")
df = pd.read_csv("/Users/krc/Desktop/modeling_pjt/pjt_df1.csv",index_col=0)


# In[5]:


# df[df['winPlacePerc'].isna()]
# #2744604 탈주닌자로 예상 drop


# In[6]:


# df.drop(index=2744604, axis=0, inplace = True)
# df.drop(columns='Id',inplace = True)
# df.drop(columns = 'groupId', inplace = True)
# df.drop(columns= 'matchId',inplace = True)
# df


# In[7]:


df


# In[8]:


df.corr()**2 


# #### 결정계수가 0.5이상 되어야 믿을만한  지표이긴하다 but..
# 
# #### 결정계수가 0.1 이상인 column
# - **assists, boosts,heals, killPlace!, kills(?), killstreak, longestkill, rideDistance, walkDistance, weaponAcquired**
# 
# #### 결정계수가  0.1보다 작지만 영향이 있을거라고 생각되는 부분
# - **DBNOS,headshot**

# #### MatchType

# In[9]:


df['matchType'].unique()


# In[10]:


df.groupby('matchType').mean().sort_values(by='winPlacePerc',ascending=False)


# In[11]:


df['matchType'].value_counts()
# one-hot incoding


# # Kill

# In[92]:


kill = df[['kills','teamKills','roadKills','longestKill','killPlace','weaponsAcquired','killStreaks','headshotKills','DBNOs','damageDealt','winPlacePerc']]
# 킬 & 데미지"
kill.describe()


# In[13]:


kill


# In[14]:


plt.figure(figsize=(15,15))
sns.heatmap(kill.corr(), linewidths = 1.0, vmax = 1.0,
           square = True, linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ### kill 결정계수

# In[15]:


kill.corr()**2


# #### DamgeDealt ( v )

# In[16]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['damageDealt'], x= kill['winPlacePerc'],data=kill)


# #### longestKill ( v )

# In[17]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['longestKill'], x= kill['winPlacePerc'],data=kill)


# #### kills ( v )

# In[18]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=kill)
plt.show()


# #### killstreaks ( ? ) New
# 
# - 중앙값은 많은 반면 편차가 큰 편이기도 함

# In[19]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killStreaks", y="winPlacePerc", data=kill)
plt.show()


# In[20]:


kill['killStreaks'].value_counts()


# In[21]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.scatterplot(x= kill['killStreaks'], y= kill['winPlacePerc'],data=kill)


# In[22]:


sns.lineplot(x='killStreaks',y='winPlacePerc',data=kill)


# #### weaponAcquired ( v )

# In[23]:


plt.figure(figsize=(30,15))
sns.boxplot(x="weaponsAcquired", y="winPlacePerc", data=kill)
plt.show()


# In[24]:


kill['weaponsAcquired'].mean()


# In[25]:


wea = kill['weaponsAcquired'].unique()
wea
#236


# In[26]:


wea1=kill['weaponsAcquired'].value_counts()
wea1.head(30)


# #### headshotkills ( ? ) New
# 
# - 해당 feature도 우상향하는 것을 보이지만, winPlaceperc를 예측하기에는 아웃라이어 값들이 많이 존재하기에 
#   논의가 필요해보임

# In[27]:


plt.figure(figsize=(20,10))
sns.boxplot(x="headshotKills", y="winPlacePerc", data=kill)
plt.show()


# In[28]:


plt.figure(figsize=(10,10))
sns.histplot(x= kill['headshotKills'], y= kill['winPlacePerc'],data=kill)


# In[29]:


plt.figure(figsize=(10,10))
sns.lineplot(x='headshotKills',y='winPlacePerc',data=kill)


# #### DBNOs

# In[86]:


plt.figure(figsize=(15,8))
sns.boxplot(x="DBNOs", y="winPlacePerc", data=kill)
plt.show()


# In[89]:


plt.figure(figsize=(10,10))
sns.lineplot(x='DBNOs',y='winPlacePerc',data=kill)

#데이터 양의 편차가 존재하기 때문에 뒷 부분 데이터 값을 어떻게 처리하면 좋을지.?


# In[88]:


kill['DBNOs'].value_counts()


# In[93]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killPlace", y="winPlacePerc", data=kill)
plt.show()


# # Heal

# In[30]:


heal=df[['boosts','heals','revives','matchDuration','winPlacePerc']]
# 회복 아이템
heal.mean()


# In[31]:


plt.figure(figsize=(10,10))
sns.heatmap(heal.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# #### boosts ( v )

# In[32]:


plt.figure(figsize=(15,8))
sns.boxplot(x="boosts", y="winPlacePerc", data=heal)
plt.show()
# 부스트 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[33]:


heal['boosts'].value_counts()


# In[34]:


heal[heal['boosts']==24]


# In[35]:


heal['boosts'].mean()


# #### heals ( v )

# In[36]:


plt.figure(figsize=(15,8))
sns.boxplot(x="heals", y="winPlacePerc", data=heal)
plt.show()
#heals 힐링 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[37]:


heal['heals'].mean()


# In[38]:


heal['heals'].value_counts()


# In[39]:


hea= heal['heals'].value_counts()
hea.head(30)


# #### revives 애매

# In[40]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()
# 팀플일 경우 부활 -> 3~4번을 넘어가면 장기전으로 이어지므로 의미가 없는 것으로 보인다.


# In[41]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()


# In[42]:


heal['revives'].value_counts()


# In[43]:


plt.figure(figsize=(15,8))
sns.jointplot(x= heal['revives'],y=heal['winPlacePerc'],kind='scatter',data = heal)
plt.show()


# # Dist

# In[46]:


dist = df[['rideDistance','walkDistance','swimDistance','winPlacePerc']]


# In[47]:


dist


# In[48]:


plt.figure(figsize=(10,10))
sns.heatmap(dist.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# #### walkDistance ( v )

# #### walkDistance == 0 경우 삭제?

# In[49]:


dist['walkDistance'].value_counts()


# In[50]:


dist[dist['walkDistance'] == 0.0]
# walkDistance 9만개 


# In[51]:


dist[(dist['walkDistance'] == 0.0) & (dist['winPlacePerc'] != 0.0)]
# 움직이지않고도 winplaceperc가 높은 경우.


# In[52]:


dist[(dist['walkDistance'] == 0.0) & (dist['rideDistance'] == 0.0) & (dist['swimDistance'] == 0.0)]


# In[53]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='walkDistance',y='winPlacePerc',data=dist)


# In[54]:


dist.describe()


# #### walkDistance 구간 별 승률 분포

# In[55]:


wal_d = dist.copy()
def wal_f (x):
    if 0<= x <500:
        return("just walk")
    elif 500<= x <1000:
        return("play well")
    elif 1000<= x <2000:
        return("goinmul")
    else:
        return("Goat")

wal_d['walkDistance'] = wal_d['walkDistance'].map(wal_f)


# In[56]:



plt.figure(figsize=(15,8))
sns.boxplot(x="walkDistance", y="winPlacePerc", data=wal_d)
plt.show()


# In[57]:


plt.figure(figsize=(10,10))
sns.countplot(x='walkDistance',data=wal_d)


# In[71]:


dist.groupby(dist['winPlacePerc']).mean()


# In[59]:


pd.cut(dist['walkDistance'],5000)
#?? or quantile


# 

# #### swimDistance 
# -  우상향 밀도를 보이지만 상관관계가 낮다.

# In[60]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='swimDistance',y='winPlacePerc',data=dist)


# In[61]:


dist['swimDistance'].value_counts()


# #### rideDistance ( v )

# In[62]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='rideDistance',y='winPlacePerc',data=dist)


# # Rankpoints

# In[76]:


df


# In[82]:


df['rankPoints'].max()


# In[83]:


df['rankPoints'].value_counts()


# In[77]:


df['killPoints'].value_counts()


# In[81]:


plt.figure(figsize=(10,10))
sns.lineplot(x='rankPoints',y='winPlacePerc',data=df)


# In[84]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='rankPoints',y='winPlacePerc',data=df)


# In[ ]:





# In[94]:


df['assists'].value_counts()


# In[ ]:




