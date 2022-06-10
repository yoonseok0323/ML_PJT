#!/usr/bin/env python
# coding: utf-8

# # PUBG_Modeling_PJT
# - **WHAT IS GOAL? -> 사용자가 몇 등을 할 것인 예측하는 것**
# 
# ## PJT_Timeline
# - **DAY1~2 : 6/7(화) ~ 6/8(수), EDA 작업**
# - **DAY3 : 6/9(목), 전처리 작업 & 중요 feature 선택**
#     - **Numeric data를 Boxplot으로 시각화하여 확인한 outlier를 이상치라고 생각하고 뜯어보고 drop해보니,   
#       정상 data로 판단하게 된 시행착오를 겪음**  
#       
# 
# - **DAY4 : 6/10(금), 전처리 작업(walkDistance) & 회귀모델 사전조사**
#     -   **Linear Regression  
#         Lasso  
#         Ridge  
#         Polynomial Regression  
#         RandomForest  
#         XGBoost  
#         LightGBM  
#         Neural Network**  
#         
#  
# - **selected feature**
#     - **walkDistance, killPlace, boosts, weaponAcquired
#       damageDealt, heals, kills, killStreaks, logestKill, rideDistance**
#   
# --- 
# ## DataProcessing_Columns
# - **아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지**
# - **이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.**
# - **데이터 스케일링**  
# 

# In[49]:


import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)


# In[58]:


#df = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/train_V2.csv")


df = pd.read_csv("/Users/krc/Desktop/modeling_pjt/pjt_df1.csv",index_col=0)
#mac

# df = pd.read_csv("D:/download/pubg-finish-placement-prediction/train_V2.csv")
#window


# In[37]:


# df[df['winPlacePerc'].isna()]
# #2744604 탈주닌자로 예상 drop


# In[5]:


# # df.drop(index=2744604, axis=0, inplace = True)
# df.drop(columns='Id',inplace = True)
# df.drop(columns = 'groupId', inplace = True)
# df.drop(columns= 'matchId',inplace = True)
# df


# In[51]:


df


# In[7]:


df.corr()


# In[8]:


df.corr()**2 


# ### 결정계수가 0.5이상 되어야 믿을만한  지표이긴하다 but..
# 
# ### 결정계수가 0.1 이상인 column
# - **assists, boosts,heals, killPlace!, kills(?), killstreak, longestkill, rideDistance, walkDistance, weaponAcquired**
# 
# ### 결정계수가  0.1보다 작지만 영향이 있을거라고 생각되는 부분
# - **DBNOS,headshot**

# ### MatchType

# In[7]:


df['matchType'].unique()


# In[34]:


df['matchType'].value_counts()


# In[8]:


df.groupby('matchType').mean().sort_values(by='winPlacePerc',ascending=False)


# In[9]:


df['matchType'].value_counts()
# one-hot incoding


# # Kill

# In[10]:


kill = df[['kills','teamKills','roadKills','longestKill','killPlace','weaponsAcquired','killStreaks','headshotKills','DBNOs','damageDealt','winPlacePerc']]
# 킬 & 데미지"
kill.describe()


# In[11]:


kill


# In[12]:


plt.figure(figsize=(15,15))
sns.heatmap(kill.corr(), linewidths = 1.0, vmax = 1.0,
           square = True, linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ## kill 결정계수

# In[13]:


kill.corr()**2


# ## DamgeDealt ( v )

# In[14]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['damageDealt'], x= kill['winPlacePerc'],data=kill)


# ## longestKill ( v )

# In[15]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['longestKill'], x= kill['winPlacePerc'],data=kill)


# ## kills ( v )

# In[16]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=kill)
plt.show()


# In[17]:


kill['kills'].value_counts()


# ## killstreaks ( ? ) New
# 
# - 중앙값은 많은 반면 편차가 큰 편이기도 함

# In[18]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killStreaks", y="winPlacePerc", data=kill)
plt.show()


# In[19]:


kill['killStreaks'].value_counts()


# In[20]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.scatterplot(x= kill['killStreaks'], y= kill['winPlacePerc'],data=kill)


# In[21]:


sns.lineplot(x='killStreaks',y='winPlacePerc',data=kill)


# ## weaponAcquired ( v )

# In[22]:


plt.figure(figsize=(30,15))
sns.boxplot(x="weaponsAcquired", y="winPlacePerc", data=kill)
plt.show()


# In[23]:


kill['weaponsAcquired'].mean()


# In[24]:


wea = kill['weaponsAcquired'].unique()
wea
#236


# In[25]:


wea1=kill['weaponsAcquired'].value_counts()
wea1.head(30)


# ## headshotkills ( ? ) New
# 
# - 해당 feature도 우상향하는 것을 보이지만, winPlaceperc를 예측하기에는 아웃라이어 값들이 많이 존재하기에 
#   논의가 필요해보임

# In[26]:


plt.figure(figsize=(20,10))
sns.boxplot(x="headshotKills", y="winPlacePerc", data=kill)
plt.show()


# In[27]:


plt.figure(figsize=(10,10))
sns.histplot(x= kill['headshotKills'], y= kill['winPlacePerc'],data=kill)


# In[28]:


plt.figure(figsize=(10,10))
sns.lineplot(x='headshotKills',y='winPlacePerc',data=kill)


# ## DBNOs

# In[29]:


plt.figure(figsize=(15,8))
sns.boxplot(x="DBNOs", y="winPlacePerc", data=kill)
plt.show()


# In[30]:


plt.figure(figsize=(10,10))
sns.lineplot(x='DBNOs',y='winPlacePerc',data=kill)

#데이터 양의 편차가 존재하기 때문에 뒷 부분 데이터 값을 어떻게 처리하면 좋을지.?


# In[31]:


kill['DBNOs'].value_counts()


# In[32]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killPlace", y="winPlacePerc", data=kill)
plt.show()


# # Heal

# In[33]:


heal=df[['boosts','heals','revives','matchDuration','winPlacePerc']]
# 회복 아이템
heal.mean()


# In[34]:


plt.figure(figsize=(10,10))
sns.heatmap(heal.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ## boosts ( v )

# In[35]:


plt.figure(figsize=(15,8))
sns.boxplot(x="boosts", y="winPlacePerc", data=heal)
plt.show()
# 부스트 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[36]:


heal['boosts'].value_counts()


# In[37]:


heal[heal['boosts']==24]


# In[38]:


heal['boosts'].mean()


# ## heals ( v )

# In[39]:


plt.figure(figsize=(15,8))
sns.boxplot(x="heals", y="winPlacePerc", data=heal)
plt.show()
#heals 힐링 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[40]:


heal['heals'].mean()


# In[41]:


heal['heals'].value_counts()


# In[42]:


hea= heal['heals'].value_counts()
hea.head(30)


# ## revives 애매

# In[43]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()
# 팀플일 경우 부활 -> 3~4번을 넘어가면 장기전으로 이어지므로 의미가 없는 것으로 보인다.


# In[44]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()


# In[45]:


heal['revives'].value_counts()


# In[46]:


plt.figure(figsize=(15,8))
sns.jointplot(x= heal['revives'],y=heal['winPlacePerc'],kind='scatter',data = heal)
plt.show()


# # Dist

# In[54]:


df


# In[69]:


df_d = df[['matchType','assists','killPlace','kills', 'boosts' ,'damageDealt', 'heals',  'killStreaks', 'longestKill','rideDistance', 'swimDistance','walkDistance','winPlacePerc']]




# In[70]:


df_d


# In[71]:


plt.figure(figsize=(10,10))
sns.heatmap(dist.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ## walkDistance ( v )

# ## walkDistance == 0 경우 삭제?

# In[72]:


df_d['walkDistance'].value_counts()


# In[73]:


df_d[df_d['walkDistance'] == 0.0]
# walkDistance 9만개 


# In[74]:


df_d[(df_d['walkDistance'] == 0.0) & (df_d['winPlacePerc'] != 0.0)]
# 움직이지않고도 winplaceperc가 높은 경우.


# In[96]:


all_zero = df_d[(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc']== 0.0)]
all_zero
# 거리와 승률이 0인 경우
# 그냥 아무것도 안 한 경우.
all_zero.describe()


# In[103]:


all_zero[all_zero['kills']==27]
#?? 진짜 핵 같은데...


# In[104]:


all_zero[all_zero['heals']==31]
# ???


# In[88]:


plt.figure(figsize=(10,10))
sns.histplot(data= all_zero,bins=30)


# In[95]:


df_d[(df_d['damageDealt'] == 0.0)&(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc'] != 0.0)]
# 거리와 딜량이 0이지만, 승률이 0이 아닌 경우
# 움직이지도 않고 딜도 넣지 않았는데 승률이 존재 한다.?
# 그럼... 아무것도 안 했는데 다른 사람들이 먼저 죽었기 때문에, 잠수였는데 운이 좋게 오래 살아남아서 winPlacePerc가 올라 간 것으로 보임.


# In[98]:


lucky = df_d[(df_d['damageDealt'] == 0.0)&(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc'] != 0.0)]
# 거리와 딜량이 0이지만, 승률이 0이 아닌 경우
# 움직이지도 않고 딜도 넣지 않았는데 승률이 존재 한다.?
# 그럼... 아무것도 안 했는데 다른 사람들이 먼저 죽었기 때문에, 잠수였는데 운이 좋게 오래 살아남아서 winPlacePerc가 올라 간 것으로 보임.


# In[99]:


lucky.describe()


# In[54]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='walkDistance',y='winPlacePerc',data=dist)


# In[55]:


dist.describe()


# ## walkDistance 구간 별 승률 분포

# In[56]:


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


# In[57]:



plt.figure(figsize=(15,8))
sns.boxplot(x="walkDistance", y="winPlacePerc", data=wal_d)
plt.show()


# In[58]:


plt.figure(figsize=(10,10))
sns.countplot(x='walkDistance',data=wal_d)


# In[59]:


dist.groupby(dist['winPlacePerc']).mean()


# In[60]:


pd.cut(dist['walkDistance'],5000)
#?? or quantile


# 

# ## swimDistance 
# -  우상향 밀도를 보이지만 상관관계가 낮다.

# In[61]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='swimDistance',y='winPlacePerc',data=dist)


# In[62]:


dist['swimDistance'].value_counts()


# ## rideDistance ( v )

# In[63]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='rideDistance',y='winPlacePerc',data=dist)


# # Rankpoints

# In[64]:


df


# In[65]:


df['rankPoints'].max()


# In[66]:


df['rankPoints'].value_counts()


# In[67]:


df['killPoints'].value_counts()


# In[68]:


plt.figure(figsize=(10,10))
sns.lineplot(x='rankPoints',y='winPlacePerc',data=df)


# In[69]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='rankPoints',y='winPlacePerc',data=df)


# In[ ]:





# # DataProcessing 
# - 아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지  
# - walkDistance, killPace, boosts, weaponAcquired
#   damageDealt, heals, kills, killStreaks, logestKill, rideDistance
# - 이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.
# - 데이터 스케일링 

# ## kills

# In[44]:


df = df[['walkDistance', 'killPlace', 'boosts','matchType' ,'weaponsAcquired' ,'damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill', 'rideDistance','winPlacePerc']]


# In[45]:


df


# In[11]:


df['kills'].value_counts()


# In[12]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=df)
plt.show()


# In[13]:


df[(df['kills']==4) & (df['winPlacePerc']< 0.21)]


# In[14]:


a = df['kills']==4
b = df['winPlacePerc']>0.2
c = df['winPlacePerc'] <=0.6
ea = a&b&c
df[ea]


# ### killls outlier
# - 3~4kill winplacepercrk 0.5미만
# - 5 ~ 16kill까지 winplaceperc가 0.7미만
# - 17~25 kill까지 winplaceperc가 0.65미만
# 
# 

# In[18]:


k1 = (df.kills >= 3) & (df.kills <5)
k2 = df['winPlacePerc'] < 0.5

df_k = df[k1 & k2]
df_k


# In[20]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=df_k)
plt.show()


# In[21]:


# 5 ~ 16kill까지 winplaceperc가 0.7미만
k3 = (df.kills > 4) & (df.kills <= 16)
k4 = df['winPlacePerc'] < 0.7

df_k2 = df[k3 & k4]
df_k2


# In[22]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=df_k2)
plt.show()


# In[23]:


# 17 ~ 25kill까지 winplaceperc가 0.7미만
k5 = (df.kills >= 17) & (df.kills <= 25)
k6 = df['winPlacePerc'] < 0.65

df_k3 = df[k5 & k6]
df_k3


# In[24]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=df_k3)
plt.show()


# In[25]:


kills = df.copy()
kills


# In[26]:


df_kills = pd.concat([df,df_k,df_k2,df_k3]).drop_duplicates(keep=False)


# In[27]:


df_kills


# In[28]:


k7 = (df.kills >= 7) & (df.kills <= 18)
k8 = df['winPlacePerc'] < 0.85

df_k4 = df[k7 & k8]
df_k4


# In[29]:


df_kills = pd.concat([df,df_k,df_k2,df_k3,df_k4]).drop_duplicates(keep=False)


# In[30]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=df_kills)
plt.show()


# In[31]:


plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
sns.boxplot(x="kills", y="winPlacePerc", data=df)

plt.subplot(2,1,2)
sns.boxplot(x="kills", y="winPlacePerc", data=df_kills)
# 7 ~ 18 0.85
plt.show()


# In[32]:


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.scatterplot(x='kills',y='winPlacePerc',data=df_kills)

plt.subplot(2,1,2)
sns.scatterplot(x='kills',y='winPlacePerc',data=df)


# ### 시행착오, 이후 작업진행
# 
# - **numeric data를 boxplot으로 시각화하여 outlier를 구하는 것은 위험한 판단**
# - **또한, 이상치라고 생각한 데이터들을 뜯어보면 test data에서도 발생할 수 있는 정상적인 data일수 있기 때문에  
#   저는 현 data를 상관관계가 높은 feature들을 뽑아서 모델링을 해볼 계획입니다.** 

# In[128]:


df.describe()


# In[129]:


df_kills.describe()


# ## walkDistance 

# ## walkDistance == 0 경우 삭제?

# In[72]:


df_d['walkDistance'].value_counts()


# In[73]:


df_d[df_d['walkDistance'] == 0.0]
# walkDistance 9만개 


# In[74]:


df_d[(df_d['walkDistance'] == 0.0) & (df_d['winPlacePerc'] != 0.0)]
# 움직이지않고도 winplaceperc가 높은 경우.


# In[96]:


all_zero = df_d[(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc']== 0.0)]
all_zero
# 거리와 승률이 0인 경우
# 그냥 아무것도 안 한 경우.
all_zero.describe()


# In[103]:


all_zero[all_zero['kills']==27]
#?? 진짜 핵 같은데...


# In[104]:


all_zero[all_zero['heals']==31]
# ???


# In[88]:


plt.figure(figsize=(10,10))
sns.histplot(data= all_zero,bins=30)


# In[105]:


df_d[(df_d['damageDealt'] == 0.0)&(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc'] != 0.0)]
# 거리와 딜량이 0이지만, 승률이 0이 아닌 경우
# 움직이지도 않고 딜도 넣지 않았는데 승률이 존재 한다.?
# 그럼... 아무것도 안 했는데 다른 사람들이 먼저 죽었기 때문에, 잠수였는데 운이 좋게 오래 살아남아서 winPlacePerc가 올라 간 것으로 보임.
# 이정도 값은 날려도 됀다고 생각이 됌.


# In[107]:


lucky = df_d[(df_d['damageDealt'] == 0.0)&(df_d['walkDistance'] == 0.0) & (df_d['rideDistance'] == 0.0) & (df_d['swimDistance'] == 0.0)&(df_d['winPlacePerc'] != 0.0)]

lucky.describe()


# In[108]:


lucky[lucky['killPlace']== 1]


# In[106]:


plt.figure(figsize=(10,10))
sns.histplot(data= lucky,bins=30)


# - solo / heal,boosts,weaponacquired의 개수가 5이상인 경우 삭제해보는 방향으로 

# # Model Machince Learning

# In[ ]:




