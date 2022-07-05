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
# - **DAY5 : 6/13(월), VSC 작업환경 setting & Preprocessing part functionalize & Git**
#     -  **Linear Regression**  
#     -  **Polynomial Regression**  
# 
# 
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

# In[28]:


import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)


# In[29]:


df = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/train_V2.csv")


# df = pd.read_csv("/Users/krc/Downloads/train_mini_pjt.csv",index_col=0)
#mac

# df = pd.read_csv("D:/download/pubg-finish-placement-prediction/train_V2.csv")
#window


# In[30]:


# df[df['winPlacePerc'].isna()]


# In[31]:


df.drop(index=2744604, axis=0, inplace = True)
df.drop(columns='Id',inplace = True)
df.drop(columns = 'groupId', inplace = True)
df.drop(columns= 'matchId',inplace = True)
df


# In[32]:


df.info()


# In[33]:


df.corr()


# In[34]:


df.corr()**2 


# ### 결정계수가 0.5이상 되어야 믿을만한  지표이긴하다 but..
# 
# ### 결정계수가 0.1 이상인 column
# - **assists, boosts,heals, killPlace!, kills(?), killstreak, longestkill, rideDistance, walkDistance, weaponAcquired**
# 
# ### 결정계수가  0.1보다 작지만 영향이 있을거라고 생각되는 부분
# - **DBNOS,headshot**

# ### MatchType

# In[35]:


df['matchType'].unique()


# In[36]:


df['matchType'].value_counts()


# In[37]:


df.groupby('matchType').mean().sort_values(by='winPlacePerc',ascending=False)


# In[38]:


df['matchType'].value_counts()
# one-hot incoding


# # Kill

# In[39]:


kill = df[['kills','teamKills','roadKills','longestKill','killPlace','weaponsAcquired','killStreaks','headshotKills','DBNOs','damageDealt','winPlacePerc']]
# 킬 & 데미지"
kill.describe()


# In[40]:


kill


# In[41]:


plt.figure(figsize=(15,15))
sns.heatmap(kill.corr(), linewidths = 1.0, vmax = 1.0,
           square = True, linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ## kill 결정계수

# In[42]:


kill.corr()**2


# ## DamgeDealt ( v )

# In[43]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['damageDealt'], x= kill['winPlacePerc'],data=kill)


# ## longestKill ( v )

# In[44]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['longestKill'], x= kill['winPlacePerc'],data=kill)


# ## kills ( v )

# In[45]:


plt.figure(figsize=(15,8))
sns.boxplot(x="kills", y="winPlacePerc", data=kill)
plt.show()


# In[81]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.histplot(y= kill['kills'], x= kill['winPlacePerc'],data=kill)


# In[46]:


kill['kills'].value_counts()


# ## killstreaks ( ? ) New
# 
# - 중앙값은 많은 반면 편차가 큰 편이기도 함

# In[47]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killStreaks", y="winPlacePerc", data=kill)
plt.show()


# In[48]:


kill['killStreaks'].value_counts()


# In[49]:


plt.figure(figsize=(10,10))
sns.set_palette('pastel')
sns.scatterplot(x= kill['killStreaks'], y= kill['winPlacePerc'],data=kill)


# In[50]:


sns.lineplot(x='killStreaks',y='winPlacePerc',data=kill)


# ## weaponAcquired ( v )

# In[51]:


plt.figure(figsize=(30,15))
sns.boxplot(x="weaponsAcquired", y="winPlacePerc", data=kill)
plt.show()


# In[52]:


kill['weaponsAcquired'].mean()


# In[53]:


wea = kill['weaponsAcquired'].unique()
wea
#236


# In[54]:


wea1=kill['weaponsAcquired'].value_counts()
wea1.head(30)


# ## headshotkills ( ? ) New
# 
# - 해당 feature도 우상향하는 것을 보이지만, winPlaceperc를 예측하기에는 아웃라이어 값들이 많이 존재하기에 
#   논의가 필요해보임

# In[55]:


plt.figure(figsize=(20,10))
sns.boxplot(x="headshotKills", y="winPlacePerc", data=kill)
plt.show()


# In[56]:


plt.figure(figsize=(10,10))
sns.histplot(x= kill['headshotKills'], y= kill['winPlacePerc'],data=kill)


# In[57]:


plt.figure(figsize=(10,10))
sns.lineplot(x='headshotKills',y='winPlacePerc',data=kill)


# ## DBNOs

# In[58]:


plt.figure(figsize=(15,8))
sns.boxplot(x="DBNOs", y="winPlacePerc", data=kill)
plt.show()


# In[59]:


plt.figure(figsize=(10,10))
sns.lineplot(x='DBNOs',y='winPlacePerc',data=kill)

#데이터 양의 편차가 존재하기 때문에 뒷 부분 데이터 값을 어떻게 처리하면 좋을지.?


# In[60]:


kill['DBNOs'].value_counts()


# In[61]:


plt.figure(figsize=(15,8))
sns.boxplot(x="killPlace", y="winPlacePerc", data=kill)
plt.show()


# # Heal

# In[62]:


heal=df[['boosts','heals','revives','matchDuration','winPlacePerc']]
# 회복 아이템
heal.mean()


# In[63]:


plt.figure(figsize=(10,10))
sns.heatmap(heal.corr(), linewidths = 1.0, vmax = 1.0,
           square = True,  linecolor = "white", annot = True, annot_kws = {"size" : 16})


# ## boosts ( v )

# In[64]:


plt.figure(figsize=(15,8))
sns.boxplot(x="boosts", y="winPlacePerc", data=heal)
plt.show()
# 부스트 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[65]:


heal['boosts'].value_counts()


# In[66]:


heal[heal['boosts']==24]


# In[67]:


heal['boosts'].mean()


# In[68]:


plt.figure(figsize=(15,8))
sns.jointplot(x= heal['boosts'],y=heal['winPlacePerc'],kind='scatter',data = heal)
plt.show()


# ## heals ( v )

# In[69]:


plt.figure(figsize=(15,8))
sns.boxplot(x="heals", y="winPlacePerc", data=heal)
plt.show()
#heals 힐링 아이템 사용 시 평균적으로 winplaceperc가 높아진다.


# In[70]:


heal['heals'].mean()


# In[71]:


heal['heals'].value_counts()


# In[72]:


hea= heal['heals'].value_counts()
hea.head(30)


# In[73]:


plt.figure(figsize=(15,8))
sns.jointplot(x= heal['heals'],y=heal['winPlacePerc'],kind='scatter',data = heal)
plt.show()


# ## revives 애매

# In[74]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()
# 팀플일 경우 부활 -> 3~4번을 넘어가면 장기전으로 이어지므로 의미가 없는 것으로 보인다.


# In[75]:


plt.figure(figsize=(15,8))
sns.boxplot(x='revives', y="winPlacePerc", data=heal)
plt.show()


# In[76]:


heal['revives'].value_counts()


# In[77]:


plt.figure(figsize=(15,8))
sns.jointplot(x= heal['revives'],y=heal['winPlacePerc'],kind='scatter',data = heal)
plt.show()


# # Dist

# In[78]:


df


# In[79]:


df_d = df[['matchType','assists','killPlace','kills', 'boosts' ,'damageDealt', 'heals',  'killStreaks', 'longestKill','rideDistance', 'swimDistance','walkDistance','winPlacePerc']]




# In[80]:


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

# # Points & MatchType merge
# - kill,rank,winpoints 결측치 채우기

# In[145]:


match_type = pd.read_csv("/Users/krc/Desktop/modeling_pjt/pjt_df1.csv",index_col=0)


# In[6]:


kwrPoints = ["killPoints","winPoints","rankPoints"]
df


# In[7]:


df_2 = df_1.copy()


# In[8]:


df_2 = df_1.copy()
df_2[df_2.rankPoints != -1].groupby('match_types').rankPoints.describe()


# In[9]:


df_3 = df_2.copy()
types = list(df_3.match_types.unique())
for col in kwrPoints:
    if col != "rankPoints":
        cond0 = df_2[col] == 0
        cond1 = df_2[col] != 0
    else:
        cond0 = df_2[col] == -1
        cond1 = df_2[col] != -1
    for m_type in types:
        cond2= df_3.match_types == m_type
        mean = df_3[cond1 & cond2][col].mean()
        std = df_3[cond1 & cond2][col].std()
        size = df_3[cond0 & cond2][col].count()
        if m_type != 'others' or col == "rankPoints":
            rand_points = np.random.randint(mean-std, mean+std, size=size)
        else:
            rand_points = np.array([mean]*size)
        print(col, m_type,rand_points)
        df_3[col].loc[cond0&cond2] = rand_points
df_3


# In[10]:


for col in kwrPoints:
    plt.figure(figsize=(16,9))
    sns.histplot(data=df_3,x=col,bins=300)
    plt.show()
    plt.close()


# In[ ]:





# In[51]:


prepro_df = pd.read_csv("/Users/krc/Downloads/train_mini_pjt.csv")

def Base_feature(feature):
    df_b = feature[['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired','damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill',
                  'headshotKills', 'rideDistance','assists','DBNOs','killPoints','matchType','rankPoints','winPoints']]
    return df_b

prepro_df


# In[52]:


Base_feature(prepro_df)


# In[ ]:





# ## test.csv

# In[56]:


import pandas as pd


# In[59]:


test_d = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/test_V2.csv")
test_d
Base_feature(test_d)


# # Model Machince Learning

# - solo / heal,boosts,weaponacquired의 개수가 5이상인 경우 삭제해보는 방향으로 

# In[ ]:


df_walk_out = df_d[(df_d['walkDistance'] == 0.0) & (df_d['winPlacePerc'] != 0.0)]
df_train= df_train.drop(df_walk_out)


# ## Linear Regression

# In[21]:


import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os
import matplotlib.dates as mdates

pd.set_option('display.max_columns',None)


# In[24]:



df_train = pd.read_csv("/Users/krc/Downloads/pubg-finish-placement-prediction/train_V2.csv")
df_walk_out = df_train[(df_train['walkDistance'] == 0.0) & (df_train['winPlacePerc'] != 0.0)]
df_walk_out['walkDistance'] == 0.0
#df_train= df_train.drop(df_walk_out)


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def preprocess(df):
    df = __delete_nan_data(df)
    new_col_name = "match_types"
    df[new_col_name] = __convert_match_type_column(df,"matchType")
    df = __change_nan_points(df)
    df = __one_hot_encode_data_frame(df, new_col_name)
    df = __select_features(df)
    return df

  
def __delete_nan_data(df):
    return df.dropna()

  
def __convert_match_type_column(prepro_df,encoding_feature):
    encoded = prepro_df[encoding_feature].agg(preprocessing_match_type)
    return encoded

  
def preprocessing_match_type(match_type):
    standard_matches = ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"]
    if match_type in standard_matches:
        return match_type
    else:
        return "others" 

      
def __change_nan_points(df):
    kill_rank_win_points = ["killPoints", "rankPoints", "winPoints"]
    match_types_list = list(df.match_types.unique())
    for col in kill_rank_win_points:
        if col != "rankPoints":
            cond0 = df[col] == 0
            cond1 = df[col] != 0
        else:
            cond0 = df[col] == -1
            cond1 = df[col] != -1
        for m_type in match_types_list:
            cond2 = df.match_types == m_type
            mean = df[cond1 & cond2][col].mean()
            std = df[cond1 & cond2][col].std()
            size = df[cond0 & cond2][col].count()
            if m_type != 'others' or col == "rankPoints":
                rand_points = np.random.randint(mean-std, mean+std, size=size)
            else:
                rand_points = np.array([mean]*size)
            df[col].loc[cond0 & cond2] = rand_points
    return df

  
def __one_hot_encode_data_frame(df, encoding_feature):
    df = pd.get_dummies(df, columns=[encoding_feature])
    return df


def __select_features(df):
    main_columns = ["winPlacePerc",'heals','DBNOs',"walkDistance",'rideDistance', "boosts", "weaponsAcquired"]
    kill_columns = ["kills", "damageDealt",'assists','killPlace','headshotKills','killStreaks','longestKill']
    point_columns = ['killPoints','rankPoints','winPoints']
    match_type_columns = df.columns[df.columns.str.contains("match_types")]
    deleted_columns = list(set(df.columns)-set(main_columns)-set(kill_columns)-set(point_columns)-set(match_type_columns))
        
#     all_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs',
#        'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
#        'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
#        'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
#        'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
#        'weaponsAcquired', 'winPoints', 'winPlacePerc']    
    #deleted_columns = list(set(df.columns)-set(all_columns)-set(match_type_columns))
    return df.drop(columns=deleted_columns)


def linear_reg(df):
    X_train, X_val, y_train, y_val = model_train_data(df)
    reg = LinearRegression().fit(X_train, y_train)
    model_pred_eval_test_data(reg,X_val,y_val)
#     return reg
    

def model_train_data(df):
    X = df.drop(columns='winPlacePerc')
    y = df.winPlacePerc
    return train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

def model_pred_eval_train_data(model,X,y):
    pred_train = model.predict(X)
    print("train:",mean_absolute_error(y_train, pred_train))


def model_pred_eval_test_data(model,X_val,y_val):
    pred_val = model.predict(X_val)
    print("linear_reg:",mean_absolute_error(y_val, pred_val))
################################################################

def poly_reg(df):
    poly = LinearRegression()
    X_train, X_val, y_train, y_val = model_train_data(df)

    
    poly_features = PolynomialFeatures(degree=2, include_bias=False) 
    train_x_poly = poly_features.fit_transform(X_train)
    np.c_[df.values[0], df.values[0]**2], train_x_poly[0]
    
    
    
    #fit
    poly_model= poly.fit(train_x_poly, y_train)
    val_x_poly = poly_features.fit_transform(X_val)
    
    #pred
    pred_y = poly_m.predict(val_x_poly)

    #eval
    mae_train_p = mean_absolute_error(y_val, pred_y)
    print(mae_train_p)

    
    

print(df_train)
print(df_train.columns)
df_train = preprocess(df_train)
print(df_train)
print(df_train.columns)

linear_reg(df_train)
# poly_reg(df_train)

# print(linear_reg(df_train))
# print(poly_reg(df_train))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## train_test

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0xC0FFEE)


# In[25]:


X = df
y = df['winPlacePerc']


# In[26]:


# df = pd.get_dummies(df, columns=['match_types'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0xC0FFEE)


# In[27]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

def linear_reg(df):
    X_train, X_val, y_train, y_val = model_train_data(df)
    reg = LinearRegression().fit(X_train, y_train)
    model_pred_eval_test_data(reg,X_val,y_val)
#     return reg
    

def model_train_data(df):
    X = df.drop(columns='winPlacePerc')
    y = df.winPlacePerc
    return train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

def model_pred_eval_train_data(model,X,y):
    pred_train = model.predict(X)
    print(mean_absolute_error(y_train, pred_train))


def model_pred_eval_test_data(model,X_val,y_val):
    pred_val = model.predict(X_val)
    print(mean_absolute_error(y_val, pred_val))
linear_reg(df)

# killplace -> 0.09
# killplace x -> 0.12

# 0.10
# 0.001


# In[ ]:


def poly_reg(df):
    poly = LinearRegression()
    X_train, X_val, y_train, y_val = model_train_data(df)

    
    poly_features = PolynomialFeatures(degree=2, include_bias=False) 
    train_x_poly = poly_features.fit_transform(X_train)
    np.c_[df.values[0], df.values[0]**2], train_x_poly[0]
    
    
    
    #fit
    poly_model= poly.fit(train_x_poly, y_train)
    val_x_poly = poly_features.fit_transform(X_val)
    
    #pred
    pred_y = poly_m.predict(val_x_poly)

    #eval
    mae_train_p = mean_absolute_error(y_val, pred_y)
    print(mae_train_p)
#     poly = LinearRegression().fit(train__x_poly,y_train)


# In[ ]:


poly_reg(df_train)


# In[45]:


# train_data = df[['walkDistance', 'boosts', 'weaponsAcquired','headshotKills','DBNOs','assists',
#       'damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill', 'rideDistance']]

# train_data_17 = df[['walkDistance', 'boosts', 'weaponsAcquired',
#   'damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill', 'headshotKills', 'rideDistance','assists','DBNOs','killPoints','rankPoints','winPoints']]

# X = train_data_17
# # X = train_data

# y= df_3['winPlacePerc']


# ## feature scaling

# In[105]:


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)


# ## Learing Result

# In[106]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[27]:


#train_data_15 -> killplace,matchtype except

# linear
reg = LinearRegression()

# poly
poly = LinearRegression()
poly_features = PolynomialFeatures(degree=2, include_bias=False) # 거듭제곱
train_x_poly = poly_features.fit_transform(X_train)
np.c_[X.values[0], X.values[0]**2], train_x_poly[0] 
#두 배열을 가로 방향(왼쪽에서 오른쪽으로)으로 합치기 - numpy.c_[]

# fit
reg.fit(X_train, y_train)

poly_m= poly.fit(train_x_poly, y_train)
test_x_poly = poly_features.fit_transform(X_train)

# pred
pred_train = reg.predict(X_train)
pred_val = reg.predict(X_val)
pred_y = poly_m.predict(test_x_poly)


# evaluate
mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)

mse_train_p = mean_squared_error(y_train, pred_y)

mae_train = mean_absolute_error(y_train, pred_train)
mae_val = mean_absolute_error(y_val, pred_val)

mae_train_p = mean_absolute_error(y_train, pred_y)


print("1. Linear Regression_MSE\t, train=%.4f, val=%.4f" % (mse_train, mse_val))
print("2. Linear Regression_MAE\t, train=%.4f, val=%.4f" % (mae_train, mae_val))
print("3. polynomial Regression_MSE\t, train=%.4f" % (mse_train_p))
print("4. polynomial Regression_MAE\t, train=%.4f" % (mae_train_p))


# In[108]:


# train_data_16 -> matchtype except

# reg = LinearRegression()
# reg.fit(X_train, y_train)

# pred_train = reg.predict(X_train)
# pred_val = reg.predict(X_val)

# mse_train = mean_squared_error(y_train, pred_train)
# mse_val = mean_squared_error(y_val, pred_val)

# mae_train = mean_absolute_error(y_train, pred_train)
# mae_val = mean_absolute_error(y_val, pred_val)

# print("1. Linear Regression_MSE\t, train=%.4f, val=%.4f" % (mse_train, mse_val))
# print("2. Linear Regression_MAE\t, train=%.4f, val=%.4f" % (mae_train, mae_val))


# In[109]:


# train_data


# reg = LinearRegression()
# reg.fit(X_train, y_train)

# pred_train = reg.predict(X_train)
# pred_val = reg.predict(X_val)

# mse_train = mean_squared_error(y_train, pred_train)
# mse_val = mean_squared_error(y_val, pred_val)

# mae_train = mean_absolute_error(y_train, pred_train)
# mae_val = mean_absolute_error(y_val, pred_val)

# print("1. Linear Regression_MSE\t, train=%.4f, val=%.4f" % (mse_train, mse_val))
# print("2. Linear Regression_MAE\t, train=%.4f, val=%.4f" % (mae_train, mae_val))


# In[ ]:





# In[ ]:





# In[ ]:




