# PUBG_Modeling_PJT
- **WHAT IS GOAL? -> 사용자가 몇 등을 할 것인 예측하는 것**

## PJT_Timeline
- **DAY1~2 : 6/7(화) ~ 6/8(수), EDA 작업**
- **DAY3 : 6/9(목), 전처리 작업 & 중요 feature 선택**
    - **Numeric data를 Boxplot으로 시각화하여 확인한 outlier를 이상치라고 생각하고 뜯어보고 drop해보니,   
      정상 data로 판단하게 된 시행착오를 겪음**

- **DAY4 : 6/10(금), 전처리 작업(walkDistance) & 회귀모델 사전조사**
    -   Linear Regression ( CYS'part ) 
        Lasso  
        Ridge  
        Polynomial Regression  ( CYS'part ) 
        RandomForest  
        XGBoost  
        LightGBM  
        Neural Network  
        
-**DAY5 : 6/13(월)**
    - **select model(linear,poly) & VScode_processing.py & team_github**
 
-**DAY6 :  6/14(화)**
    -**VScode_processing.py / learn.py_ing & team_github**
    
-**DAY7 :  6/15(수)**
    -**VScode_processing.py_Done / learn.py_Done & team_github & 대면 feadback( All_feature 넣고 해보기 )**
    
-**DAY8 :  6/16(목)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done(conflict 해결) & 대면 feadback( All_feature 넣고 해보기 )_Done**

-**DAY9 :  6/17(금)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done & 대면 feadback( All_feature 넣고 해보기 )_Done & ppt작업**

-**DAY10 :  6/290(월)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done & 대면 feadback( All_feature 넣고 해보기 )_Done & ppt작업_Done**



  
--- 
## DataProcessing_Columns
- **아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지**
- **이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.**
- ** 위의 2가지 내용은 해당 부분도 test_data에 실제 data일 경우의 수가 있기 때문에 이상치 처리하지 않음**

- **이상치 데이터 정리**
    -**Points(killPoints, rankPoints, winPoints) : 결측치 데이터 채우기**
    -**categorical data(matchType) : encoding 적용**


- **데이터 스케일링**
    - standardscaler , minmaxscaler

 - **selected feature**
    - **'walkDistance', 'killPlace', 'boosts', 'weaponsAcquired','damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill',
                  'headshotKills', 'rideDistance','assists','DBNOs','killPoints','matchType','rankPoints','winPoints**

