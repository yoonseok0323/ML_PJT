# PUBG_Modeling_PJT
- **WHAT IS GOAL? -> 사용자가 몇 등을 할 것인 예측하는 것**

## PJT_Timeline
- **DAY1~2 : 6/7(화) ~ 6/8(수), EDA 작업**
- **DAY3 : 6/9(목), 전처리 작업 & 중요 feature 선택**
    - **Numeric data를 Boxplot으로 시각화하여 확인한 outlier를 이상치라고 생각하고 뜯어보고 drop해보니,   
      정상 data로 판단하게 된 시행착오를 겪음**

- **DAY4 : 6/10(금), 전처리 작업(walkDistance) & 회귀모델 사전조사**
    -   Linear Regression  
        Lasso  
        Ridge  
        Polynomial Regression  
        RandomForest  
        XGBoost  
        LightGBM  
        Neural Network  
        
 
- **selected feature**
    - **walkDistance, killPlace, boosts, weaponAcquired
      damageDealt, heals, kills, killStreaks, logestKill, rideDistance**
  
--- 
## DataProcessing_Columns
- **아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지**
- **이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.**
- **데이터 스케일링**  

