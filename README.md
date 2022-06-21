# PUBG Finish Placement Prediction_Modeling_PJT
- **WHAT IS GOAL? -> 사용자가 몇 등을 할 것인지 예측**

## PJT_Timeline
- **DAY1~2 : 6/7(화) ~ 6/8(수), EDA 작업**
- **DAY3 : 6/9(목), 전처리 작업 & 중요 feature 선택**
    - **Numeric data를 Boxplot으로 시각화하여 확인한 outlier를 이상치라고 생각하고 뜯어보고 drop해보니,   
      정상 data로 판단하게 된 시행착오를 겪음**

- **DAY4 : 6/10(금), 전처리 작업(walkDistance) & 회귀모델 사전조사**
    -   **Linear Regression ( CYS'part )   
        Lasso  
        Ridge  
        Polynomial Regression ( CYS'part )   
        RandomForest  
        XGBoost  
        LightGBM  
        Neural Network**    
        
- **DAY5 : 6/13(월)**
    - **select model(linear,poly) & VScode_processing.py & team_github**
 
- **DAY6 :  6/14(화)**
    -**VScode_processing.py / learn.py_ing & team_github**
    
- **DAY7 :  6/15(수)**
    -**VScode_processing.py_Done / learn.py_Done & team_github & 대면 feadback( All_feature 넣고 해보기 )**
    
- **DAY8 :  6/16(목)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done(conflict 해결) & 대면 feadback( All_feature 넣고 해보기 )_Done**

- **DAY9 :  6/17(금)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done & 대면 feadback( All_feature 넣고 해보기 )_Done & ppt작업**

- **DAY10 :  6/290(월)**
    -**VScode_processing.py_Done / learn.py_Done & team_github_Done & 대면 feadback( All_feature 넣고 해보기 )_Done & ppt작업_Done**



  
--- 
## Data_PreProcessing & Feature Engineering
- **아웃라이어 제거?, 컬럼별 아웃라이어 제거, 제거 시, 데이터 수? 얼마나 데이터가 제거되는지**
- **이상한 부분? Ex) 이동거리 0인데 승률이 높은 경우.**
- **위의 2가지 내용은 해당 부분도 test_data에 실제 data일 경우의 수가 있기 때문에 이상치 처리하지 않음**

- **이상치 데이터 정리**
    -**Points(killPoints, rankPoints, winPoints) : 결측치 데이터 채우기**
    -**categorical data(matchType) : encoding 적용**


- **데이터 스케일링**
    - standardscaler , minmaxscaler

 - **selected feature**
    - **'walkDistance', 'killPlace', 'boosts', 'weaponsAcquired','damageDealt', 'heals', 'kills', 'killStreaks', 'longestKill',
                  'headshotKills', 'rideDistance','assists','DBNOs','killPoints','matchType','rankPoints','winPoints**

- **VIF**
    - **다중공선성을 확인해보며 독립변수들 간에 영향을 주는 feature를 확인**


- **차원축소**
    - **유의미한 관계가 있는 features(ex. kill+assits)를 묶어서 확인**
 

---
## Learn_Model

- **Linear_regression**
- **Poly_regression**


---

## Review & Insight
- **EDA & Data_PreProcessing**
    - **Numeric_data 시각화 시 boxplot보다 histplot으로 확인이 필요하다.**
    - **상관계수, 결정계수, VIF를 통해 feature 분류**
    - **해당 과정을 통해서 이상치라고 생각했던 data가 정상 data일 수 있겠다는 insight 획득, 전처리과정의 중요성!**

- **VScode**
    - **함수명의 직관성 & 간결성 / 코드 재사용의 중요성**
    
- **Github**
    - **Used branch & push to Team_organize_github**
    - **Pull request시 발생하는 conflict issue 발생 및 해결**

- **Learning_Model**
    - **scaler,hyperparameter-tunning 보다 feature의 drop,add 부분에서 예측 값의 변동 추이가 있었다.**

---

## Remind process for next project  
- **작업환경: Jupyter -> VScode & Github

1. EDA ( Tool: Jupyter notebook)  
    1-1. Kaggle과 같은 competition이 아니라면, 어떤 목적을 가지고 예측을 진행할 것인지 확실히 논리를 가지고 구조화.  
    1-2. Feature별 EDA를 충분히 진행  
    1-3. Numeric & Category에 알맞는 visualizing 진행.  
    
2. Preprocess & Feature Engineering  
    2-1. Check_corr,corr ** 2  
    2-2. Check_VIF  
    2-3. Visualize_feature importance   
    2-4. Try_PCA (dimension reduction)  
    2-5. Drop_Outlier  
    2-6. Select_feature  
    
3. Learn_Model  
    3-1. Hyperparameter tunnig  
        - Grid search  
        - Optuna  
        
    3-2. scaler  
        - Standard scaler  
        - Minmax scaler  
        - Robuster scaler  
        - etc...  
        
    3-3. Running_part  
        - Excel or sheet 활용하여 Select_feature data정리 ( spendtime,feature,result,parameter ...etc). 
        - Check_Overfitting (how?)
        - Visualize model_result
        

---
