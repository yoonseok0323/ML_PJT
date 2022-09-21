# :crystal_ball: PUBG Finish Placement Prediction_Modeling_PJT

<img width="678" alt="스크린샷 2022-09-21 오전 11 07 41" src="https://user-images.githubusercontent.com/103194475/191398862-99266782-02c8-489c-9a01-6101463770ed.png">

## :heavy_check_mark: 프로젝트 기간 & 인원
2022년 6월 7일 ~ 2022년 6월 21일  
팀 프로젝트: 총 4명  
</br>
### 역할 및 사용 모델
- 박성배(팀장): Neural Network
- 최윤석 : Linear Regression & Poly Regression
- 김성수 : lightGBM
- 송희연 : Lasso & Ridge
</br>

## :heavy_check_mark: 사용 기술 스택 & 라이브러리
- Python
- Sklearn
- Juypter & Vscode
- Numpy
- Pandas
- Seaborn
- Git
</br>

## :heavy_check_mark: 주제
### PUBG Finish Placement Prediction 
사용자가 몇 등으로 게임을 종료할지(final stats)를 예측하는 모델링 프로젝트   
  
  
</br>

## :heavy_check_mark: 프로젝트 진행 과정 & 계획
:arrow_forward: EDA : 6/7 ~ 6/8   

:arrow_forward: 전처리 : 6/9 ~ 6/10   

:arrow_forward: 모델 학습 및 평가 : 6/13 ~ 6/17   

:arrow_forward: 자료 취합 및 PPT 작성 : 6/17 ~ 6/22   

</br>

## :heavy_check_mark: 프로젝트 정리 및 회고  


:arrow_forward: 다른 분석 프로젝트를 진행할 때, 확인하고 진행해야할 사항들

:one: EDA 파트  

    1-1. Kaggle과 같은 competition이 아니라면, 어떤 목적을 가지고 예측을 진행할 것인지 확실히 논리를 가지고 구조화하는 것이 필요  
    
    1-2. Feature별 EDA를 충분히 진행이 필요  
    
    1-3. Numeric & Category에 알맞는 visualizing 과정 필요  
    
    - Numeric data를 Boxplot으로 시각화해서 확인한 outlier 데이터를 이상치라고 생각 후 해당 칼럼을 drop해보니,   
      정상 데이터였던 것을 판단하게 된 시행착오를 겪음
    
:two: Preprocess & Feature Engineering 파트  

:arrow_forward: Feature Engineering 시 진행해야할 과정 및 방법  
    
    2-1. Check_corr,corr ** 2  
    2-2. Check_VIF  ( 독립변수들 간에 영향을 주는 feature를 확인 )
    2-3. Visualize_feature importance   
    2-4. Try_PCA (Dimension reduction)  
    2-5. Drop_Outlier  
    2-6. Select_feature  
    
:three: Learn_Model 파트    


    3-1. Hyperparameter tunnig  
        - Grid search  
        - Optuna  
        
    3-2. scaler  
        - Standard scaler  
        - Minmax scaler  
        - Robuster scaler  
        - etc...  
        
    3-3. Running_part  
        - Excel or sheet 활용하여 Select_feature 및 모델 학습 결과 문서화 ( 학습시간,사용 칼럼,결과,사용된 파라미터 ...etc). 
        - Check_Overfitting
        - Visualize model_result
        
 </br>
 
:heavy_check_mark: PUBG Finish Placement Prediction 프로젝트는 데이터 EDA부터 실제 머신러닝 모델 학습하고 결과 값을 도출해내는 A to Z 과정의 프로젝트였다.  
EDA,전처리,모델 학습 전 과정에 걸쳐 다양한 시행착오를 겪으며, 벽도 많이 느꼈었고 도중에 팀원 한분이 하차하는 등 여러 이슈가 발생했었다.   

그래도 기술적으로 깊은 식견을 가지신 팀장님의 리드 덕분에 프로젝트가 계획적으로 진행 될 수 있었고, 모르는 부분 또한 지속적인 피드백을 통해 해결해주시며 새로운 지식도 많이 얻어 갈 수 있었다.  
Vscode 이용하여 여러개의 .py를 생성하며, 함수와 클래스를 많이 사용했었는데 이때 큰 도움을 받을 수 있었다.  

1. 전역변수의 최소화  
2. Snake_type 변수 명명법 사용
3. 주석 사용 최소화 ( 함수,클래스 이름만 보고도 어떠한 기능을 할 수 있는지 알 수 있도록 )  
위와 같은 방법을 사용하도록 하는 것을 지향하셨고 연습할 수 있도록 피드백을 많이 해주셨다.  

또한, Git-bracnh 작업을 통한 pull & request을 활용했었는데 repos의 구성이라던지 pull & request에서 충돌이 발생하는 이슈를 함께 논의하여 해결하며   
Git에 대해서 이해를 높일 수 있었다.  

이번 프로젝트는 여러모로 한계를 많이 느끼면서도 많은 것을 얻어가는 프로젝트였던 것 같다.   

팀장님의 리드를 따라가다보니 데이터를 바라보는 시선과 코드 작성 및 결과물에 대한 정리의 차이를 체감하게 되었지만, 
그래도 다음 프로젝트때에는 EDA ~ Modeling의 과정까지 어떠한 흐름으로 진행되고 무엇을 하면 되는지 알게 되었고 배울 수 있었고  
PUBG 데이터를 뜯어보며 다양한 인사이트를 접근 해볼수 있는 프로젝트였다. : )






