# 수도권 아파트 전세가 예측 AI 대회

## 💡Team (사진)

| 강현구 | 서동준 | 이도걸 | 이수미 | 최윤혜 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/92253cc6-6b10-4245-a2c0-d2890cdad1b5" style="width:150px; height:150px;"/> | <img src="https://github.com/user-attachments/assets/67d55bee-4817-4401-98aa-d276a00546ad" style="width:150px; height:150px;"/> | <img src="https://github.com/user-attachments/assets/896c0009-4741-42c1-a8f5-ae66ba33397b" style="width:150px; height:150px;"/> | <img src="https://github.com/user-attachments/assets/f649e5ee-c338-4883-aad0-9a77f3fe2381" style="width:150px; height:150px;"/> | <img src="https://github.com/user-attachments/assets/b3de7f45-454e-4907-a618-c653f381a4d6" style="width:150px; height:150px;"/> |
| Data EDA, <br>Hyperparameter <br>tuning, <br>KNN modeling, <br>Ensemble | Data Merging, <br>Feature Creation, <br>FT-transformer, <br>Retrieval Modeling, <br>Hyperparameter tuning, <br>Stacking | Modularization, <br>DB connection, <br>LGBM/XGB/<br>Catboost/RF<br> modeling | Time series analysis, <br>LSTM modeling, <br>Feature selection work, <br>Clustering, <br>Data Merging, <br>Ensemble | Deep learning <br>modeling, <br>MLP, <br>GNN modeling, <br>Added features <br>to the dataset |


</br>

## 💻Introduction
아파트는 한국에서 중요한 자산이며, 가계 자산의 70% 이상을 차지합니다. 특히 전세 시장은 매매 시장과 밀접하게 연관되어 부동산 정책 수립과 시장 예측의 중요한 지표가 됩니다. 이번 대회의 목표는 단순한 가격 예측을 넘어, 부동산 시장의 정보 비대칭성을 해소하는 것입니다.

대회의 성능 평가는 **Mean Absolute Error (MAE)** 지표로 진행되며, 리더보드와 최종 순위는 MAE를 기준으로 결정됩니다.

</br>

## 💾Datasets
제공된 데이터셋은 아파트 전세 실거래가 예측을 목표로 합니다. 학습 데이터는 모델 훈련에 사용되며, 테스트 데이터는 모델 성능 평가 및 실제 예측에 사용됩니다.
- **train.csv**: 대회 훈련용 데이터 (1,801,228개의 행)
- **test.csv**: 대회 추론용 데이터 (150,172개의 행)
- **sample_submission.csv**: 정답 제출용 샘플 데이터 (150,172개의 행)
- **subwayInfo.csv**: 위도와 경도로 이루어진 지하철 위치 정보 (700개의 행)
- **interestRate.csv**: 연월로 이루어진 금리 정보 (66개의 행)
- **schoolInfo.csv**: 위도와 경도로 이루어진 학교 정보 (11,992개의 행)
- **parkInfo.csv**: 위도, 경도, 면적으로 이루어진 공원 정보 (17,564개의 행)


</br>

## ⭐Project Summary
- dataloader : 학습, 검증에 사용할 데이터 셋을 불러오는 함수가 포함되어 있습니다.
- dataset : 데이터 셋에 새로운 Feature를 추가하는 다양한 함수들이 포함되어 있습니다.
- eda : 데이터 EDA ipynb 파일이 포함되어 있습니다.
- experiments : 최종 모델에 사용하지 않은 다양한 알고리즘이 포함되어 있습니다. 
- models : 여러 모델 클래스와 파라미터를 포함하고 있으며, 각각은 공통된 구조를 따릅니다.
- utils : 프로젝트 전반에 걸쳐 사용되는 다양한 유틸리티 함수들이 포함되어 있습니다.
- ensemble.ipynb : 모델을 앙상블하는 코드입니다.
- hyperparameter_tuning.ipynb : 모델의 파라미터를 튜닝하는 코드입니다.
- merge_data.ipynb : 학습에 사용할 데이터를 만드는 코드입니다.
- trainer.ipynb : 모델 학습에 사용하는 코드입니다.
- weighted_ensemble.ipynb : 가중치를 기반으로 한 모델을 앙상블하는 코드입니다.

</br>

## 📑Wrap-up Report
[RecSys_Level2_RecSys_팀 리포트(02조).pdf]()


</br>

## 📂Architecture
```
📦level2-competitiveds-recsys-02
 ┣ 📂dataloader
 ┃ ┗ 📜dataloader.py
 ┣ 📂dataset
 ┃ ┗ 📜merge_data.py
 ┣ 📂eda
 ┃ ┣ 📜EDA_1016.ipynb
 ┃ ┣ 📜kmeans_clustering.ipynb
 ┃ ┣ 📜kmeans_plus.py
 ┃ ┗ 📜time_series_analysis.ipynb
 ┣ 📂experiments
 ┃ ┣ 📜embedding_transformer_MLP.ipynb
 ┃ ┣ 📜feature_selection.ipynb
 ┃ ┣ 📜GNN_model.ipynb
 ┃ ┣ 📜LSTM.py
 ┃ ┣ 📜predict_deposit_mean.ipynb
 ┃ ┣ 📜retrieval_model.ipynb
 ┃ ┗ 📜retrieval_pred.ipynb
 ┣ 📂models
 ┃ ┣ 📂params
 ┃ ┃ ┣ 📜catboost_param.yaml
 ┃ ┃ ┣ 📜lgbm_param.yaml
 ┃ ┃ ┣ 📜lr_param.yaml
 ┃ ┃ ┣ 📜rf_param.yaml
 ┃ ┃ ┗ 📜xgb_param.yaml
 ┃ ┣ 📜catboost.py
 ┃ ┣ 📜FT-transformer.py
 ┃ ┣ 📜knn_for_ensemble.ipynb
 ┃ ┣ 📜lgbm.py
 ┃ ┣ 📜linear_regression.py
 ┃ ┣ 📜MLP.py
 ┃ ┣ 📜MLP_model.ipynb
 ┃ ┣ 📜randomforest.py
 ┃ ┣ 📜retrieval_model.py
 ┃ ┣ 📜train_model.py
 ┃ ┗ 📜xgb.py
 ┣ 📂utils
 ┃ ┣ 📜clustering.py
 ┃ ┣ 📜data_split.py
 ┃ ┣ 📜load_params.py
 ┃ ┣ 📜mysql.py
 ┃ ┣ 📜weighted_ensemble.py
 ┃ ┗ 📜__init__.py
 ┣ 📜ensemble.ipynb
 ┣ 📜hyperparameter_tuning.ipynb
 ┣ 📜merge_data.ipynb
 ┣ 📜trainer.ipynb
 ┗ 📜weighted_ensemble.ipynb
            
```

## ⚒️Development Environment
- 서버 스펙 : AI Stage GPU (Tesla V100)
- 협업 툴 : Github / Zoom / Slack / Google Drive 
- 기술 스택 : Python / Scikit-Learn / Scikit-Optimize / Pandas / Numpy / MySQL
