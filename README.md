# 수도권 아파트 전세가 예측 AI 대회

## 💡Team (사진)

|강현구|서동준|이도걸|이수미|최윤혜|
|:---:|:---:|:---:|:---:|:---:|
|<img src="" width="150" height="150"/>|<img src="" width="150" height="150"/>|<img src="" width="150" height="150"/>|<img src="" width="150" height="150"/>|<img src="" width="150" height="150"/>|
|각자 한 것|각자 한 것|각자 한 것|각자 한 것|각자 한 것|

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


</br>

## 📑Wrap-up Report
[RecSys_Level2_RecSys_팀 리포트(02조).pdf]()


</br>

## 📂Architecture
```

            
```

## ⚒️Development Environment
- 서버 스펙 : AI Stage GPU (Tesla V100)
- 협업 툴 : Github / Zoom / Slack / Google Drive 
- 기술 스택 : Python / Scikit-Learn / Scikit-Optimize / Pandas / Numpy / MySQL
