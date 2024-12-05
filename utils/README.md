# Utils 폴더 설명서

이 폴더에는 프로젝트 전반에 걸쳐 사용되는 다양한 유틸리티 함수들이 포함되어 있습니다. 각 스크립트의 용도와 내부 함수는 아래와 같습니다.



## clustering.py
클러스터링 방법(DBSCAN, K-means++) 및 그 결과를 시각화하는 도구를 제공합니다.
- **함수 목록**
  - check_k_distances: DBSCAN을 위한 최적의 epsilon 값을 k-거리 그래프를 사용하여 탐색.
  - check_silhouette_score: DBSCAN 클러스터링 결과에 대한 실루엣 점수를 계산.
  - apply_dbscan: 주어진 DataFrame에 DBSCAN 클러스터링 적용.
  - initialize_centroids: 데이터에서 랜덤하게 k개의 센트로이드를 초기화.
  - assign_clusters: 각 데이터 포인트를 가장 가까운 센트로이드에 할당.
  - update_centroids: 현재 클러스터 할당을 기반으로 새로운 센트로이드를 업데이트.
  - k_means_plus: K-means++ 알고리즘 실행.
  - visualize_result: 위도와 경도를 기반으로 클러스터링 결과를 2D로 시각화.


## data_split.py
데이터셋을 무작위 또는 시간 기반으로 학습 및 검증 세트로 분할하는 기능을 제공합니다.
- **함수 목록**
  - data_split: 지정된 방식에 따라 학습 및 검증 세트로 데이터 분할.


## load_params.py
특정 모델과 유형에 맞는 하이퍼파라미터를 YAML 파일에서 불러오는 기능을 제공합니다.
- **함수 목록**
  - load_params: 모델 이름과 유형에 따라 하이퍼파라미터를 불러옴.


## mysql.py
MySQL 데이터베이스와의 연결 및 데이터 삽입을 관리하는 클래스를 제공합니다.
- **함수 목록**
  - db_connect: MySQL 데이터베이스에 연결.
  - db_disconnect: MySQL 데이터베이스 연결을 종료.
  - db_insert: 특정 테이블에 데이터를 삽입.

