# dataset 폴더 설명서
이 폴더에는 데이터 셋에 새로운 feature를 추가하는 다양한 함수들이 포함되어 있습니다. 각 스크립트의 용도와 내부 함수는 아래와 같습니다.

## merge_data.py
- **함수 목록**
  - _type: train/test
  - contract_date: 계약 날짜
  - contract_year: 계약 연도
  - contract_month: 계약 월
  - transaction_count: 거래량
  - nearest_park_distance: 가장 가까운 공원까지의 거리
  - park_count_500m: 500m 반경 내 공원 수
  - total_park_area_500m: 500m 반경 내 공원 총 면적
  - park_count_1000m: 1000m 반경 내 공원 수
  - total_park_area_1000m: 1000m 반경 내 공원 총 면적
  - park_count_2000m: 2000m 반경 내 공원 수
  - total_park_area_2000m: 2000m 반경 내 공원 총 면적
  - weighted_park_score: 가중 공원 점수
  - avg_distance_5_parks: 5개의 가장 가까운 공원까지의 평균 거리
  - park_distance_skewness: 공원 거리의 왜도(NaN 값 1127개)
  - park_distance_kurtosis: 공원 거리의 첨도(NaN 값 1127개)
  - nearest_large_park_distance: 가장 가까운 대형 공원까지의 거리
  - large_park_count_3km: 3km 반경 내 대형 공원 수
  - large_park_count_5km: 5km 반경 내 대형 공원 수
  - large_park_count_10km: 10km 반경 내 대형 공원 수
  - total_large_park_area_10km: 10km 반경 내 대형 공원 총 면적
  - nearest_subway_distance_km: 가장 가까운 지하철까지의 거리
  - nearest_subway_latitude: 가장 가까운 지하철의 위도
  - nearest_subway_longitude: 가장 가까운 지하철의 경도
  - school_count_within_1km: 1km 반경 내 학교 수
  - closest_elementary_distance: 가장 가까운 초등학교까지의 거리
  - closest_middle_distance: 가장 가까운 중학교까지의 거리
  - closest_high_distance: 가장 가까운 고등학교까지의 거리
  - deposit_mean: 전세가 월별 평균(이자율 반영)
  - interest_rate: 이자율
  - interest_rate_diff: 이자율 변화량