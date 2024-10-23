import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from prophet import Prophet
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
from utils.clustering import ClusteringMethods
from haversine import haversine, Unit
from tqdm import tqdm

# base data features
# area_m2, contract_year_month, contract_day, floor, built_year, latitude, longitude, age, deposit, contract_type

# additional features
# _type: train/test
# contract_date: 계약 날짜
# contract_year: 계약 연도
# contract_month: 계약 월

# transaction_count: 거래량

# nearest_park_distance: 가장 가까운 공원까지의 거리
# park_count_500m: 500m 반경 내 공원 수
# total_park_area_500m: 500m 반경 내 공원 총 면적
# park_count_1000m: 1000m 반경 내 공원 수
# total_park_area_1000m: 1000m 반경 내 공원 총 면적
# park_count_2000m: 2000m 반경 내 공원 수
# total_park_area_2000m: 2000m 반경 내 공원 총 면적
# weighted_park_score: 가중 공원 점수
# avg_distance_5_parks: 5개의 가장 가까운 공원까지의 평균 거리
# park_distance_skewness: 공원 거리의 왜도(NaN 값 1127개)
# park_distance_kurtosis: 공원 거리의 첨도(NaN 값 1127개)

# nearest_large_park_distance: 가장 가까운 대형 공원까지의 거리
# large_park_count_3km: 3km 반경 내 대형 공원 수
# large_park_count_5km: 5km 반경 내 대형 공원 수
# large_park_count_10km: 10km 반경 내 대형 공원 수
# total_large_park_area_10km: 10km 반경 내 대형 공원 총 면적

# nearest_subway_distance_km: 가장 가까운 지하철까지의 거리
# nearest_subway_latitude: 가장 가까운 지하철의 위도
# nearest_subway_longitude: 가장 가까운 지하철의 경도

# school_count_within_1km: 1km 반경 내 학교 수
# closest_elementary_distance: 가장 가까운 초등학교까지의 거리
# closest_middle_distance: 가장 가까운 중학교까지의 거리
# closest_high_distance: 가장 가까운 고등학교까지의 거리

# deposit_mean: 전세가 월별 평균(이자율 반영)
# interest_rate: 이자율
# interest_rate_diff: 이자율 변화량


class MergeData:
    """
    Attributes:
        train_df (DataFrame): 학습 데이터 프레임
        test_df (DataFrame): 테스트 데이터 프레임
        park_df (DataFrame): 공원 데이터 프레임
        subway_df (DataFrame): 지하철 데이터 프레임
        school_df (DataFrame): 학교 데이터 프레임
        interest_df (DataFrame): 이자율 데이터 프레임
        merged_df (DataFrame): 통합된 데이터 프레임
        lat_lon_df (DataFrame): 위도, 경도 데이터 프레임

    Methods: 
        merge_all(): train_df와 test_df를 병합하고, 공원, 지하철, 학교, 이자율 특성을 생성하여 merge_df에 저장.
        save(path: str): merge_df를 csv 파일로 저장.
    """
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, park_df: pd.DataFrame = None, subway_df: pd.DataFrame = None, school_df: pd.DataFrame = None, interest_df: pd.DataFrame = None):
        """
        Args:
            train_df (DataFrame): 아파트 학습 데이터.
            test_df (DataFrame): 아파트 테스트 데이터.
            park_df (DataFrame): 공원 데이터.
            subway_df (DataFrame): 지하철 데이터.
            school_df (DataFrame): 학교 데이터.
            interest_df (DataFrame): 이자율 데이터.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.park_df = park_df
        self.subway_df = subway_df
        self.school_df = school_df
        self.interest_df = interest_df
        if self.interest_df is not None:
            self.interest_df.rename(columns={'year_month': 'contract_year_month'}, inplace=True)
        
        # 이상치 제거
        self.remove_outliers()
        # 중복 제거
        # self.remove_duplicates()
        
        # Test deposit 채우기
        self.test_df['deposit'] = 0
        self.train_df['_type'] = 'train'
        self.test_df['_type'] = 'test'
        self.merged_df = self.train_df.copy()
        self.merged_df = pd.concat([self.merged_df, self.test_df], axis=0)

        # 위도 경도 unique df
        self.lat_lon_df = self.merged_df[['latitude', 'longitude']].drop_duplicates()

    
    # 이상치 제거
    def remove_outliers(self, remove_indexs: List = [1588478]) -> None:
        print("Before removing outliers:", self.train_df.shape)
        print("Removing outliers:", remove_indexs)
        self.train_df = self.train_df[~self.train_df.index.isin(remove_indexs)]
        print("After removing outliers:", self.train_df.shape)
    
    # Train 중복 데이터 제거 -> 사용 안함
    def remove_duplicates(self) -> None:
        # index drop
        self.train_df = self.train_df.drop('index', axis=1)
        print('Before duplicates removal:', self.train_df.shape)
        self.train_df = self.train_df.drop_duplicates()
        print('After duplicates removal:', self.train_df.shape)
    

    # Haversine 공식 함수 정의 (두 지점 간의 거리 계산)
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371  # 지구 반경 (단위: km)
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c  # 결과를 km 단위로 반환

    # deposit by area feature 생성
    def create_deposit_by_area_feature(self) -> pd.DataFrame:
        self.merged_df['deposit_by_area'] = self.merged_df['deposit'] / self.merged_df['area_m2']
        print("Deposit by area feature created.", ['deposit_by_area'])
        return self.merged_df

    # 날짜 feature 생성    
    def create_year_month_features(self) -> pd.DataFrame:
        # contract_date 를 contract_year_month feature와 contract_day feature 합쳐서 datetime으로 변환
        self.merged_df['contract_date'] = pd.to_datetime(self.merged_df['contract_year_month'].astype(str) + self.merged_df['contract_day'].astype(str), format='%Y%m%d')
        # contract_date에서 year, month, day 생성
        self.merged_df['contract_year'] = self.merged_df['contract_date'].dt.year
        self.merged_df['contract_month'] = self.merged_df['contract_date'].dt.month
        self.merged_df['contract_day'] = self.merged_df['contract_date'].dt.day
        
        print("Year-month features created.", ['contract_date', 'contract_year', 'contract_month', 'contract_day'])
        
        return self.merged_df

    # 거래량 feature 생성
    def create_transaction_features(self) -> pd.DataFrame:
        # create_year_month_features 실행 안했으면 먼저 하라고 말해주기
        if 'contract_date' not in self.merged_df.columns:
            print("Run create_year_month_features() first.")
            return self.merged_df
        
        # df를 날짜로 sorting하고
        self.merged_df = self.merged_df.sort_values(by='contract_date')
        self.merged_df['transaction_count'] = self.merged_df.groupby(['latitude', 'longitude'])['contract_date'].cumcount()

        print("Transaction features created.", ['transaction_count'])
        
        return self.merged_df
        
    
    # 공원
    def create_park_features(self) -> pd.DataFrame:
        if self.park_df is None:
            print("Park data is not provided. Park features are not created.")
            return self.merged_df
        
        lat_lon_df = self.lat_lon_df.copy()
        
        def nearest_park_distance(apartment_coords, park_tree):
            distances, _ = park_tree.query(np.deg2rad(apartment_coords), k=1)
            return distances.ravel() * 6371
        
        def count_parks_in_radius(apartment_coords, park_tree, radius):
            return park_tree.query_radius(np.deg2rad(apartment_coords), r=radius/6371, count_only=True)
        
        def weighted_park_score(apartment_coords, park_tree, park_df):
            distances, indices = park_tree.query(np.deg2rad(apartment_coords), k=10)
            distances = distances * 6371
            areas = park_df.loc[indices.ravel(), 'area'].values.reshape(distances.shape)
            return np.sum(areas / (distances + 1), axis=1)
        
        def total_park_area_in_radius(apartment_coords, park_tree, park_df, radius):
            indices = park_tree.query_radius(np.deg2rad(apartment_coords), r=radius/6371)
            return [park_df.loc[idx, 'area'].sum() for idx in indices]

        def park_distribution_stats(apartment_coords, park_tree):
            distances, _ = park_tree.query(np.deg2rad(apartment_coords), k=5)
            distances = distances * 6371
            return (
                np.mean(distances, axis=1),
                np.apply_along_axis(skew, 1, distances),
                np.apply_along_axis(kurtosis, 1, distances)
            )
        coords = self.park_df[['latitude', 'longitude']].values
        park_tree = BallTree(np.deg2rad(coords), metric='haversine')
        
        apartment_coords = lat_lon_df[['latitude', 'longitude']].values

        features = pd.DataFrame(index=lat_lon_df.index)
        
        features['nearest_park_distance'] = nearest_park_distance(apartment_coords, park_tree)
        
        for radius in [0.5, 1, 2]:
            features[f'park_count_{int(radius*1000)}m'] = count_parks_in_radius(apartment_coords, park_tree, radius)
            features[f'total_park_area_{int(radius*1000)}m'] = total_park_area_in_radius(apartment_coords, park_tree, self.park_df, radius)
        
        features['weighted_park_score'] = weighted_park_score(apartment_coords, park_tree, self.park_df)
        
        avg_dist, skewness, kurtosis_val = park_distribution_stats(apartment_coords, park_tree)
        features['avg_distance_5_parks'] = avg_dist
        features['park_distance_skewness'] = skewness
        features['park_distance_kurtosis'] = kurtosis_val
        lat_lon_df = pd.concat([lat_lon_df, features], axis=1)
        
        print("Park features created.", features.columns.tolist())
        
        self.merged_df = pd.merge(self.merged_df, lat_lon_df, on=['latitude', 'longitude'], how='left')
    
        return self.merged_df

    # 대형 공원
    def create_large_park_features(self, size_threshold=100000) -> pd.DataFrame:
        if self.park_df is None:
            print("Park data is not provided. Large park features are not created.")
            return self.merged_df
        
        lat_lon_df = self.lat_lon_df.copy()
        
        # 대형 공원만 필터링
        large_parks = self.park_df[self.park_df['area'] >= size_threshold].reset_index(drop=True)
        
        # BallTree 생성
        large_park_coords = large_parks[['latitude', 'longitude']].values
        large_park_tree = BallTree(np.deg2rad(large_park_coords), metric='haversine')
        
        # 아파트 좌표
        apartment_coords = lat_lon_df[['latitude', 'longitude']].values
        
        # 새로운 특성 생성
        features = pd.DataFrame(index=lat_lon_df.index)
        
        # 가장 가까운 대형 공원까지의 거리
        distances, _ = large_park_tree.query(np.deg2rad(apartment_coords), k=1)
        features['nearest_large_park_distance'] = distances.ravel() * 6371  # km로 변환
        
        # 3km, 5km, 10km 반경 내 대형 공원의 수
        for radius in [3, 5, 10]:
            count = large_park_tree.query_radius(np.deg2rad(apartment_coords), r=radius/6371, count_only=True)
            features[f'large_park_count_{radius}km'] = count
        
        # 10km 반경 내 대형 공원의 총 면적
        indices = large_park_tree.query_radius(np.deg2rad(apartment_coords), r=10/6371)
        total_areas = [large_parks.loc[idx, 'area'].sum() if len(idx) > 0 else 0 for idx in indices]
        features['total_large_park_area_10km'] = total_areas
        lat_lon_df = pd.concat([lat_lon_df, features], axis=1)
        print("Large park features created.", features.columns.tolist())
        
        self.merged_df = pd.merge(self.merged_df, lat_lon_df, on=['latitude', 'longitude'], how='left')
        
        return self.merged_df

    # 지하철
    def create_subway_distance_features(self) -> pd.DataFrame:
        if self.subway_df is None:
            print("Subway data is not provided. Subway features are not created.")
            return self.merged_df
        
        lat_lon_df = self.lat_lon_df.copy()
        
        # KD-Tree를 사용해 가장 가까운 지하철을 찾는 함수
        def find_nearest_subway(lat, lon, subway_tree, subway_coordinates):
            # 주어진 좌표에 대해 가장 가까운 지하철 역 인덱스 찾기
            distance, index = subway_tree.query([lat, lon], k=1)
            
            # 가장 가까운 지하철 역의 좌표
            nearest_subway = subway_coordinates[index]
            
            # Haversine 공식을 사용하여 거리 계산
            dist_km = self.haversine(lat, lon, nearest_subway[0], nearest_subway[1])
            
            return dist_km, nearest_subway[0], nearest_subway[1]

        # subway 데이터의 좌표를 KD-Tree로 변환
        subway_coordinates = self.subway_df[['latitude', 'longitude']].values
        subway_tree = cKDTree(subway_coordinates)

        lat_lon_df['nearest_subway_distance_km'], lat_lon_df['nearest_subway_latitude'], lat_lon_df['nearest_subway_longitude'] = zip(*lat_lon_df.apply(lambda x: find_nearest_subway(x['latitude'], x['longitude'], subway_tree, subway_coordinates), axis=1))

        print("Subway distance features created.", ['nearest_subway_distance_km', 'nearest_subway_latitude', 'nearest_subway_longitude'])
        
        self.merged_df = pd.merge(self.merged_df, lat_lon_df, on=['latitude', 'longitude'], how='left')
        
        return self.merged_df

    # 학교
    def create_school_distances_features(self) -> pd.DataFrame:
        if self.school_df is None:
            print("School data is not provided. School features are not created.")
            return self.merged_df
        
        lat_lon_df = self.lat_lon_df.copy()
        
        school_locations = self.school_df[['latitude', 'longitude']].values
        school_levels = self.school_df['schoolLevel'].values

        # KDTree 생성
        kdtree = cKDTree(school_locations)

        nearby_school_counts = []
        closest_schools = {'elementary': [], 'middle': [], 'high': []}

        for index, row in lat_lon_df.iterrows():
            current_location = (row['latitude'], row['longitude'])

            # KDTree로 주변 학교들의 인덱스와 거리를 구함
            distances, indices = kdtree.query(current_location, k=len(school_locations))

            # Euclidean distance를 사용하여 1km 이내의 학교만 카운팅
            geo_distances = self.haversine(row['latitude'], row['longitude'], school_locations[:, 0], school_locations[:, 1])
            
            # 1km 이내의 학교만 카운팅
            within_1km = np.where(geo_distances < 1.0)[0]
            nearby_school_count = len(within_1km)

            # 가장 가까운 초등학교, 중학교, 고등학교의 거리 계산
            for level in ['elementary', 'middle', 'high']:
                level_indices = indices[school_levels[indices] == level]
                if len(level_indices) > 0:
                    closest_school_distance = geo_distances[level_indices[0]]
                else:
                    closest_school_distance = np.nan  # 해당 학교가 없을 경우

                closest_schools[level].append(closest_school_distance)

            nearby_school_counts.append(nearby_school_count)

        # 결과를 DataFrame에 추가
        lat_lon_df['school_count_within_1km'] = nearby_school_counts
        lat_lon_df['closest_elementary_distance'] = closest_schools['elementary']
        lat_lon_df['closest_middle_distance'] = closest_schools['middle']
        lat_lon_df['closest_high_distance'] = closest_schools['high']

        print("School distance features created.", lat_lon_df.columns[-4:].tolist())

        self.merged_df = pd.merge(self.merged_df, lat_lon_df, on=['latitude', 'longitude'], how='left')
        
        return self.merged_df

    # 전세가 월별 평균
    def create_deposit_mean_interest_features(self) -> pd.DataFrame:
        # Train_df에서 contract_year_month가 같은 row들의 deposit 평균을 column으로 추가
        self.train_df['deposit_mean'] = self.train_df.groupby('contract_year_month')['deposit'].transform('mean')

        # deposit_mean을 interest_rate_df에 추가 하나씩만
        interest_rate_df = pd.merge(self.interest_df, self.train_df[['contract_year_month', 'deposit_mean']], on='contract_year_month', how='left')
        # unique한 contract_year_month만 남기기
        interest_rate_df.drop_duplicates(subset='contract_year_month', keep='first', inplace=True)
        # index 초기화
        interest_rate_df.reset_index(drop=True, inplace=True)

        # contract_year_month 202406 추가 하고 interest_rate 4칸(최대) shift
        interest_rate_df = pd.concat([interest_rate_df, pd.DataFrame({'contract_year_month': [202406], 'interest_rate': [0]})])
        # index는 contract_year_month로
        interest_rate_df.set_index('contract_year_month', inplace=True)
        # index 순서대로 정렬
        interest_rate_df.sort_index(inplace=True)
        interest_rate_df['interest_rate'] = interest_rate_df['interest_rate'].shift(3)

        # 이전 금리와 차이값 feature 추가
        interest_rate_df['interest_rate_diff'] = interest_rate_df['interest_rate'].diff()

        # 2019 04 이후 데이터만 사용
        interest_rate_df = interest_rate_df[interest_rate_df.index >= 201904]

        # index feature로 사용하기 위해 reset_index
        interest_rate_df.reset_index(inplace=True)


        # interest_rate_df에서 필요한 컬럼만 선택
        # contract_year_month datetime으로 변환
        interest_rate_df['contract_year_month'] = pd.to_datetime(interest_rate_df['contract_year_month'], format='%Y%m')

        df_prophet = interest_rate_df[['contract_year_month', 'deposit_mean', 'interest_rate']].copy()


        # Prophet이 인식할 수 있도록 컬럼명 변경
        df_prophet.rename(columns={'contract_year_month': 'ds', 'deposit_mean': 'y'}, inplace=True)
        # 모델 정의
        model = Prophet()
        model.add_regressor('interest_rate')

        # 모델 학습
        model.fit(df_prophet.dropna())

        # 미래 데이터프레임 생성
        # ds 2024-01-01 ~ 2024-05-01
        # 예측 수행
        future = model.make_future_dataframe(periods=6, freq='MS')
        future['interest_rate'] = df_prophet['interest_rate']
        forecast = model.predict(future)
        # interest_rate_df에 merge
        interest_rate_df = pd.merge(interest_rate_df, forecast[['ds', 'trend']], left_on='contract_year_month', right_on='ds', how='left')

        # trend의 2024-01-01 ~ 2024-06-01 값만 interest_rate_df의 deposit_mean에 대입
        interest_rate_df.loc[interest_rate_df['contract_year_month'] >= '2024-01-01', 'deposit_mean'] = interest_rate_df.loc[interest_rate_df['contract_year_month'] >= '2024-01-01', 'trend'].values
        interest_rate_df['contract_year_month'] = interest_rate_df['contract_year_month'].dt.strftime('%Y%m').astype(int)
        interest_rate_df = interest_rate_df[['contract_year_month', 'deposit_mean', 'interest_rate', 'interest_rate_diff']]
        print("Interest rate features created.", ['deposit_mean', 'interest_rate', 'interest_rate_diff'])
        
        self.merged_df = pd.merge(self.merged_df, interest_rate_df, on='contract_year_month', how='left')
        
        return self.merged_df
    
    # 클러스터링 변수 생성
    def create_cluster_features(self) -> pd.DataFrame:
        apartment = self.merged_df[self.merged_df['contract_year_month'] < 202401][['latitude', 'longitude', 'nearest_subway_distance_km', 'deposit_by_area', 'area_m2', 'built_year', 'school_count_within_1km']]

        apartment_avg_deposit = apartment.groupby(['latitude', 'longitude']).agg({
            'deposit_by_area': 'mean',
            'nearest_subway_distance_km': 'first',
            'area_m2' : 'mean',
            'built_year': 'first',
            'school_count_within_1km' : 'first'
        }).reset_index()

        print(f"Number of apartments: {len(apartment_avg_deposit)}")
        
        scaler = StandardScaler()
        apartment_scaled = scaler.fit_transform(apartment_avg_deposit)
        apartment_scaled_df = pd.DataFrame(apartment_scaled, columns=['latitude', 'longitude', 'deposit_by_area', 'nearest_subway_distance_km', 'area_m2', 'built_year','school_count_within_1km'])

        clustering = ClusteringMethods()

        ## DBCAN
        df_cluster = clustering.apply_dbscan(apartment_scaled_df,min_samples=150)
        df_cluster.head(3)

        # K-means ++
        X = np.array(apartment_scaled_df.drop(['latitude', 'longitude'],axis = 1))

        k = 5  # 클러스터 개수 설정
        labels, centroids = clustering.k_means_plus(X, k)
        
        apartment_cluster = apartment_avg_deposit.copy()
        apartment_cluster['Is_Outside'] = df_cluster['cluster_0']
        apartment_cluster['Is_Outside'] = apartment_cluster['Is_Outside'].replace(-1, 0)

        apartment_cluster['cluster_kmeans'] = labels
        
        self.merged_df = pd.merge(self.merged_df, apartment_cluster[['latitude', 'longitude', 'Is_Outside', 'cluster_kmeans']], on=['latitude', 'longitude'], how='left')
        
        non_null_clusters = self.merged_df[self.merged_df['cluster_kmeans'].notnull()]
        null_clusters = self.merged_df[self.merged_df['cluster_kmeans'].isnull()]

        non_null_coords = non_null_clusters[['latitude', 'longitude']].values
        null_coords = null_clusters[['latitude', 'longitude']].values

        # cKDTree를 사용하여 가장 가까운 이웃 찾기
        tree = cKDTree(non_null_coords)

        # 가장 가까운 이웃의 인덱스를 찾음
        _, idx = tree.query(null_coords)

        # 가장 가까운 비결측값 행의 cluster_kmeans 값으로 대체
        self.merged_df.loc[self.merged_df['cluster_kmeans'].isnull(), 'cluster_kmeans'] = non_null_clusters.iloc[idx]['cluster_kmeans'].values
        self.merged_df.loc[self.merged_df['Is_Outside'].isnull(), 'Is_Outside'] = non_null_clusters.iloc[idx]['Is_Outside'].values

        print("Cluster features created.", ['Is_Outside', 'cluster_kmeans'])
        return self.merged_df

    # 주기성 feature 생성
    def create_seasonal_features(self) -> pd.DataFrame:
        # 월 추출 
        self.merged_df['month'] = self.merged_df['contract_year_month'].astype(str).str[4:6].astype(int)

        # 월을 기준으로 1년 주기성 변환
        self.merged_df['month_sin'] = np.sin(2 * np.pi * self.merged_df['month'] / 12)
        self.merged_df['month_cos'] = np.cos(2 * np.pi * self.merged_df['month'] / 12)

        self.merged_df = self.merged_df.drop('month',axis = 1)
        
        return self.merged_df
    
    # 강남역까지 거리 feature 생성
    def create_gangnam_distance_features(self) -> pd.DataFrame:
        gangnam_lat = 37.498132408887
        gangnam_lon = 127.02839523744


        self.merged_df['distance_from_gangnam'] = self.merged_df.apply(
            lambda row: self.haversine(row['latitude'], row['longitude'], gangnam_lat, gangnam_lon), axis=1)

    # 인근 지하철역 개수 feature 생성
    def create_subway_count_features(self) -> pd.DataFrame:
        """
        모든 아파트에 대해 연산하면 시간이 오래 걸리므로 위도 경도에 대해서 계산을 수행한 후 Merge한다.
        아파트의 위치 정보를 추출한다. 이후 중복된 위치 정보를 제거한다.
        """

        # 고유한 위치 정보 추출
        unique_locations = self.merged_df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        
        distances = []
        
        # 아파트로부터 지하철까지의 거리 계산
        for idx, location in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Calculating distances"):
            apartment_location = (location['latitude'], location['longitude'])
            
            # 하나의 아파트로부터 모든 지하철역까지의 거리 계산
            for _, subway in self.subway_df.iterrows():
                subway_location = (subway['latitude'], subway['longitude'])
                distance = haversine(apartment_location, subway_location, unit=Unit.METERS)
                distances.append((location['latitude'], location['longitude'], distance))
        
        # 거리 데이터프레임 생성
        distance_df = pd.DataFrame(distances, columns=['latitude', 'longitude', 'distance'])
        
        # 500m, 1000m, 2000m 안의 거리 계산
        counts = distance_df.groupby(['latitude', 'longitude']).agg(
            subways_within_500m=('distance', lambda x: (x <= 500).sum()),
            subways_within_1km=('distance', lambda x: (x <= 1000).sum()),
            subways_within_2km=('distance', lambda x: (x <= 2000).sum())
        ).reset_index()
        
        self.merged_df = pd.merge(self.merged_df, counts, on=['latitude', 'longitude'], how='left')
        
        return self.merged_df

    
    # target mean encoding feature 생성
    def create_target_mean_encoding_features(self) -> pd.DataFrame:
        train_data = self.merged_df[self.merged_df['_type'] == 'train']
        test_data = self.merged_df[self.merged_df['_type'] == 'test']
        deposit_mean = train_data.groupby(['latitude', 'longitude'])['deposit_by_area'].mean().reset_index().rename(columns = {'deposit_by_area' : 'apt_deposit_mean'})
        train_data = train_data.merge(deposit_mean, how = 'left')
        test_data = test_data.merge(deposit_mean, how = 'left')
        # train에 이미 존재하는 아파트
        test_already_exist = test_data[~test_data.apt_deposit_mean.isna()]
        
        # 기존 아파트
        existing_apt_data = train_data[['latitude', 'longitude', 'deposit_by_area']]
        existing_apt_loc = existing_apt_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

        # 새로운 아파트
        new_apt_data = test_data[test_data.apt_deposit_mean.isna()]
        new_apt_loc = new_apt_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

        # 각 새 아파트에 대해 가장 가까운 k개의 기존 아파트 전세가 평균 계산
        tree = cKDTree(existing_apt_loc.values)
        _, indices = tree.query(new_apt_loc.values, k=5)

        estimated_deposits = {}
        for i, (new_lat, new_lon) in enumerate(new_apt_loc.values):
            nearest_apts_indices = indices[i]
            nearest_apts = existing_apt_loc.iloc[nearest_apts_indices]
            nearest_apts_data = nearest_apts.merge(existing_apt_data, how = 'left')
            
            avg_deposit = nearest_apts_data['deposit_by_area'].mean()
            estimated_deposits[(new_lat, new_lon)] = avg_deposit
            
        new_apt_data['apt_deposit_mean'] = new_apt_data.apply(
            lambda row: estimated_deposits.get((row['latitude'], row['longitude'])), axis=1
        )
        
    
        test_data = pd.concat([test_already_exist, new_apt_data]).sort_index()
        
        # 지난 아파트 거래가 변수
        train_data = train_data.sort_values(['contract_year_month', 'contract_day'])
        train_data['last_deposit_by_area'] = train_data.groupby(['latitude', 'longitude'])['deposit_by_area'].shift(1)

        last_apt_deposit = train_data.groupby(['latitude', 'longitude']).last().reset_index()[['latitude', 'longitude', 'deposit_by_area']].rename(columns = {'deposit_by_area' : 'last_deposit_by_area'})
        test_data = test_data.merge(last_apt_deposit, how= 'left')
        
        self.merged_df = pd.concat([train_data, test_data]).sort_index()
        
        return self.merged_df

    def merge_all(self) -> pd.DataFrame:
        # deposit by area feature 생성
        self.create_deposit_by_area_feature()
        
        # 위치 feature 생성
        self.create_park_features()
        self.create_large_park_features()
        self.create_subway_distance_features()
        self.create_school_distances_features()
        
        # interest_rate feature 생성
        self.create_deposit_mean_interest_features()
        
        # 클러스터링 feature 생성
        self.create_cluster_features()
        
        # 주기성 feature 생성
        self.create_seasonal_features()
        
        # 강남역까지 거리 feature 생성
        self.create_gangnam_distance_features()
        
        # 인근 지하철역 개수 feature 생성
        self.create_subway_count_features()
        
        # target mean encoding feature 생성
        self.create_target_mean_encoding_features()
        
        try:
            self.merged_df.drop(columns=['index'], inplace=True)
        except:
            pass
        
        print("All features merged. shape:", self.merged_df.shape)
        
        return self.merged_df
    
    def save(self, path: str):
        self.merged_df.to_csv(path, index=False)
        print(f"Data saved to {path}")