import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

class ClusteringMethods:
    def __init__(self) -> None:
        """ClusteringMethods 클래스 초기화"""
        pass

    # DBSCAN 관련 메소드
    def check_k_distances(self, df: pd.DataFrame, k: int = 100, threshold: float = 0.05, visualize: bool = False) -> np.ndarray:
        """
        DBSCAN을 위한 최적의 epsilon 값을 k-거리 그래프를 사용하여 탐색.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터.
            k (int): 고려할 최근접 이웃의 수 (기본값: 100).
            threshold (float): 거리 변화의 변곡점을 탐지하기 위한 임계값 (기본값: 0.05).
            visualize (bool): True일 경우 k-거리 그래프를 시각화 (기본값: False).

        Returns:
            np.ndarray: 변곡점에서 탐지된 잠재적인 epsilon 값 목록.
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(df)
        distances, _ = nbrs.kneighbors(df)

        # k-최근접 이웃 거리 정렬
        k_distances = np.sort(distances[:, 1])  
        deltas = np.diff(k_distances)  # 거리의 1차 미분 계산

        # 거리 변화가 threshold를 초과하는 변곡점 식별
        change_points = np.where(np.abs(deltas) > threshold)[0]
        return_list = []

        # 변곡점 출력
        for point in change_points:
            print(f"변곡점 인덱스 {point}, 거리: {k_distances[point + 1]}")
            return_list.append(k_distances[point + 1])

        # k-거리 그래프 시각화
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(k_distances, label='k-Distance', color='blue')
            for point in change_points:
                plt.axvline(x=point, color='red', linestyle='--')
            plt.title('k-Distance Graph')
            plt.xlabel(f'{k}th Nearest Neighbor Distance')
            plt.ylabel('Distance')
            plt.grid()
            plt.legend()
            plt.show()

        return np.array(return_list)

    def check_silhouette_score(self, df: pd.DataFrame, eps: float = 0.4152, min_samples: int = 100) -> None:
        """
        DBSCAN 클러스터링 결과에 대한 실루엣 점수 계산.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터.
            eps (float): DBSCAN의 epsilon 파라미터 (기본값: 0.4152).
            min_samples (int): 코어 포인트로 간주되기 위한 이웃 내 최소 샘플 수 (기본값: 100).
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(df)

        # 클러스터가 2개 이상일 경우 실루엣 점수를 계산
        if len(set(labels)) > 1:  
            silhouette_avg = silhouette_score(df, labels)
            print(f"실루엣 점수: {silhouette_avg:.4f}")
        else:
            print("실루엣 점수를 계산할 클러스터가 충분하지 않습니다.")

    def apply_dbscan(self, df: pd.DataFrame, eps: float = 0.4152, min_samples: int = 100, num: int = 0) -> pd.DataFrame:
        """
        주어진 DataFrame에 DBSCAN 클러스터링 적용.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터.
            eps (float): DBSCAN의 epsilon 파라미터 (기본값: 0.4152).
            min_samples (int): 코어 포인트로 간주되기 위한 이웃 내 최소 샘플 수 (기본값: 100).
            num (int): 여러 클러스터링 결과를 구분하기 위한 번호 (기본값: 0).

        Returns:
            pd.DataFrame: 클러스터 레이블이 추가된 원본 DataFrame.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(df)
        df[f'cluster_{num}'] = labels
        return df


    # K-means++ 관련 메소드
    def initialize_centroids(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        데이터 X에서 랜덤하게 k개의 센트로이드를 초기화.

        Args:
            X (np.ndarray): 데이터 배열.
            k (int): 센트로이드 개수.

        Returns:
            np.ndarray: 초기화된 k개의 센트로이드.
        """
        n_samples = X.shape[0]
        seeds = np.random.choice(n_samples, size=k, replace=False)
        centroids = X[seeds]
        return centroids

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        각 데이터 포인트를 가장 가까운 센트로이드에 할당.

        Args:
            X (np.ndarray): 데이터 배열.
            centroids (np.ndarray): 센트로이드 배열.

        Returns:
            np.ndarray: 클러스터 할당 결과.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, ...], axis=-1)
        labels = np.argmin(distances, axis=-1)
        return labels

    def update_centroids(self, X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """
        현재 클러스터 할당을 기반으로 새로운 센트로이드를 업데이트.

        Args:
            X (np.ndarray): 데이터 배열.
            labels (np.ndarray): 현재 클러스터 할당.
            k (int): 센트로이드 개수.

        Returns:
            np.ndarray: 업데이트된 센트로이드.
        """
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = X[labels == i].mean(axis=0)
        return centroids

    def k_means_plus(self, X: np.ndarray, k: int, num_iter: int = 300, tol: float = 1e-6):
        """
        K-means++ 알고리즘 실행.

        Args:
            X (np.ndarray): 데이터 배열.
            k (int): 클러스터 개수.
            num_iter (int): 최대 반복 횟수 (기본값: 300).
            tol (float): 수렴을 위한 허용 오차 (기본값: 1e-6).

        Returns:
            tuple: 클러스터 레이블 및 센트로이드.
        """
        centroids = self.initialize_centroids(X, k)
        for _ in range(num_iter):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels, k)
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        return labels, centroids


    # 시각화
    def visualize_result(self, df: pd.DataFrame, col: str = 'cluster_0') -> None:
            """
            위도와 경도를 기반으로 클러스터링 결과를 2D로 시각화

            Args:
                df (pd.DataFrame): 클러스터링 결과가 포함된 DataFrame.
                col (str): 클러스터 레이블이 저장된 열의 이름 (기본값: 'cluster_0').
            """
            unique_labels = set(df[col])

            plt.figure(figsize=(10, 10))
            for label in unique_labels:
                if label == -1:
                    color = 'k'  # 노이즈 포인트는 검은색으로 표시
                else:
                    color = plt.cm.jet(label / (max(unique_labels) + 1))

                plt.scatter(df.loc[df[col] == label, 'longitude'], 
                            df.loc[df[col] == label, 'latitude'], 
                            s=50, c=color, label=f'Cluster {label}')

            plt.title('DBSCAN Clustering Results (Latitude/Longitude Based 2D)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid()
            plt.show()