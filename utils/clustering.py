import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

class MakeCluster:
    def __init__(self) -> None:
        """MakeCluster 클래스 초기화"""
        pass

    def find_eps(self, df: pd.DataFrame, k: int = 100, threshold: float = 0.05, visualize: bool = False) -> np.ndarray:
        """
        DBSCAN에 대한 최적의 epsilon 값을 찾기 위해 k-거리 그래프 분석.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터가 포함된 DataFrame.
            k (int): 고려할 최근접 이웃의 수 (기본값: 100).
            threshold (float): 거리 변화의 변곡점을 탐지하기 위한 임계값 (기본값: 0.05).
            visualize (bool): True일 경우 k-거리 그래프를 플롯합니다 (기본값: False).

        Returns:
            np.ndarray: 변곡점에서 탐지된 잠재적인 epsilon 값의 목록.
        """
        # k-최근접 이웃에 대한 거리 계산을 위한 NearestNeighbors 모델을 적합시킵니다.
        nbrs = NearestNeighbors(n_neighbors=k).fit(df)
        distances, _ = nbrs.kneighbors(df)
        
        # k-최근접 이웃에 대한 거리를 정렬합니다.
        k_distances = np.sort(distances[:, 1])  # k-번째 이웃의 거리
        deltas = np.diff(k_distances)  # 거리의 1차 미분
        
        # 거리 변화가 임계값을 초과하는 변곡점을 식별합니다.
        change_points = np.where(np.abs(deltas) > threshold)[0]
        return_list = []
        
        # 변곡점과 해당 거리 출력
        for point in change_points:
            print(f"변곡점 인덱스 {point}, 거리: {k_distances[point + 1]}")
            return_list.append(k_distances[point + 1])

        if visualize:  # 요청 시 k-거리 그래프를 플롯합니다.
            plt.figure(figsize=(10, 6))
            plt.plot(k_distances, label='k-Distance', color='blue')

            for point in change_points:
                plt.axvline(x=point, color='red', linestyle='--')

            plt.title('k-Distance Graph')
            plt.xlabel('{}번째 최근접 이웃에 대한 거리'.format(k))
            plt.ylabel('거리')
            plt.grid()
            plt.legend()
            plt.show()

        return return_list
  
    def check_silhouette_score(self, df: pd.DataFrame, eps: float = 0.4152, min_samples: int = 100) -> None:
        """
        클러스터링 결과에 대한 실루엣 점수 계산.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터가 포함된 DataFrame.
            eps (float): DBSCAN의 epsilon 파라미터 (기본값: 0.4152).
            min_samples (int): 코어 포인트로 간주되기 위한 이웃 내 최소 샘플 수 (기본값: 100).
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
        labels = dbscan.fit_predict(df)

        # 클러스터가 2개 이상일 경우에만 실루엣 점수를 계산합니다.
        if len(set(labels)) > 1:  # 클러스터가 1개 이상이어야 함
            silhouette_avg = silhouette_score(df, labels)
            print(f"실루엣 점수: {silhouette_avg:.4f}")
        else:
            print("실루엣 점수를 계산할 클러스터가 충분하지 않습니다.")

    def apply_dbscan(self, df: pd.DataFrame, eps: float = 0.4152, min_samples: int = 100, num: int = 0) -> pd.DataFrame:
        """
        주어진 DataFrame에 DBSCAN 클러스터링 적용.

        Args:
            df (pd.DataFrame): 클러스터링할 데이터가 포함된 DataFrame.
            eps (float): DBSCAN의 epsilon 파라미터 (기본값: 0.4152).
            min_samples (int): 코어 포인트로 간주되기 위한 이웃 내 최소 샘플 수 (기본값: 100).
            num (int): 여러 클러스터링 결과를 구분하기 위한 정수 (기본값: 0).

        Returns:
            pd.DataFrame: 클러스터 레이블이 추가된 원본 DataFrame.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(df)
        df[f'cluster_{num}'] = labels  # DataFrame에 클러스터 레이블 추가

        return df
  
    def visualize_result(self, df: pd.DataFrame, col: str = 'cluster_0') -> None:
        """
        클러스터링 결과를 2D 또는 3D 시각화.

        Args:
            df (pd.DataFrame): 클러스터링 결과가 포함된 DataFrame.
            col (str): 클러스터 레이블이 저장된 열의 이름 (기본값: 'cluster_0').
        """
        # 클러스터 레이블 열을 제외한 나머지 열 선택
        data = df.drop(columns=[col])
        unique_labels = set(df[col])

        # 2차원 데이터의 경우 2D 시각화
        if data.shape[1] == 2:
            plt.figure(figsize=(10, 10))

            for label in unique_labels:
                if label == -1:
                    color = 'k'  # 노이즈 포인트는 검은색으로 표시
                else:
                    color = plt.cm.jet(label / (max(unique_labels) + 1))  # 클러스터 레이블에 따라 색상 할당

                plt.scatter(data.loc[df[col] == label, data.columns[1]], 
                            data.loc[df[col] == label, data.columns[0]], 
                            s=50, c=color, label=f'클러스터 {label}')

            plt.title('DBSCAN 클러스터링 결과 (2D)')
            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            plt.legend()
            plt.grid()
            plt.show()

        # 3차원 데이터의 경우 3D 시각화
        elif data.shape[1] == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            for label in unique_labels:
                if label == -1:
                    color = 'k'  # 노이즈 포인트는 검은색으로 표시
                else:
                    color = plt.cm.jet(label / (max(unique_labels) + 1))  # 클러스터 레이블에 따라 색상 할당

                ax.scatter(data.loc[df[col] == label, data.columns[0]], 
                           data.loc[df[col] == label, data.columns[1]], 
                           data.loc[df[col] == label, data.columns[2]], 
                           s=50, c=color, label=f'클러스터 {label}')

            ax.set_title('DBSCAN 클러스터링 결과 (3D)')
            ax.set_xlabel(data.columns[0])
            ax.set_ylabel(data.columns[1])
            ax.set_zlabel(data.columns[2])
            plt.legend()
            plt.grid()
            plt.show()

        else:
            print("DataFrame은 2차원 또는 3차원이어야 합니다.")
