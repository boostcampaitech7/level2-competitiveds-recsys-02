import numpy as np

class KmeansPlus:
    def __init__(self, X: np.ndarray, k:int):
        self.X = X
        self.k = k
        
    def initialization(self):
        """
        데이터 X에서 랜덤하게 k개의 centorid를 선택하여 리턴합니다.
        
        Returns:
        numpy.ndarray: 랜덤하게 k개의 centorid를 가지고 있는 numpy.ndarray
        """

        # 전체 데이터 개수
        n_samples = self.X.shape[0]

        # k개의 indice 선택
        seeds = np.random.choice(
            n_samples,
            size=self.k,
            replace=False,
        )

        # [k, 데이터 차원] 크기의 ndarray 생성
        centroids = np.zeros((self.k, self.X.shape[-1]))
        # centroid 복사
        centroids[:] = self.X[seeds]
        self.centroids = centroids
        return centroids


    def assign_cluster(self):
        """
        데이터 X와 centorid 사이의 거리를 구하고, 가장 가까운 centroid의 index를 리턴합니다.

        Returns:
        numpy.ndarray: 가장 가까운 centroid의 index
        """

        # 거리를 계산합니다. 브로드캐스팅을 통해서 한번에 계산합니다.
        # [n_samples, 1, 데이터 차원] - [1, k, 데이터 차원] => [n_samples, k]
        distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids[np.newaxis, ...], axis=-1)

        # `argmin` 함수를 통해서 거리가 최소인 index를 찾습니다.
        labels = np.argmin(distances, axis=-1)
        self.labels = labels


    def update_centorids(self):
        """
        새로운 centroids를 계산합니다.

        Returns:
        numpy.ndarray: 랜덤하게 k개의 centorid를 가지고 있는 numpy.ndarray
        """

        # [k, 데이터 차원] 크기의 ndarray 생성
        centroids = np.zeros((self.k, self.X.shape[-1]))

        for i in range(self.k):
            # i번째 클러스터에 아무런 데이터도 할당 되지 않을 수 있기 때문에
            if np.any(self.labels == i):
                centroids[i] = self.X[self.labels == i].mean(axis=0)
                
        self.centroids = centroids
        return centroids


    def k_means(self, num_iter=300, tol=1e-6):
        # 1. centroid 초기화
        prev_centroids = self.initialization()

        # 반복을 위한 조건을 설정합니다.
        for _ in range(num_iter):
            # 2. 클러스터에 할당
            self.assign_cluster()
            # 3. 새로운 centroid 계산
            self.update_centorids()

            # stop 조건을 만족하는지 확인합니다.
            # centroid의 모든 값이 tol보다 적게 바뀌면 멈춥니다.
            if np.all(np.abs(self.centroids - prev_centroids) < tol):
                prev_centroids = self.centroids
                break

            # stop 조건 계산을 위해 저장합니다.
            prev_centroids = self.centroids

        return self.labels, prev_centroids