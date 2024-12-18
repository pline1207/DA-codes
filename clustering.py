from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
n_clusters = 8  # 원하는 클러스터 개수를 설정하세요.
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
#kmeans = KMeans(n_clusters=n_clusters, random_state=42, init = 'k-means++', n_init=10, algorithm='elkan')
clusters = kmeans.fit_predict(scaled_data)
data['Cluster'] = clusters
y = data['Cluster']

#KMEANS 하이퍼 파라미터
n_clusters	생성할 클러스터의 개수	int, > 0	8
init	초기 클러스터 중심값 설정 방법	'k-means++', 'random', ndarray	'k-means++'
n_init	초기 중심값 설정 후 반복 횟수	int, > 0	10
max_iter	단일 실행에 대한 최대 반복 횟수	int, > 0	300
tol	수렴을 판단하는 허용 오차값	float, >= 0.0	1e-4
precompute_distances	거리 계산 방식을 사전 계산할지 여부	deprecated, 사라짐	auto
random_state	난수 시드 설정 (재현성 확보)	int, RandomState, None	None
algorithm	알고리즘 방식 선택	'lloyd', 'elkan', 'auto'	'lloyd'
verbose	실행 시 출력되는 로그의 상세 정도	int, 0 (출력 없음) 또는 1 이상	0




#DBSCAN
#하이퍼 파라미터
eps	데이터 포인트 간의 최대 거리(반경)	float, > 0	0.5
min_samples	하나의 클러스터로 인식하기 위한 최소 샘플 개수	int, > 0	5
metric	거리 측정 방식	'euclidean', callable	'euclidean'
algorithm	근접 이웃 탐색 알고리즘	'auto', 'ball_tree', 'kd_tree', 'brute'	'auto'
leaf_size	트리 기반 알고리즘에서의 리프 노드 크기	int, > 0	30
p	Minkowski 거리의 파라미터 (p=2이면 유클리드 거리)	int, float	2
n_jobs	병렬 실행 시 사용될 CPU 코어 수 (병렬 처리)	int, None	None
#예시 코드
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 예제 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# DBSCAN 모델 생성 (하이퍼파라미터 포함)
model = DBSCAN(
    eps=0.5,              # 반경(클러스터 밀도를 결정하는 거리)
    min_samples=5,        # 반경 내 최소 샘플 수
    metric='euclidean',   # 거리 계산 방법 (유클리드 거리)
    algorithm='auto',     # 근접 이웃 탐색 알고리즘 ('auto', 'ball_tree', 'kd_tree', 'brute')
    leaf_size=30,         # 트리 기반 알고리즘에서 사용하는 리프 노드 크기
    p=None,               # 거리 계산 시 Minkowski 거리의 차수 (기본값: None)
    n_jobs=-1             # 병렬 처리에 사용할 CPU 코어 수 (-1은 모든 코어 사용)
)

# 모델 학습 및 예측
labels = model.fit_predict(X)




#계층적 군집화
#하이퍼 파라미터
n_clusters	생성할 클러스터의 개수	int, None	2
affinity	거리 계산 방식	'euclidean', 'l1', 'l2', 'manhattan', 'cosine', precomputed	'euclidean'
linkage	클러스터 간 병합 기준	'ward', 'complete', 'average', 'single'	'ward'
distance_threshold	병합을 중단할 거리 임계값 (n_clusters와 함께 사용 불가)	float, None	None
compute_full_tree	전체 트리 계산 여부	'auto', True, False	'auto'
connectivity	클러스터 간 연결성 제약	array-like 또는 None	None
#예시 코드
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 예제 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# AgglomerativeClustering 모델 생성 (하이퍼파라미터 포함)
model = AgglomerativeClustering(
    n_clusters=3,            # 클러스터 개수
    affinity='euclidean',    # 거리 계산 방법 ('euclidean', 'manhattan', 'cosine' 등)
    linkage='ward',          # 병합 기준 ('ward', 'complete', 'average', 'single')
    connectivity=None,       # 데이터 간의 연결을 정의하는 행렬 (기본값: None)
    compute_distances=True   # 거리 행렬을 계산할지 여부 (클러스터링 후 거리 계산)
)

# 모델 학습 및 예측
labels = model.fit_predict(X)
