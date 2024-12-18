#데이터 분할하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#y 데이터 결정 후 나누기
X = df.drop(columns=['position'])
y = df['position']

knn = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='minkowski', algorithm='ball_tree', leaf_size=20, p=1 )  
knn.fit(X_train_scaled, y_train)

#knn classifier
#파라미터	설명	입력값	기본값
#n_neighbors	이웃으로 고려할 데이터의 개수 (K 값)	int, 1 이상의 정수	5
#weights	이웃의 가중치 계산 방식	'uniform' (동일 가중치), 'distance', 사용자 정의 함수	'uniform'
#algorithm	이웃을 찾기 위한 알고리즘	'auto', 'ball_tree', 'kd_tree', 'brute'	'auto'
#leaf_size	BallTree와 KDTree의 리프 크기	int, 1 이상의 정수	30
#p	거리 계산에서 사용할 Minkowski 거리의 차수 (p값)	int, 1 (맨해튼 거리), 2 (유클리드 거리)	2
#metric	거리 계산에 사용할 메트릭	str (예: 'minkowski', 'euclidean', 'manhattan') 또는 사용자 정의 함수	'minkowski'
#metric_params	거리 메트릭에 추가로 전달할 파라미터	dict, 추가 메트릭 설정	None
#n_jobs	병렬 처리를 위한 CPU 코어 수	int, -1 (모든 코어 사용), None (1개만 사용)	None

# metric 파라미터는 다양한 거리 계산 방식을 제공합니다.

# 'minkowski': Minkowski 거리 (p=1 → 맨해튼 거리, p=2 → 유클리드 거리)
# 'euclidean': 유클리드 거리 (p=2와 동일)
# 'manhattan': 맨해튼 거리 (p=1과 동일)
# 'chebyshev': 체비쇼프 거리 (max 거리)
# 'hamming': 해밍 거리
# 'canberra': 캔버라 거리
# 사용자 정의 함수: callable 함수로 사용자가 직접 정의 가능

# 예제 코드
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    algorithm='kd_tree',
    leaf_size=30,
    p=1,
    metric='manhattan'
)
