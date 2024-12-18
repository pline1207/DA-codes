import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

gb_classifier = GradientBoostingClassifier(
    n_estimators=400,        # 트리 개수 200일때 65.70, 300일때 66.03, 400일때 66.20
    learning_rate=0.05,      # 학습률
    max_depth=5,             # 트리의 최대 깊이
    min_samples_split=4,     # 분할 시 최소 샘플 수 #원래 4
    min_samples_leaf=2,      # 리프 노드 시 최소 샘플 수
    subsample=0.8,           # 샘플링 비율
    max_features='sqrt',     # 특성 수 log2면 65.70, sqrt여도 65.70
    loss='log_loss',         # log면 64.93, 
    random_state=42
)
gb_classifier.fit(X_train, y_train)

#하이퍼 파라미터
loss	사용할 손실 함수 (최적화 목표)	'log_loss', 'deviance', 'exponential'	'log_loss'
learning_rate	학습률. 각 트리의 기여도를 조절함	float, > 0	0.1
n_estimators	부스팅에 사용할 트리의 개수	int, > 0	100
subsample	각 트리를 학습할 때 사용할 샘플 비율	float, 0.0 < subsample <= 1.0	1.0
criterion	트리 분할 기준	'friedman_mse', 'squared_error', 'mse'	'friedman_mse'
min_samples_split	노드를 분할하기 위한 최소 샘플 수	int 또는 float	2
min_samples_leaf	리프 노드가 되기 위한 최소 샘플 수	int 또는 float	1
min_weight_fraction_leaf	가중치를 고려한 리프 노드의 최소 샘플 비율	float, 0.0 <= value <= 0.5	0.0
max_depth	개별 트리의 최대 깊이	int, > 0	3
min_impurity_decrease	노드를 분할하기 위해 필요한 최소 불순도 감소량	float, >= 0.0	0.0
init	초기 모델. 부스팅에 사용할 시작 예측값	'zero' 또는 BaseEstimator	None
random_state	난수 생성 시드 (재현성 확보)	int, RandomState, None	None
max_features	트리에서 사용할 최대 특성 수	int, float, 'auto', 'sqrt', 'log2', None	None
verbose	학습 과정 중 출력되는 로그의 상세 정도	int, 0 (출력 없음) 또는 1 이상	0
max_leaf_nodes	트리의 최대 리프 노드 수	int, None	None
warm_start	이전 학습 결과를 이어서 학습할지 여부	bool, True 또는 False	False
validation_fraction	과적합 방지를 위한 검증 세트 비율	float, 0.0 < value < 1.0	0.1
n_iter_no_change	조기 종료(Early Stopping)를 위한 기준 반복 횟수	int, None	None
tol	조기 종료를 위한 오차 허용값	float, >= 0.0	1e-4
ccp_alpha	비용 복잡도 가지치기 파라미터	float, >= 0.0	0.0
