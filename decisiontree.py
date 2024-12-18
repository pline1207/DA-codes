from sklearn.model_selection import train_test_split

clf = DecisionTreeClassifier(random_state=42)

#classifier 하이퍼 파라미터
criterion	분할 기준 (불순도 측정 방법)	'gini', 'entropy', 'log_loss'	'gini'
splitter	노드를 분할할 때 사용할 전략	'best', 'random'	'best'
max_depth	트리의 최대 깊이. 깊이를 제한하면 과적합 방지에 도움	int, None	None
min_samples_split	노드를 분할하기 위한 최소 샘플 수	int, float	2
min_samples_leaf	리프 노드를 만들기 위한 최소 샘플 수	int, float	1
min_weight_fraction_leaf	가중치를 고려한 리프 노드의 최소 샘플 비율	float, 0.0 <= value <= 0.5	0.0
max_features	노드를 분할할 때 고려할 최대 특성 수	int, float, 'auto', 'sqrt', 'log2', None	None
random_state	난수 생성기 시드 (재현성 확보)	int, RandomState, None	None
max_leaf_nodes	트리의 최대 리프 노드 수	int, None	None
min_impurity_decrease	노드를 분할하기 위해 필요한 최소 불순도 감소량	float, >= 0.0	0.0
class_weight	클래스별 가중치 설정 (불균형 데이터 처리)	dict, 'balanced', None	None
ccp_alpha	비용 복잡도 가지치기 파라미터 (트리의 크기 제어)	float, >= 0.0	0.0

#Regressor 하이퍼 파라미터
criterion	분할 기준 (불순도 측정 방법)	'squared_error', 'friedman_mse', 'absolute_error', 'poisson'	'squared_error'
splitter	노드를 분할할 때 사용할 전략	'best', 'random'	'best'
max_depth	트리의 최대 깊이. 깊이를 제한하면 과적합 방지에 도움	int, None	None
min_samples_split	노드를 분할하기 위한 최소 샘플 수	int, float	2
min_samples_leaf	리프 노드를 만들기 위한 최소 샘플 수	int, float	1
min_weight_fraction_leaf	가중치를 고려한 리프 노드의 최소 샘플 비율	float, 0.0 <= value <= 0.5	0.0
max_features	노드를 분할할 때 고려할 최대 특성 수	int, float, 'auto', 'sqrt', 'log2', None	None
random_state	난수 생성기 시드 (재현성 확보)	int, RandomState, None	None
max_leaf_nodes	트리의 최대 리프 노드 수	int, None	None
min_impurity_decrease	노드를 분할하기 위해 필요한 최소 불순도 감소량	float, >= 0.0	0.0
ccp_alpha	비용 복잡도 가지치기 파라미터 (트리의 크기 제어)	float, >= 0.0	0.0

#예시
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    ccp_alpha=0.01
)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='auto',
    random_state=42,
    ccp_alpha=0.01
)
