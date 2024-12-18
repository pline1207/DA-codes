svm_model = SVC(kernel='rbf', C=1.0, gamma='auto', degree=3, coef0=0, probability=False, tol=1e-3, random_state=43)
y=svm_model.predict(x)

#SVC 하이퍼 파라미터
C	오류 허용 정도를 조절하는 규제 파라미터 (높을수록 오류에 엄격)	float, > 0	1.0
kernel	커널 함수의 종류를 설정	'linear', 'poly', 'rbf', 'sigmoid', 사용자 정의 함수	'rbf'
degree	다항 커널('poly')의 차수	int, > 0	3
gamma	커널 함수의 계수 ('rbf', 'poly', 'sigmoid'에 적용됨)	'scale', 'auto', float	'scale'
coef0	'poly'와 'sigmoid' 커널에서 사용하는 상수항	float	0.0
shrinking	Shrinking heuristic 적용 여부 (속도 최적화 옵션)	bool, True 또는 False	True
probability	확률 추정 수행 여부 (추정 시 predict_proba 사용 가능)	bool, True 또는 False	False
tol	수렴 허용 오차 (최적화 종료 조건)	float, > 0	1e-3
cache_size	커널 계산 시 사용하는 캐시 메모리 크기 (MB 단위)	float, > 0	200
class_weight	클래스 가중치 설정 (불균형 데이터 처리에 사용)	dict, 'balanced', None	None
verbose	학습 과정 출력 여부	bool, True 또는 False	False
max_iter	최적화 반복 횟수 제한 (음수 값은 무한대로 설정)	int, > 0 또는 -1	-1
decision_function_shape	다중 클래스 분류 방식 설정 ('ovr': one-vs-rest, 'ovo': one-vs-one)	'ovr', 'ovo'	'ovr'
break_ties	동점 분류 결과 처리 여부 (다중 클래스에서만 적용됨)	bool	False
random_state	난수 생성기 시드 (재현성 확보)	int, None	None

#SVR 하이퍼 파라미터
C	규제 파라미터. 높은 값일수록 오류를 줄이지만 과적합 위험도 증가	float, > 0	1.0
kernel	커널 함수의 종류를 설정	'linear', 'poly', 'rbf', 'sigmoid', 사용자 정의 함수	'rbf'
degree	다항 커널('poly')의 차수	int, > 0	3
gamma	커널 함수의 계수 ('rbf', 'poly', 'sigmoid'에 적용됨)	'scale', 'auto', float	'scale'
coef0	'poly'와 'sigmoid' 커널에서 사용하는 상수항	float	0.0
tol	수렴 허용 오차 (최적화 종료 조건)	float, > 0	1e-3
epsilon	무감각 영역 크기 설정 (이 범위 내의 오류를 무시함)	float, >= 0	0.1
shrinking	Shrinking heuristic 적용 여부 (속도 최적화 옵션)	bool, True 또는 False	True
cache_size	커널 계산 시 사용하는 캐시 메모리 크기 (MB 단위)	float, > 0	200
verbose	학습 과정 출력 여부	bool, True 또는 False	False
max_iter	최적화 반복 횟수 제한 (음수 값은 무한대로 설정)	int, > 0 또는 -1	-1

# 예시코드
from sklearn.svm import SVC

model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    degree=3,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight='balanced',
    verbose=False,
    max_iter=-1
)

# 예시코드
from sklearn.svm import SVR

model = SVR(
    C=1.0,
    kernel='poly',
    degree=3,
    gamma='auto',
    epsilon=0.1,
    shrinking=True,
    tol=1e-3,
    verbose=False,
    max_iter=-1
)
