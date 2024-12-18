import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# 1. CSV 파일 불러오기
file_path = 'data.csv'  # 파일 경로를 입력하세요
df = pd.read_csv(file_path)

# 2. 데이터 확인
print("데이터 확인:")
print(df.head())

# 3. 문자열 데이터에 대해 원핫 인코딩
string_columns = df.select_dtypes(include=['object']).columns  # 문자열 데이터 컬럼 선택

encoder = OneHotEncoder(sparse=False)  # 원핫인코딩 객체 생성
encoded_data = encoder.fit_transform(df[string_columns])  # 원핫인코딩 수행

# 인코딩된 데이터프레임 생성
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(string_columns))

# 원본 데이터프레임에서 문자열 컬럼 제거하고 원핫인코딩된 데이터 병합
df = df.drop(columns=string_columns)  # 기존 문자열 컬럼 제거
df = pd.concat([df, encoded_df], axis=1)

# 4. 결과 데이터 확인
print("\n원핫인코딩 후 데이터:")
print(df.head())

# 5. 머신러닝 알고리즘에 적용 예시

# 타겟(레이블)과 피처 분리
X = df.drop(columns=['target'])  # 'target'은 예시 레이블 컬럼 (실제 타겟 컬럼 이름으로 수정)
y = df['target']

# 데이터 분할 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 예제
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("\nKNN 예측 결과:", knn.predict(X_test))

# SVM 모델 예제
svm = SVC()
svm.fit(X_train, y_train)
print("\nSVM 예측 결과:", svm.predict(X_test))

# 클러스터링 (K-Means) 예제
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print("\nKMeans 클러스터링 결과 (라벨):", kmeans.labels_)
