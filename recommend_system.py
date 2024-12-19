import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# CSV 파일 로드
df = pd.read_csv('your_data.csv')

# 데이터 형태 예시
# df = pd.DataFrame({
#     'user_id': [1, 1, 2, 2, 3, 3],
#     'item_id': [101, 102, 101, 103, 102, 103],
#     'rating': [5, 3, 4, 5, 2, 3]
# })

# 피벗 테이블로 변환 (아이템을 행으로, 사용자별로 평점을 열로)
ratings_matrix = df.pivot_table(index='item_id', columns='user_id', values='rating').fillna(0)

# KNN 모델 학습
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)  # 6개 이웃을 찾음
knn.fit(ratings_matrix.values)

# 특정 아이템(예: item_id 101)과 가장 유사한 5개의 아이템 찾기
item_id = 101
item_idx = ratings_matrix.index.get_loc(item_id)  # 아이템의 인덱스를 찾기

# KNN으로 가장 유사한 아이템 찾기
distances, indices = knn.kneighbors(ratings_matrix.iloc[item_idx].values.reshape(1, -1), n_neighbors=6)

# 유사한 아이템 출력 (자신을 제외한 5개의 아이템)
print(f'아이템 {item_id}와 유사한 아이템:')
for i in range(1, len(indices.flatten())):
    similar_item = ratings_matrix.index[indices.flatten()[i]]
    print(f'아이템 {similar_item} (유사도: {1 - distances.flatten()[i]:.2f})')


# 또 다른 예시 코드
df = pd.read_csv('your_data.csv')

# 1. 데이터 준비: 사용자-아이템 평점 매트릭스로 변환
ratings_matrix = df.pivot_table(index='item_id', columns='user_id', values='rating').fillna(0)

# 2. KNN 모델 정의
knn = NearestNeighbors()

# 3. 하이퍼파라미터 탐색 범위 설정
param_grid = {
    'n_neighbors': [3, 5, 7, 10],  # 가장 가까운 이웃 수
    'metric': ['cosine', 'euclidean', 'manhattan'],  # 거리 계산 방법
    'algorithm': ['brute', 'kd_tree', 'ball_tree'],  # 거리 계산 알고리즘
    'leaf_size': [10, 20, 30]  # kd_tree나 ball_tree 알고리즘에 해당하는 파라미터
}

# 4. GridSearchCV를 사용하여 하이퍼파라미터 튜닝
grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, verbose=1)

# 5. 학습 및 튜닝
grid_search.fit(ratings_matrix.values)

# 6. 최적의 하이퍼파라미터 출력
print("Best hyperparameters found: ", grid_search.best_params_)

# 7. 최적의 KNN 모델로 예측하기
best_knn = grid_search.best_estimator_

# 8. 예시로 특정 아이템과 가장 유사한 아이템 찾기
item_id = 101
item_idx = ratings_matrix.index.get_loc(item_id)  # 아이템의 인덱스를 찾기

# 9. KNN으로 가장 유사한 아이템 찾기
distances, indices = best_knn.kneighbors(ratings_matrix.iloc[item_idx].values.reshape(1, -1), n_neighbors=6)

# 10. 유사한 아이템 출력 (자신을 제외한 5개의 아이템)
print(f'아이템 {item_id}와 유사한 아이템:')
for i in range(1, len(indices.flatten())):
    similar_item = ratings_matrix.index[indices.flatten()[i]]
    print(f'아이템 {similar_item} (유사도: {1 - distances.flatten()[i]:.2f})')

# 11. 모델 하이퍼파라미터와 성능을 CSV 파일로 저장
# 하이퍼파라미터와 추천 결과를 CSV로 저장하기

# 최적의 하이퍼파라미터를 저장
best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv('best_hyperparameters.csv', index=False)

# 유사한 아이템 결과를 저장
similar_items = []
for i in range(1, len(indices.flatten())):
    similar_item = ratings_matrix.index[indices.flatten()[i]]
    similar_items.append({
        'item_id': item_id,
        'similar_item': similar_item,
        'similarity': 1 - distances.flatten()[i]
    })

similar_items_df = pd.DataFrame(similar_items)
similar_items_df.to_csv('similar_items.csv', index=False)

print("Best hyperparameters and similar items saved to CSV files.")
