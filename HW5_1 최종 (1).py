#!/usr/bin/env python
# coding: utf-8

# In[177]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
file = 'trainHW5_1.csv'  # 여기서 csv 파일을 입력하세요
df = pd.read_csv(file)

print('전체 데이터 건수 :', len(df))

label_encoder = LabelEncoder()
df['position'] = label_encoder.fit_transform(df['position'])

# 필요한 경우, 다른 열 삭제 (불필요한 경우)
df = df.drop(columns=['SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN'])  # 'SEASON_ID', 'TEAM_ID' 열을 제거

# 결측치 처리: -1로 채우기
df = df.fillna(-1)

X = df.drop(columns=['position'])  # 특징 변수
y = df['position']  # 목표 변수

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
X_train = X
y_train = y
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 훈련 데이터에 대해 fit & transform
#X_test = scaler.transform(X_test)


weights = np.array([1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5
                    , 2, 1, 3, 5, 1, 3, 1, 1, 1]) # 5ㄹ,ㄹ 4ㅇ,러 바꿈
X_train = X_train * weights # weight가 들어가면 0.03이 더 좋아짐


# In[178]:


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


# In[172]:


X_test = X_test * weights

# 예측
y_pred = gb_classifier.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# 분류 리포트 (Precision, Recall, F1-score 등)
print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[179]:


import csv 

out_file = 'sample_sandbox_classificationHW5_1.csv'

test_file='testHW5_1.csv'

tdf = pd.read_csv(test_file)
tdf = tdf.drop(columns=['ID'])
tdf = tdf.drop(columns=['SEASON_ID'])
tdf = tdf.drop(columns=['TEAM_ID'])

scaler = StandardScaler()
x = scaler.fit_transform(tdf)

x = x * weights

y = gb_classifier.predict(x)
y = label_encoder.inverse_transform(y)

with open(out_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)

for i, data in enumerate(y, start=1):
    if len(rows) > i:  # 데이터가 있는 행에만 삽입
        rows[i][1] = data  # B열에 예측 결과 삽입

with open(out_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)


# In[ ]:




