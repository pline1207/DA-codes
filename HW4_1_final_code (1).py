#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'train_clf.csv'
df = pd.read_csv(file)

print('전체 데이터 건수 :', len(df)) #데이터 행수 확인

from sklearn.preprocessing import LabelEncoder

# 'position' 열을 숫자형 레이블로 변환
label_encoder = LabelEncoder()
df['position'] = label_encoder.fit_transform(df['position'])

# 'season'도 처리
#label_encoder = LabelEncoder()
#df['SEASON_ID'] = label_encoder.fit_transform(df['SEASON_ID'])

df = df.drop(columns=['GP'])
df = df.drop(columns=['GS'])
df = df.drop(columns=['MIN'])

df = df.drop(columns=['FG_PCT'])
df = df.drop(columns=['FG3_PCT'])
df = df.drop(columns=['SEASON_ID'])

df = df.fillna(-1)
print('전체 데이터 건수 :', len(df)) #데이터 행수 확인
df.head()


# In[19]:


X = df.drop(columns=['position'])
y = df['position']

X.head()


# In[20]:


from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train=X
y_train=y
#y_test = label_encoder.inverse_transform(y_test)


# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='minkowski', algorithm='ball_tree', leaf_size=20, p=1 )  
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)


# In[ ]:





# In[22]:


out_file='sample_sandbox_classification.csv'


# In[23]:


test_file='test_clf.csv'
from sklearn.preprocessing import LabelEncoder
tdf = pd.read_csv(test_file)

tdf = tdf.drop(columns=['ID'])
tdf = tdf.drop(columns=['FG_PCT'])
tdf = tdf.drop(columns=['FG3_PCT'])
tdf = tdf.drop(columns=['SEASON_ID'])


# In[24]:


odf = pd.read_csv(out_file)


# In[25]:


scaler = StandardScaler()
x = scaler.fit_transform(tdf)


# In[26]:


y=knn.predict(x)
y = label_encoder.inverse_transform(y)


# In[27]:


import csv
with open(out_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)


# In[28]:


for i, data in enumerate(y, start=1):  # start=1 -> B2부터 시작
    if len(rows) > i:  # 데이터가 있는 행에만 삽입
        rows[i][1] = data  # B열 (두 번째 열, 인덱스 1)에 값 삽입


# In[29]:


with open(out_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)


# In[ ]:





# In[13]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# KNN 모델 정의
knn = KNeighborsClassifier()

# 하이퍼파라미터 그리드 정의
param_grid = {
    'leaf_size': [20, 30, 40, 50, 60],  # 다양한 leaf_size 값 실험
    'n_neighbors': [5, 7, 9],           # k 값
    'weights': ['uniform', 'distance'], # 가중치 설정
    'metric': ['minkowski', 'euclidean']
}

# GridSearchCV로 최적의 파라미터 찾기
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# 최적의 하이퍼파라미터 출력
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Best Accuracy: {grid_search.best_score_:.4f}')


# In[ ]:




