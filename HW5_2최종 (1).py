#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


# In[4]:


file = 'trainHW5_2.csv'
df = pd.read_csv(file)

print('전체 데이터 건수 :', len(df)) #데이터 행수 확인

from sklearn.preprocessing import LabelEncoder



df = df.drop(columns=['SEASON_ID'])
df = df.drop(columns=['TEAM_ID'])
df = df.drop(columns=['position'])

df = df.fillna(-1)
print('전체 데이터 건수 :', len(df)) #데이터 행수 확인
pd.set_option('display.max_columns', None)
df.head()


# In[5]:


weights = np.array([1, 1, 1, 1, 1, 1, 1,1, 0.5,0.5,0.5,1, 1, 3, 5, 1, 3, 1, 1,1])

X = df.drop(columns=['MIN'])  # 특징 변수
y = df['MIN']  # 목표 변수

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
#X_train = X
#y_train = y


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 학습 데이터 기준으로 fit
#X_train= X_train_scaled * weights


# In[ ]:


svm_regressor = SVR(kernel='rbf')  # 'rbf' 커널을 사용

# 6. 모델 훈련
svm_regressor.fit(X_train, y_train)

# 7. 예측
y_pred = svm_regressor.predict(X_test)

# 8. 성능 평가 (MSE, Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[ ]:


import csv 

out_file = 'sample_sandbox_regressionHW5_2.csv'

test_file='testHW5_2.csv'

tdf = pd.read_csv(test_file)
tdf = tdf.drop(columns=['ID'])
tdf = tdf.drop(columns=['SEASON_ID'])
tdf = tdf.drop(columns=['TEAM_ID'])

scaler = StandardScaler()
x = scaler.fit_transform(tdf)

#x = x * weights

y=svm_regressor.predict(x)

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





# In[ ]:





# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
file = 'trainHW5_2.csv'
df = pd.read_csv(file)

print('전체 데이터 건수 :', len(df))

df.head()

# 'position' 열을 숫자형 레이블로 변환
label_encoder = LabelEncoder()
df['SEASON_ID'] = label_encoder.fit_transform(df['SEASON_ID'])

# 'SEASON_ID' 열 삭제 (불필요한 경우)
#df = df.drop(columns=['SEASON_ID', 'position'])
df = df.drop(columns=['position'])

# 결측값 처리
df = df.fillna(-1)

print('전체 데이터 건수 :', len(df))

# 목표 변수 'MIN'과 특징 변수 'X' 설정
X = df.drop(columns=['MIN'])
y = df['MIN']

weights = np.array([0.2, 0.4, 1, 1, 1, 1, 1, 1,1, 0.5,0.5,0.5, 2, 1, 3, 5, 1, 3, 1, 1, 1])


# 훈련 세트와 테스트 세트 분리
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
X_train = X
y_train = y

# 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train = X_train * weights
#X_test = scaler.transform(X_test)

# SVM 회귀 모델 생성
svm_regressor = SVR(kernel='rbf', gamma = 'scale', shrinking = True, C=1000, degree = 2, epsilon = 1)
# 모델 훈련
svm_regressor.fit(X_train, y_train)


# In[103]:


# 예측
X_test = scaler.fit_transform(X_test)
X_test = X_test * weights
y_pred = svm_regressor.predict(X_test)

# 성능 평가 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[8]:


import csv 

out_file = 'sample_sandbox_regressionHW5_2.csv'

test_file='testHW5_2.csv'


tdf = pd.read_csv(test_file)
tdf = tdf.drop(columns=['ID'])
#tdf = tdf.drop(columns=['SEASON_ID'])
#tdf = tdf.drop(columns=['TEAM_ID'])


tdf['SEASON_ID'] = label_encoder.fit_transform(tdf['SEASON_ID'])


scaler = StandardScaler()
x = scaler.fit_transform(tdf)

x = x * weights

y=svm_regressor.predict(x)

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


i


# In[ ]:




