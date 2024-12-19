#!/usr/bin/env python
# coding: utf-8

# In[336]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'train_reg.csv'
df = pd.read_csv(file)

df.head() #데이터 미리보기

print('전체 데이터 건수 :', len(df)) #데이터 행수 확인

from sklearn.preprocessing import LabelEncoder

# 'position' 열을 숫자형 레이블로 변환
label_encoder = LabelEncoder()
df['position'] = label_encoder.fit_transform(df['position'])

# 'season'도 처리|
label_encoder = LabelEncoder()
df['SEASON_ID'] = label_encoder.fit_transform(df['SEASON_ID'])
df = df.drop(columns=['TEAM_ID'])
df = df.drop(columns=['position'])
df = df.drop(columns=['FG_PCT'])
df = df.drop(columns=['FG3_PCT'])
df = df.drop(columns=['SEASON_ID'])

df = df.fillna(-2)
print('전체 데이터 건수 :', len(df)) #데이터 행수 확인

df.head()


# In[337]:


X = df.drop(columns=['MIN'])
y = df['MIN']
X.head()


# In[338]:


from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X
y_train = y


# In[368]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_regressor = DecisionTreeRegressor(max_depth = 12, random_state=42, min_samples_leaf = 4, criterion = 'squared_error')
dt_regressor.fit(X_train_scaled, y_train)
y_pred = dt_regressor.predict(X_test_scaled)


# In[360]:


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² Score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')


# In[361]:


out_file='sample_sandbox_regression.csv'
import csv
with open(out_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)


# In[362]:


test_file='test_reg.csv'
tdf = pd.read_csv(test_file)


# 'position' 열을 숫자형 레이블로 변환
label_encoder = LabelEncoder()

# 'season'도 처리
label_encoder = LabelEncoder()
tdf = tdf.drop(columns=['ID'])
tdf = tdf.drop(columns=['FG_PCT'])
tdf = tdf.drop(columns=['FG3_PCT'])
tdf = tdf.drop(columns=['SEASON_ID'])
tdf = tdf.drop(columns=['TEAM_ID'])


# In[363]:


odf = pd.read_csv(out_file)


# In[364]:


scaler = StandardScaler()
x = scaler.fit_transform(tdf)


# In[365]:


y=dt_regressor.predict(x)


# In[366]:


for i, data in enumerate(y, start=1):  # start=1 -> B2부터 시작
    if len(rows) > i:  # 데이터가 있는 행에만 삽입
        rows[i][1] = data  # B열 (두 번째 열, 인덱스 1)에 값 삽입


# In[367]:


with open(out_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)


# 

# In[ ]:




