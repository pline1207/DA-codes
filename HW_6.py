#!/usr/bin/env python
# coding: utf-8

# In[460]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

file = 'survey.csv'
data = pd.read_csv(file)
print('전체 데이터 건수 :', len(data))

label_encoder_occ = LabelEncoder()
data['Occupation'] = label_encoder_occ.fit_transform(data['Occupation'])

label_encoder_sex = LabelEncoder()
data['sex'] = label_encoder_sex.fit_transform(data['sex'])

label_encoder_rac = LabelEncoder()
data['race'] = label_encoder_rac.fit_transform(data['race'])

label_encoder_rel = LabelEncoder()
data['relationship'] = label_encoder_rel.fit_transform(data['relationship'])

#label_encoder_nc = LabelEncoder()
#data['native country'] = label_encoder_nc.fit_transform(data['native country'])

#label_encoder_inc = LabelEncoder()
#data['income'] = label_encoder_inc.fit_transform(data['income'])

label_encoder_mat = LabelEncoder()
data['marital status'] = label_encoder_mat.fit_transform(data['marital status'])

numeric_data = data.select_dtypes(include=['float64', 'int64', 'int32'])
numeric_data.head()


# In[409]:


from sklearn.preprocessing import LabelEncoder

#label_encoder_edu = LabelEncoder()
#data['education'] = label_encoder_edu.fit_transform(data['education'])

label_encoder_mat = LabelEncoder()
data['marital status'] = label_encoder_mat.fit_transform(data['marital status'])

label_encoder_occ = LabelEncoder()
data['Occupation'] = label_encoder_occ.fit_transform(data['Occupation'])

label_encoder_rel = LabelEncoder()
data['relationship'] = label_encoder_rel.fit_transform(data['relationship'])

label_encoder_rac = LabelEncoder()
data['race'] = label_encoder_rac.fit_transform(data['race'])

label_encoder_sex = LabelEncoder()
data['sex'] = label_encoder_sex.fit_transform(data['sex'])

label_encoder_nc = LabelEncoder()
data['native country'] = label_encoder_nc.fit_transform(data['native country'])

label_encoder_inc = LabelEncoder()
data['income'] = label_encoder_inc.fit_transform(data['income'])

data.head()


# In[461]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv


# In[462]:


# NaN 값 처리
numeric_data = numeric_data.fillna(numeric_data.mean())

# 데이터 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)



# In[463]:


weight = [0.5, 0.5, 0.001, 7, 1, 0.7, 1.1, 1, 1, 1]
scaled_data = scaled_data * weight


# In[464]:


# 3. KMeans 클러스터링
n_clusters = 8  # 원하는 클러스터 개수를 설정하세요.
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
#kmeans = KMeans(n_clusters=n_clusters, random_state=42, init = 'k-means++', n_init=10, algorithm='elkan')
clusters = kmeans.fit_predict(scaled_data)

# 클러스터 결과를 원본 데이터에 추가
data['Cluster'] = clusters

y_out = data['Cluster']

# 4. 결과 저장
out_file = 'sample_submission.csv'

# 8. 결과를 CSV에 저장
with open(out_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)

for i, data in enumerate(y_out, start=1):  # start=1 -> B2부터 시작
    if len(rows) > i:  # 데이터가 있는 행에만 삽입
        rows[i][1] = data  # B열 (두 번째 열, 인덱스 1)에 값 삽입

with open(out_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)


# In[91]:


import seaborn as sns

sns.pairplot(data, diag_kind='kde')
plt.show()


# In[ ]:




