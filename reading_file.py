import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'train.csv'
df = pd.read_csv(file)

df.head() #데이터 미리보기
print('전체 데이터 건수 :', len(df)) #데이터 행수 확인
df = df.dropna()
print('전체 데이터 건수 :', len(df)) #데이터 행수 확인

# columns drop하기
df = df.drop(columns=['FG_PCT'])
df = df.drop(columns=['FG3_PCT'])
df = df.drop(columns=['FT_PCT'])
df = df.drop(columns=['SEASON_ID'])

# 'label encoding'도 처리
label_encoder = LabelEncoder()
df['SEASON_ID'] = label_encoder.fit_transform(df['SEASON_ID'])

# 'reverse label encoding'도 처리
y=knn.predict(x)
y = label_encoder.inverse_transform(y)

# 파일 csv로 출력하는 법
import csv
with open(out_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)

#y가 결과값
for i, data in enumerate(y, start=1):  # start=1 -> B2부터 시작
    if len(rows) > i:  # 데이터가 있는 행에만 삽입
        rows[i][1] = data  # B열 (두 번째 열, 인덱스 1)에 값 삽입

with open(out_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
