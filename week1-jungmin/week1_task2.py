# -*- coding: utf-8 -*-
'''
GDGoC SCHU AI 스터디 1주차
과제 2 선형 회귀와 k-최근접 이웃(k-NN)을 사용한 삶의 만족도 예측

1. 제공 데이터를 활용한 GDP, 삶의 만족도 생성
2. 데이터프레임 변환
3. 시각화
4. 선형 회귀 정의 및 학습
5. k-최근접 이웃 정의 및 학습
6. 키프로스 1인당 GDP를 입력하여 두 모델의 예측 출력
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = datasets.load_iris()

# 데이터 샘플 출력
print("데이터셋 샘플(앞 5개)")
print(data.data[:5])
# 타깃 샘플 출력
print("타깃 샘플(앞 5개)")
print(data.target[:5])

# 특성 shape, 타깃 shape 출력
print("특성(shape)", data.data.shape)
print("타깃(shape)", data.target.shape)

# 데이터 분할
train_in, test_in, train_target, test_target = train_test_split(data.data, data.target, train_size=0.2, random_state=42)

# 모델 정의 및 훈련
lr = LogisticRegression()
lr.fit(train_in, train_target)

# 품종 예측
res=lr.predict(test_in)
print("모델 정확도", accuracy_score(test_target, res))

'''
# 실행 결과
`데이터셋 샘플(앞 5개)
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
타깃 샘플(앞 5개)
[0 0 0 0 0]
특성(shape) (150, 4)
타깃(shape) (150,)
모델 정확도 0.9666666666666667`
'''