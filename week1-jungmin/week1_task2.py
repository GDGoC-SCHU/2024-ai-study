# -*- coding: utf-8 -*-
'''
GDGoC SCHU AI 스터디 1주차
과제 2 Iris 데이터셋을 이용한 Logistic Regression 학습 수행 및 정확도 평가
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