# -*- coding: utf-8 -*-
'''
GDGoC SCHU AI 스터디 1주차
과제 1 선형 회귀와 k-최근접 이웃(k-NN)을 사용한 삶의 만족도 예측

1. 제공 데이터를 활용한 GDP, 삶의 만족도 생성
2. 데이터프레임 변환
3. 시각화
4. 선형 회귀 정의 및 학습
5. k-최근접 이웃 정의 및 학습
6. 키프로스 1인당 GDP를 입력하여 두 모델의 예측 출력
'''
# 데이터 생성
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data = {
    "GDP per capita (USD)": [ 30000, 40000, 50000, 60000 ],
    "Life satisfaction": [ 5.5, 6.0, 6.5, 7.0 ]
}
# 데이터 프레임 변환
df = pd.DataFrame(data)

# 시각화
df.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP와 삶의 만족도 관계")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()

# 모델 정의, 학습, 예측
x = df[["GDP per capita (USD)"]].values
y = df[["Life satisfaction"]].values

# 선형 회귀 정의와 학습
linear_model = LinearRegression()
linear_model.fit(x, y)

# k-최근접 이웃 정의와 학습
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x, y)

# 키프로스 1인당 GDP 입력과 예측
gdp_input=[[int(input("삶의 만족도 예측을 원하는 GDP 값을 입력하세요."))]]
country_input=input("이 GDP는 어느 국가의 수치인가요?")

linear_predict = linear_model.predict(gdp_input)
knn_predict = knn_model.predict(gdp_input)

print(country_input, "의 삶의 질은")
print(f"선형 회귀 분석 결과 {linear_predict[0, 0]:.2f}")
print (f"k-NN 회귀 분석 결과 {knn_predict[0, 0]:.2f}")
print("로 예상됩니다.")

'''
# 출력 결과

`삶의 만족도 예측을 원하는 GDP 값을 입력하세요.37655
이 GDP는 어느 국가의 수치인가요?키프로스
키프로스 의 삶의 질은
선형 회귀 분석 결과 5.88
k-NN 회귀 분석 결과 6.00`

두 분석 결과는 예시와 동일하며, 장단점을 정리하면 아래와 같다.

## 선형 회귀
- 장점: 삶의 질 데이터와 같은 연속적인 데이터에서 분류 기법 대비 좀 더 정밀한 예측값을 보여준다.
- 단점: 비연속적 데이터에는 적합하지 않다.

## KNN
- 장점: 가장 근접한 데이터끼리 같은 값을 나타내므로 분포를 나타내기에 적합하다.
- 단점: 최적의 k값을 사람이 직접 찾아 지정할 필요가 있다.
'''
