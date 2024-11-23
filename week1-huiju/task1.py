import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
data = {"GDP per capita (USD)": [30000, 40000, 50000, 60000],
        "Life satisfaction": [5.5, 6.0, 6.5, 7.0]}
df = pd.DataFrame(data)

# 데이터 시각화
df.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP와 삶의 만족도 관계")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()

# 1. 선형 회귀 모델
from sklearn.linear_model import LinearRegression

# 독립 변수(X)와 종속 변수(y)
X = df[["GDP per capita (USD)"]].values
y = df[["Life satisfaction"]].values

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측
X_new = [[37655]]  # 키프로스의 1인당 GDP
prediction = model.predict(X_new)
print(f"선형 회귀 예측 결과: {prediction[0, 0]:.2f}")

# 2. k-최근접 이웃 모델
from sklearn.neighbors import KNeighborsRegressor

# k-최근접 이웃 회귀 모델 생성
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# 예측
prediction_knn = model.predict(X_new)
print(f"k-최근접 이웃 예측 결과: {prediction_knn[0, 0]:.2f}")

# 두 모델의 장단점
# 선형회귀
# 장점 : 간단하고 해석하기 쉬우며, 빠르게 계산할 수 있음. 선형 관계가 있는 데이터에 매우 효과적임.
# 단점 : 독립 변수와 종속 변수 간의 선형 관계 가정이 필요하며, 이상치에 민감함.

# k-최근접 이웃 회귀(k-NN)
# 장점 : 데이터 분포를 잘 반영하며, 복잡한 패턴을 다룰 수 있음.
# 단점 : 데이터 양이 많을수록 비용이 증가하며, 최적의 k값을 찾는 것이 중요함.