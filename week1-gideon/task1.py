import matplotlib.pyplot as plt
import pandas as pd

data = {
    "GDP per capita (USD)": [30000, 40000, 50000, 60000],
    "Life satisfaction": [5.5, 6.0, 6.5, 7.0]
}
df = pd.DataFrame(data)

df.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP와 삶의 만족도 관계")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()

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

# k-최근접 이웃 모델
from sklearn.neighbors import KNeighborsRegressor

# k-최근접 이웃 회귀 모델 생성
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# 예측
prediction_knn = model.predict(X_new)
print(f"k-최근접 이웃 예측 결과: {prediction_knn[0, 0]:.2f}")