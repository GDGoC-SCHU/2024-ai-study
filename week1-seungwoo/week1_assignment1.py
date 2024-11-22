import pandas as pd
import matplotlib.pyplot as plt

data = {"GDP per capita (USD)" : [30000, 40000, 50000, 60000],
        "Life satisfaction" : [5.5, 6.0, 6.5, 7.0]}

MLdf = pd.DataFrame(data)

MLdf.plot(kind = "scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP&Life satisfaction relation")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()


## 선형 회귀 모델 
from sklearn.linear_model import LinearRegression

X = MLdf[["GDP per capita (USD)"]].values
y = MLdf[["Life satisfaction"]].values

model = LinearRegression()
model.fit(X, y)

X_new = [[37655]]
predict =model.predict(X_new)
print(f"선형 회귀 예측 결과: {predict[0, 0]:.2f}")

## K-최근접 이웃 회귀

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()
model.fit(X, y)

predict_knn = model.predict(X_new)
print(f"k-최근접 이웃 예측 : {predict_knn[0, 0]:.2f}")


