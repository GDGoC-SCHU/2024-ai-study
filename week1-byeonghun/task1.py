# 과제 1. 선형 회귀와 k-최근접 이웃을 사용한 삶의 만족도 예측
# 목표: 선형 회귀와 k-최근접 이웃을 사용하여 키프로스의 삶의 만족도를 예측하고, 결과를 비교하세요.

# 라이브러리 가져오기
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

##############################################################################################

# 1. 데이터 준비
data = {"GDP": [30000, 40000, 50000, 60000],
        "LS": [5.5, 6.0, 6.5, 7.0]}
df = pd.DataFrame(data)

df.plot(kind="scatter", x="GDP", y="LS") # 산점도를 통해 두 변수 간의 상관관계 확인
plt.title("Correlation between GDP and Life Satisfaction")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show() # 결과해석: 양의 상관관계를 가진다고 볼 수 있음

##############################################################################################

# 2. 모델 학습
X = df[["GDP"]].values
y = df[["LS"]].values # Target value

linear_model = LinearRegression()
linear_model.fit(X, y)

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X, y)

##############################################################################################

# 3. 예측
X_new = [[37655]]
prediction = linear_model.predict(X_new)
print(f"선형 회귀 예측 결과: {prediction[0, 0]:.2f}")

prediction_knn = knn_model.predict(X_new)
print(f"k-최근접 이웃 예측 결과: {prediction_knn[0, 0]:.2f}")

##############################################################################################

# 4. 결과 비교
# 선형회귀
# 장점: 해석이 쉬움, 선형적인 데이터에 적합
# 단점: 비선형적 데이터에 대해 적합하지 않음

# k-최근접 이웃
# 장점: 비선형 데이터에도 적용 가능
# 단점: 본 예시와 같이 데이터가 적으면 과적합 가능성이 높아짐, 최적의 k값을 설정할 때 주관이 들어감 