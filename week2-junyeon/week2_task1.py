from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = load_iris()
x = iris["data"][:, 3:]  # 꽃잎 너비
y = (iris["target"] == 2).astype(int)  # Iris virginica 여부

print("특성 이름 ", iris.feature_names)
print("타깃 이름 ", iris.target_names)
print("데이터 크기 ", x.shape)

# 데이터 분할 및 모델 학습
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("훈련 세트 크기 ", x_train.shape)
print("테스트 세트 크기", x_test.shape)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train) # 모델 학습

res=log_reg.predict(x_test) # 예측 수행
accuracy=accuracy_score(y_test, res) # 정확도 계산

print("모델 정확도(Accuracy) ", accuracy) # 결과 출력

# 결정 경계 시각화
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)
decision_boundary = x_new[y_proba[:, 1] >= 0.5][0]

plt.plot(x_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.axvline(x=decision_boundary, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()

# 해당 모델은 정확도가 1.0으로 주어진 데이터셋에서는 매우 높은 성능을 보인다.
# 결정 경계도 Petal width가 0.5 이상인 값들 중에서 설정하였고 Iris virginica를 분류하기에 적절하였다. 
# 해당 데이터셋은 단순하여 하나의 특성(Petal width)만으로도 분류가 정확히 되었지만 
# 일반화를 위해서는 여러가지의 특성을 사용하는 것이 더 효율적일 것 같다.