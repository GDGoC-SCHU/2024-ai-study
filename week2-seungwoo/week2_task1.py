from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


#1.데이터 준비
iris = load_iris()
X = iris["data"][:, 3 :] #꽃잎 너비
y = (iris["target"] == 2).astype(int)

print(f"특성 이름:  {iris['feature_names'][:4]}")
print(f"타깃 이름: {iris['target_names'][:3]}")
print(f"데이터 크기: {X.shape}")

#2.데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)
print(f"훈련 세트 크기 : {X_train.shape}")
print(f"테스트 세트 크기 : {X_test.shape}")

model = LogisticRegression(C = 0.1, penalty= 'l1', solver = 'liblinear', max_iter=1000)

model.fit(X_train, y_train)
res = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score (y_test, res)

print(f"모델 정확도(Accuracy) : {accuracy}")



X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = model.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.plot(X_new, y_proba[:, 1], "g-", label = "Iris virginica")
plt.axvline(x=decision_boundary, color = 'r', linestyle = "--", label = "Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()

print(f"결정 경계: {decision_boundary}")

#모델의 accuracy가 매우 높은 편이고, 결정경계가 적절한 위치에서 데이터셋을 잘 분류하고 있음을 알 수 있다.