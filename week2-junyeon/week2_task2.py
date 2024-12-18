from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Petal length, Petal width 선택
y = iris.target

# 특징 선택
setosa_versicolor = (y == 0) | (y == 1)
X = X[setosa_versicolor]
y = y[setosa_versicolor]

y = np.where(y == 0, 0, 1)

print(f"특성 이름: {iris.feature_names[2:4]}")
print(f"타깃 이름: {[iris.target_names[0], iris.target_names[1]]}")
print(f"데이터 크기: {X.shape}\n")

# 모델 학습
model = SVC(kernel="linear")
model.fit(X, y)

# 지원 벡터 출력
print("지원 벡터(Support Vectors):")
print(model.support_vectors_)

# 데이터 산점도
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='green', label='Setosa')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Versicolor')

# 결정 경계 및 마진 계산
w = model.coef_[0]
b = model.intercept_[0]
x_boundary = np.linspace(0, 5, 100)
decision_boundary = -(w[0] * x_boundary + b) / w[1]
margin = 1 / np.sqrt(np.sum(w ** 2))
upper_margin = decision_boundary + margin
lower_margin = decision_boundary - margin

# 결정 경계 및 마진 시각화
plt.plot(x_boundary, decision_boundary, "k-")
plt.plot(x_boundary, upper_margin, "k--")
plt.plot(x_boundary, lower_margin, "k--")

# 서포트 벡터 시각화
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
    facecolors='none', edgecolors='black', label="Support Vectors")

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid()
plt.show()