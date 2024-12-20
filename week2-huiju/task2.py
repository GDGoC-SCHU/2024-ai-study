from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 준비
iris = load_iris()
X = iris.data[:, 2:4]  # 꽃잎 길이와 꽃잎 너비
y = iris.target

# Iris Setosa와 Iris Versicolor 클래스만 선택
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# 데이터 정보 출력
print("특성 이름:", iris.feature_names[2:4])
print("타깃 이름:", iris.target_names[:2])
print("데이터 크기:", X.shape)

# 2. 모델 학습
svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)

# 지원 벡터 출력
print("지원 벡터(Support Vectors):")
print(svm_clf.support_vectors_)

# 3. 결정 경계 시각화
def plot_decision_boundary(svm_clf, X, y):
    # 결정 경계 및 마진 계산
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 결정 경계 및 마진 시각화
    plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=1, linewidths=2)  # 결정 경계
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 1], linestyles='dashed')  # 마진

    # 데이터 포인트 시각화
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='green', label='Setosa', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Versicolor', marker='s')

    # 지원 벡터 표시
    plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1],
                facecolors='none', edgecolors='k', s=200, label='Support Vectors', marker='o')
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend()
    plt.grid()

# 시각화 함수 호출
plot_decision_boundary(svm_clf, X, y)
plt.show()
