from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target


print(f"특성 이름:  {iris["feature_names"][:2]}")
print(f"타깃 이름: {iris["target_names"][:2]}")
print(f"데이터 크기: {X[:100].shape}")

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)

print("Support Vectors:", svm_clf.support_vectors_)

import matplotlib.pyplot as plt

x0s = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
x1s = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
x0, x1 = np.meshgrid(x0s, x1s)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_decision = svm_clf.decision_function(X_new).reshape(x0.shape)

# 데이터 산점도
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="green", label="Setosa", s=50)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Versicolor", s=50)


plt.contour(x0, x1, y_decision, levels=[-1, 0, 1], colors="k", linestyles=["dashed", "solid", "dashed"], linewidths=1)

plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=200, facecolors='none', edgecolors="black", label="Support Vectors")


plt.title("SVM Decision Boundary with Margins", fontsize=14, pad=15)
plt.xlabel("Petal length (cm)", fontsize=12)
plt.ylabel("Petal width (cm)", fontsize=12)


plt.legend(loc="upper left", fontsize=10)
plt.grid(visible=True, alpha=0.6)
plt.tight_layout()  
plt.show()