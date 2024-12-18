from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = load_iris(as_frame=True)
x = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = iris.target

print("특성 이름 ", [iris.feature_names[2], iris.feature_names[3]])
print("타깃 이름 ", [iris.target_names[0], iris.target_names[1]])
print("데이터 크기 ", x.shape)

setosa_or_versicolor = (y == 0) | (y == 1)
x = x[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(x, y)

print("지원 벡터 ", svm_clf.support_vectors_)
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='SV')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.legend()
plt.title('Support Vectors')
plt.show()