# 과제: 선형 SVM 분류

## 목표

- Iris 데이터셋에서 Iris Setosa와 Iris Versicolor를 분류하는 선형 SVM
- 모델을 학습하고, 결정 경계를 시각화합니다.

## 1. 데이터 준비

### sklearn.datasets.load_iris를 사용하여 데이터를 로드합니다.

```
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
```

### 꽃잎 길이(Petal length)와 꽃잎 너비(Petal width)를 특징(feature)으로 선택합니다.

```
x = iris.data[["petal length (cm)", "petal width (cm)"]].values
```

### Iris Setosa와 Iris Versicolor 클래스만 선택합니다.

```
setosa_or_versicolor = (y == 0) | (y == 1)
x = x[setosa_or_versicolor]
y = y[setosa_or_versicolor]
```

## 2. 모델 학습

### SVC(kernel="linear")를 사용하여 선형 SVM 모델을 정의합니다.

```
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=1)
```

### 모델을 학습시키고, 학습된 모델의 지원 벡터(Support Vectors)를 출력합니다.

```
svm_clf.fit(x, y)

print("Support Vectors:", svm_clf.support_vectors_)
```

### 3. 결정 경계 시각화

#### matplotlib를 사용하여 데이터를 산점도로 시각화합니다.

### 선형 SVM의 결정 경계와 마진을 플롯합니다.

```
# 결정 경계와 마진을 그리기 위한 함수
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # 결정 경계
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    # 마진
    margin = 1 / np.sqrt(np.sum(svm_clf.coef_ ** 2))
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# 데이터 산점도
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="winter", s=50)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")

# 서포트 벡터
plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=200, facecolors='none', edgecolors='k')

# 결정 경계와 마진 플롯
plot_svc_decision_boundary(svm_clf, 0, 5.5)

plt.title("SVM Decision Boundary with Margins")
plt.show()
```
