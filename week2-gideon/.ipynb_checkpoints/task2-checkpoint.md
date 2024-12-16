
# 5장 - 서포트 벡터 머신 (SVM)

서포트 벡터 머신(SVM)은 강력한 분류, 회귀 및 이상 탐지 모델입니다. 

---


### 1. 선형 SVM 분류
- 선형적으로 분리 가능한 데이터에 대해 사용됩니다.
- SVM은 최대 마진으로 결정 경계를 정의하여 클래스 간 분리를 최적화합니다.

#### 예시 코드
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)

print("Support Vectors:", svm_clf.support_vectors_)
```

#### 예상 출력 결과
```
Support Vectors: [[1.9 0.4]
 [3.  1.1]]
```
---

### 2. 비선형 SVM 분류
- 선형적으로 분리되지 않는 데이터를 처리하기 위해 커널 기법을 사용합니다.
- 일반적으로 사용되는 커널: 다항식 커널, RBF(가우시안) 커널.

#### 예시 코드
```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=5, C=0.001))
svm_clf.fit(X, y)

print("SVM RBF Kernel Support Vectors:", svm_clf[-1].support_)
```

#### 예상 출력 결과
```
SVM RBF Kernel Support Vectors: [ 3  4  5  6  8  9 10 11 13 15 18 19 22 23 25 26 28 30 31 33 35 38 39 40
 41 42 44 45 47 49 50 51 52 53 54 56 58 63 66 71 73 77 82 83 84 86 87 88
 94 97  0  1  2  7 12 14 16 17 20 21 24 27 29 32 34 36 37 43 46 48 55 57
 59 60 61 62 64 65 67 68 69 70 72 74 75 76 78 79 80 81 85 89 90 91 92 93
 95 96 98 99]
```

---

### 3. SVM 회귀
- SVM은 회귀 문제에도 사용할 수 있으며, 특정 마진 내에서 최대한 데이터를 포함하도록 모델을 학습합니다.

#### 예시 코드
```python
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(50)

svm_reg = make_pipeline(StandardScaler(), SVR(kernel="poly", degree=2, C=10, epsilon=0.1))
svm_reg.fit(X, y)

print("SVR Predictions:", svm_reg.predict(X[:5]))
```
#### 예상 출력 결과
```
SVR Predictions: [6.65551825 7.23838276 6.83463758 6.69807393 6.83979358]
```
---

### 4. 과제: 선형 SVM 분류

---

## 목표
**Iris 데이터셋**에서 `Iris Setosa`와 `Iris Versicolor`를 분류하는 선형 SVM 모델을 학습하고, 결정 경계를 시각화합니다.

---

## 단계별 과제

### 1. 데이터 준비
1. `sklearn.datasets.load_iris`를 사용하여 데이터를 로드합니다.
2. **꽃잎 길이(Petal length)**와 **꽃잎 너비(Petal width)**를 특징(feature)으로 선택합니다.
3. `Iris Setosa`와 `Iris Versicolor` 클래스만 선택합니다.

#### 예상 출력 결과
```
특성 이름: ['petal length (cm)', 'petal width (cm)']
타깃 이름: ['setosa', 'versicolor']
데이터 크기: (100, 2)
```

### 2. 모델 학습
1. `SVC(kernel="linear")`를 사용하여 선형 SVM 모델을 정의합니다.
2. 모델을 학습시키고, 학습된 모델의 **지원 벡터(Support Vectors)**를 출력합니다.

#### 예상 출력 결과
```
지원 벡터(Support Vectors):
[[1.9 0.4]
 [3.  1.1]]
```

### 3. 결정 경계 시각화
1. **matplotlib**를 사용하여 데이터를 산점도로 시각화합니다.
2. 선형 SVM의 결정 경계와 마진을 플롯합니다.

<img width="599" alt="스크린샷 2024-11-25 오후 8 16 15" src="https://github.com/user-attachments/assets/e7d30471-b062-4f10-be50-e4a0a474549d">

위의 출력 결과를 보고 코드를 작성해주세요.

