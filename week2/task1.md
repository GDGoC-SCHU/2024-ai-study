# 4장 – 모델 훈련


### 1. 선형 회귀

#### 정의
- **선형 회귀**는 입력 변수 \(X\)와 출력 변수 \(y\) 간의 선형 관계를 모델링함
- 모델의 수식은 다음과 같다:

<img width="705" alt="스크린샷 2024-11-25 오후 7 19 09" src="https://github.com/user-attachments/assets/0c0ccf34-89c4-4cc2-b339-af7be30c15a8">



#### 정규 방정식
- 선형 회귀에서 파라미터 \(\theta\)를 다음과 같은 **정규 방정식**을 통해 계산할 수 있다:

<img width="583" alt="스크린샷 2024-11-25 오후 7 20 06" src="https://github.com/user-attachments/assets/b8ad0f31-e1da-4833-8815-7c090a222049">

#### 예제 코드
```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# 정규 방정식을 사용한 파라미터 계산
X_b = np.c_[np.ones((m, 1)), X]  # X에 x0 = 1 추가
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# 예측
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta_best

# 결과 시각화
plt.plot(X, y, "b.")
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.grid()
plt.show()
```

<img width="602" alt="스크린샷 2024-11-25 오후 7 21 35" src="https://github.com/user-attachments/assets/cb1b4369-b243-4c28-b88c-ec42c5f4c0ac">


# 2. 경사 하강법

## 정의
**경사 하강법(Gradient Descent)**은 모델의 손실 함수를 최소화하기 위해 반복적으로 파라미터를 업데이트하는 최적화 알고리즘

### 배치 경사 하강법
- 모든 데이터를 사용하여 손실 함수의 기울기를 계산한 후 파라미터를 업데이트함

### 예제 코드
```python
eta = 0.1  # 학습률
n_epochs = 1000
m = len(X_b)
theta = np.random.randn(2, 1)  # 랜덤 초기화

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print("학습된 파라미터:", theta)
```
### 예상 출력 결과
```
학습된 파라미터: [[4.21509616]
 [2.77011339]]
```

# 3. 로지스틱 회귀

## 정의
**로지스틱 회귀**는 분류 문제를 다루는 모델로, **시그모이드 함수**를 사용하여 입력값을 확률로 변환

---

## 시그모이드 함수

<img width="631" alt="스크린샷 2024-11-25 오후 7 26 53" src="https://github.com/user-attachments/assets/0024cc0c-68ed-4b60-8af9-77cf9310dfac">

---

## 결정 경계
- 로지스틱 회귀는 클래스 간 **결정 경계(Decision Boundary)**를 찾아 데이터를 분류함

---


## 예제 코드
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = load_iris()
X = iris["data"][:, 3:]  # 꽃잎 너비
y = (iris["target"] == 2).astype(int)  # Iris virginica 여부

# 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 결정 경계 시각화
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.axvline(x=decision_boundary, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
```

<img width="601" alt="스크린샷 2024-11-25 오후 7 28 52" src="https://github.com/user-attachments/assets/9f2cd295-ff78-4e1a-9975-f45adcbfba7a">

# 4. 과제: 로지스틱 회귀를 사용한 분류 문제

---

## 목표
- **로지스틱 회귀(Logistic Regression)**를 사용하여 분류 문제를 해결합니다.
- **Iris 데이터셋**을 활용하여 `Iris Virginica` 품종을 분류하는 모델을 구현합니다.

---

## 단계별 과제

### 1. 데이터 준비
1. `sklearn.datasets.load_iris`를 사용하여 데이터를 로드합니다.
2. 데이터의 구조를 이해하기 위해 주요 정보를 출력하세요:
   - 특징(feature)의 이름
   - 타깃(target)의 이름
   - 데이터의 크기
3. `Iris Virginica` 품종 여부를 예측하기 위해 타깃 데이터를 이진화하세요:
   - `target == 2`를 1로, 나머지는 0으로 변환.

### 예상 출력 결과
```
특성 이름: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
타깃 이름: ['setosa' 'versicolor' 'virginica']
데이터 크기: (150, 1)
```

---

### 2. 데이터 분할
1. 데이터를 **훈련 세트(80%)**와 **테스트 세트(20%)**로 분할한 후, 훈련 세트와 테스트 세트의 크기를 출력하세요
2. `train_test_split` 함수를 사용하고, `random_state=42`로 고정하세요.

### 예상 출력 결과
```
훈련 세트 크기: (120, 1)
테스트 세트 크기: (30, 1)
```
---

### 3. 모델 학습 /  예측 및 평가
1. `LogisticRegression` 클래스를 사용하여 모델을 정의하세요.
2. 훈련 데이터를 사용하여 모델을 학습시키세요.
3. 테스트 데이터를 사용하여 예측을 수행하세요.
4. 모델의 **정확도(accuracy)**를 계산하세요.
   - `accuracy_score`를 사용하세요.

### 예상 출력 결과
```
모델 정확도 (Accuracy): 1.0
```
---

### 4. 결정 경계 시각화
1. **꽃잎 너비(petal width)** 값을 기반으로 결정 경계를 시각화하세요.
2. \(X\)의 값을 0부터 3까지 1000개의 점으로 나눠 예측 확률을 계산하세요.
3. 결정 경계를 그래프로 표시하고, 그래프에 다음 정보를 출력하세요:
   - `Iris Virginica` 확률 곡선.
   - 결정 경계(세로선).

### 예상 출력 결과
<img width="619" alt="스크린샷 2024-11-25 오후 7 43 38" src="https://github.com/user-attachments/assets/e32ef1d3-10ac-4f71-98a0-b54bdee3815a">


## 제출 형식
1. **코드**: 위의 단계를 구현한 Python 코드.
2. **결과**:
   - 데이터 크기와 타깃 정보 출력.
   - 정확도(Accuracy)와 확률(Probabilities) 출력.
   - 결정 경계 시각화 그래프.
3. **분석**:
   - 모델의 성능을 평가하고, 결정 경계가 적절히 데이터셋을 분류하는지 간단히 서술하세요.

---
