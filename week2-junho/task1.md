# 과제: 로지스틱 회귀를 사용한 분류 문제

## 목표

- 로지스틱 회귀(Logistic Regression)를 사용하여 분류 문제를 해결합니다.
- Iris 데이터셋을 활용하여 Iris Virginica 품종을 분류하는 모델을 구현합니다.

## 1. 데이터 준비

### 1. sklearn.datasets.load_iris를 사용하여 데이터를 로드합니다.

```
from sklearn.datasets import load_iris

# 데이터 로드 및 준비
iris = load_iris()
```

### 2. 데이터의 구조를 이해하기 위해 주요 정보를 출력하세요:

- 특징(feature)의 이름
- 타깃(target)의 이름
- 데이터의 크기

```
from sklearn.datasets import load_iris

# 데이터 로드 및 준비
iris = load_iris()

print('특징 이름: ', iris.feature_names)
print('타깃 이름: ', iris.target_names)
print('데이터 크기: ',iris.data.shape)
```

### 3. Iris Virginica 품종 여부를 예측하기 위해 타깃 데이터를 이진화하세요:

#### target == 2를 1로, 나머지는 0으로 변환.

```
# 타깃 데이터 이진화
binary_target = (iris.target == 2).astype(int)

# 결과 확인
print(binary_target)
```

### 2. 데이터 분할

#### 데이터를 **훈련 세트(80%)**와 **테스트 세트(20%)**로 분할한 후, 훈련 세트와 테스트 세트의 크기를 출력하세요. train_test_split 함수를 사용하고, random_state=42로 고정하세요.

```
from sklearn.model_selection import train_test_split

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(iris.data, binary_target, test_size=0.2, random_state=42)


print(len(x_train))
print(len(x_test))
```

### 3. 모델 학습 / 예측 및 평가

#### LogisticRegression 클래스를 사용하여 모델을 정의하세요.

```
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
```

#### 훈련 데이터를 사용하여 모델을 학습시키세요.

```
log_reg.fit(x_train, y_train)
```

#### 테스트 데이터를 사용하여 예측을 수행하세요.

```
log_reg.predict(x_test)
```

#### 모델의 **정확도(accuracy)**를 계산하세요. accuracy_score를 사용하세요.

```
y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("모델 정확도 (Accuracy):", accuracy)
```

### 4. 결정 경계 시각화

#### 꽃잎 너비(petal width) 값을 기반으로 결정 경계를 시각화하세요.

#### (X)의 값을 0부터 3까지 1000개의 점으로 나눠 예측 확률을 계산하세요.

#### 결정 경계를 그래프로 표시하고, 그래프에 다음 정보를 출력하세요:

- Iris Virginica 확률 곡선.
- 결정 경계(세로선).

처음엔 다음과 같이 작성 함.

```
# 결정 경계 시각화 (Petal width만 사용)
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
x_new = np.hstack([x_new, np.zeros((1000, 3))])  # 나머지 3개의 feature를 0으로 채움
y_proba = log_reg.predict_proba(x_new)[:, 1]

plt.plot(x_new[:, 0], y_proba, "g-", label="Iris virginica")
plt.axhline(y=0.5, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
```

하지만, 나머지 값을 0으로 채울 경우 정확한 판단이 되지 않음.
그렇다고 실제 값을 넣으면 꽃잎 너비의 값에 따른 그래프가 아닌 결과가 나올것임.
따라서 처음에 데이터 부분 부터 전부 수정하였음.

```
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = load_iris()
x = iris.data[:, 3].reshape(-1, 1)  # 꽃잎 너비

# 타깃 데이터 이진화
binary_target = (iris.target == 2).astype(int)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, binary_target, test_size=0.2, random_state=42)

# 모델 학습
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

# 모델 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("모델 정확도 (Accuracy):", accuracy)

# 결정 경계 시각화 (Petal width만 사용)
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)[:, 1]

plt.plot(x_new[:, 0], y_proba, "g-", label="Iris virginica")
plt.axhline(y=0.5, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
```

### 최종 코드

```
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 및 준비
iris = load_iris()
x = iris.data[:, 3].reshape(-1, 1)  # 꽃잎 너비

print('특징 이름: ', iris.feature_names)
print('타깃 이름: ', iris.target_names)
print('데이터 크기: ',iris.data.shape)

# 타깃 데이터 이진화
binary_target = (iris.target == 2).astype(int)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, binary_target, test_size=0.2, random_state=42)

# 모델 학습
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

# 모델 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("모델 정확도 (Accuracy):", accuracy)

# 모델 확률 출력
probabilities_test = log_reg.predict_proba(x_test)[:, 1]
print("모델 확률:", probabilities_test)

# 결정 경계 시각화 (Petal width만 사용)
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)[:, 1]

plt.plot(x_new[:, 0], y_proba, "g-", label="Iris virginica")
plt.axhline(y=0.5, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
```

### 분석

로지스틱 회귀는 마지막 결과가 확률적으로 나오게 된다. <br/>
이때의 확률에서 특정 값을 기준으로 분류를 하게 된다. <br/>
따라서 참, 거짓과 같은 분류에선 효과적으로 작동하지만 강아지, 고양이, 사람 분류 문제에선 잘 작동할지 의문,<br/>
그럼에도 2진 분류에 대해서는 좋은 성능을 보여주는 것으로 보임
