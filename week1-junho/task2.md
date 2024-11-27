# 과제: 2. Iris 데이터셋을 사용하여 꽃의 품종을 예측하는 간단한 분류 모델 만들기

## 1. 데이터 탐색

- sklearn.datasets.load_iris를 사용하여 Iris 데이터셋을 불러오세요.
- 데이터를 간단히 출력하고 특성과 타깃의 형태를 확인하세요.

```
from sklearn.datasets import load_iris

iris = load_iris()

print("데이터셋 샘플:")
print(iris.data[:5], "\n")

print("타깃 샘플:")
print(iris.target[:5], "\n")

print("특성(shape):", iris.data.shape, "\n")
print("타깃(shape):", iris.target.shape)
```

### 출력

```
데이터셋 샘플:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]

타깃 샘플:
[0 0 0 0 0]

특성(shape): (150, 4)

타깃(shape): (150,)
```

## 2. 데이터 분할

- 데이터를 훈련 세트와 테스트 세트로 나누세요. (train_test_split 사용)
- 테스트 데이터 비율은 20%로 설정하고, random_state=42로 고정하세요.

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
```

## 3. 모델 정의 및 훈련

- LogisticRegression 모델을 정의하고 훈련 데이터로 학습시키세요.

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

#모델 정의
model = LogisticRegression()
#trian 데이터를 가지고 학습
model.fit(x_train, y_train)
```

## 4. 예측 및 평가

- 테스트 데이터를 사용하여 품종을 예측하세요.
- 예측 결과를 기반으로 정확도를 출력하세요. (accuracy_score 사용)

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

#모델 정의
model = LogisticRegression()
#trian 데이터를 가지고 학습
model.fit(x_train, y_train)

# test 데이터를 가지고 예측하기
pred = model.predict(x_test)
print("예측 결과:", pred)

#accuracy_score을 가지고 예측에 대한 정확도 출력하기
accuracy  = accuracy_score(y_test, pred)
print("accuracy_score:", accuracy)
```

### 출력

```
예측 결과: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
accuracy_score: 1.0
```
