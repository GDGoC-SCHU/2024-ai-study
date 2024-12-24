# 과제

## 목표

- 결정 트리를 사용해 moons 데이터셋 분류 문제를 해결하고, 랜덤 포레스트로 정확도를 향상시켜 보세요.

### 1. 데이터 준비

#### make_moons(n_samples=10000, noise=0.4)를 사용해 데이터를 생성하세요.

```
from sklearn.datasets import make_moons

# 데이터 생성
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
```

#### 데이터를 훈련 세트(80%)와 테스트 세트(20%)로 나누세요(train_test_split 사용).

```
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. 결정 트리 하이퍼파라미터 최적화

#### GridSearchCV를 사용해 결정 트리의 max_depth, max_leaf_nodes 등 최적의 하이퍼파라미터를 찾으세요.

```
# 결정 트리 모델 생성
dt_clf = DecisionTreeClassifier(random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20, 30, 40, 50]
}

# GridSearchCV 설정
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
```

#### 최적의 하이퍼파라미터로 훈련된 모델의 테스트 정확도를 확인하세요.

```
# 모델 학습
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best hyperparameters:", grid_search.best_params_)

# 최적의 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### 3. 랜덤 포레스트 구현

#### 훈련 세트를 랜덤하게 샘플링한 서브셋을 생성하세요(100개의 서브셋).

#### 각 서브셋에 대해 결정 트리를 훈련시키고 정확도를 확인하세요.

```
# 랜덤 포레스트 파라미터
n_trees = 100
n_samples = len(X_train)

# 서브셋 생성 및 결정 트리 훈련
accuracy_list = []
for _ in range(n_trees):
    # 랜덤 샘플링
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_subset, y_subset = X_train[indices], y_train[indices]

    # 결정 트리 훈련
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_subset, y_subset)

    # 예측 및 정확도 계산
    y_pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# 평균 정확도 출력
mean_accuracy = np.mean(accuracy_list)
print("Mean accuracy:", mean_accuracy)
```

### 4. 다수결 앙상블

테스트 데이터에 대해 100개의 결정 트리의 예측값을 모으세요.
다수결 방식으로 최종 예측을 생성하고 정확도를 측정하세요.

```
predictions = np.zeros((n_trees, len(X_test)), dtype=int)
...
    # 예측 저장
    predictions[i] = y_pred
...
# 다수결 앙상블
final_predictions = np.squeeze(mode(predictions, axis=0).mode)

# 정확도 계산
accuracy = accuracy_score(y_test, final_predictions)
print("Ensemble accuracy:", accuracy)
```

### 최종 코드

```
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# 데이터 생성
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 생성
dt_clf = DecisionTreeClassifier(random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20, 30, 40, 50]
}

# GridSearchCV 설정
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best hyperparameters:", grid_search.best_params_)

# 최적의 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 랜덤 포레스트 파라미터
n_trees = 100
n_samples = len(X_train)

# 서브셋 생성 및 결정 트리 훈련
predictions = np.zeros((n_trees, len(X_test)), dtype=int)
accuracy_list = []
for i in range(n_trees):
    # 랜덤 샘플링
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_subset, y_subset = X_train[indices], y_train[indices]

    # 결정 트리 훈련
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_subset, y_subset)

    # 예측 및 정확도 계산
    y_pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

    # 예측 저장
    predictions[i] = y_pred

# 평균 정확도 출력
mean_accuracy = np.mean(accuracy_list)
print("Mean accuracy:", mean_accuracy)

# 다수결 앙상블
final_predictions = np.squeeze(mode(predictions, axis=0).mode)

# 정확도 계산
accuracy = accuracy_score(y_test, final_predictions)
print("Ensemble accuracy:", accuracy)
```
