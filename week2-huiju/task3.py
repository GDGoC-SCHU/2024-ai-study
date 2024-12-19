# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# 1. 데이터 준비
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 결정 트리 하이퍼파라미터 최적화
param_grid = {
    'max_depth': [3, 5, 6, 7, 8],  # None 제거
    'max_leaf_nodes': [10, 15, 17, 20],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 최적의 하이퍼파라미터로 훈련된 모델의 테스트 정확도 확인
test_accuracy_dt = accuracy_score(y_test, best_model.predict(X_test))
print("Best Parameters:", best_params)
print("Single Tree Accuracy:", test_accuracy_dt)

# 3. 랜덤 포레스트 구현
n_subsets = 100
subtree_accuracies = []

for _ in range(n_subsets):
    # 랜덤하게 샘플링한 서브셋 생성
    X_sub, y_sub = resample(X_train, y_train, n_samples=len(X_train), random_state=_)
    rf_tree = DecisionTreeClassifier(max_depth=best_params['max_depth'], 
                                      max_leaf_nodes=best_params['max_leaf_nodes'], 
                                      min_samples_split=best_params['min_samples_split'])
    rf_tree.fit(X_sub, y_sub)
    subtree_accuracies.append(accuracy_score(y_test, rf_tree.predict(X_test)))

# 4. 다수결 앙상블
ensemble_predictions = np.zeros((len(X_test), n_subsets))

for i in range(n_subsets):
    X_sub, y_sub = resample(X_train, y_train, n_samples=len(X_train), random_state=i)
    rf_tree = DecisionTreeClassifier(max_depth=best_params['max_depth'], 
                                      max_leaf_nodes=best_params['max_leaf_nodes'], 
                                      min_samples_split=best_params['min_samples_split'])
    rf_tree.fit(X_sub, y_sub)
    ensemble_predictions[:, i] = rf_tree.predict(X_test)

# 다수결 방식으로 최종 예측 생성
final_predictions = np.round(np.mean(ensemble_predictions, axis=1))
ensemble_accuracy = accuracy_score(y_test, final_predictions)

# 결과 출력
print("Average Single Tree Accuracy:", np.mean(subtree_accuracies))
print("Ensemble Accuracy:", ensemble_accuracy)
