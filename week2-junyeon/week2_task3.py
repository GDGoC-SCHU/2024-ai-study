import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 생성
x, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# GridSearchCV 설정
param_grid = {
    "max_depth": [3, 5, 6, 10, None],
    "max_leaf_nodes": [10, 15, 17, 20, 50, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(x_train, y_train)

# 최적의 하이퍼파라미터 탐색
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 최적 하이퍼파라미터로 모델 학습 및 테스트 정확도
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Single Tree Accuracy:", round(test_accuracy, 4))

# 랜덤 포레스트 구현
n_trees = 100
# 서브셋 생성
n_samples_per_tree = len(x_train) // 10  
subsets = [ np.random.choice(len(x_train), n_samples_per_tree, replace=True) for _ in range(n_trees)]

trees = []
tree_accuracies = []
for subset in subsets:
    tree = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        max_leaf_nodes=best_params["max_leaf_nodes"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42
    )
    tree.fit(x_train[subset], y_train[subset])
    trees.append(tree)
    tree_accuracy = accuracy_score(y_train[subset], tree.predict(x_train[subset]))
    tree_accuracies.append(tree_accuracy)

print("Average Single Tree Accuracy:", round(np.mean(tree_accuracies), 4))

# 테스트 데이터에 대한 각 트리의 예측값 모으기
all_predictions = np.array([tree.predict(x_test) for tree in trees])

# 다수결 방식으로 최종 예측 생성
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)

# 최종 앙상블 정확도 확인
ensemble_accuracy = accuracy_score(y_test, final_predictions)
print("Ensemble Accuracy:", round(ensemble_accuracy, 4))