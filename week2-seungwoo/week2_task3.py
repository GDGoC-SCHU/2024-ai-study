from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score

x, y = make_moons(n_samples= 1000, noise= 0.4)


#데이터 가시성을 위해서 시각화
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=100,
            edgecolor="k", linewidth=2)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)  # 입력 데이터 크기
print(y_train.shape)

#GridSearchCV
# 사용자가 직접 모델의 하이퍼 파라미터의 값을 리스트로 입력하면 
# 값에 대한 경우의 수마다 예측 성능을 측정 평가하여 비교하면서 최적의 하이퍼 파라미터 값을 찾는 과정을 진행

param_RF = {
   
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [10, 17, 20],
    'max_features': ['sqrt', 'log2', None],
    
}

from sklearn.metrics import r2_score

rf = RandomForestRegressor()
gscv = GridSearchCV(estimator=rf, param_grid= param_RF, cv = 5, n_jobs = -1, scoring = 'r2')
gscv.fit(x_train, y_train)

best_model = gscv.best_estimator_

y_pred = best_model.predict(x_test)
best_score = r2_score(y_test, y_pred)

print("Best Parameters:", gscv.best_params_)
print(f'Single Tree Accuracy: {gscv.score(x_test, y_test):.4f}')

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
n_subset = 100
subset_size = len(x_train)

test_score = []
all_predict = []

for i in range(n_subset) :

    x_subset, y_subset = resample(x_train, y_train, n_samples=subset_size, random_state=i)

    tree = DecisionTreeRegressor(random_state=42,  **gscv.best_params_)
    tree.fit(x_subset, y_subset)
    

    y_test_pred = tree.predict(x_test)

    
    test_score.append(r2_score(y_test, y_test_pred))

    all_predict.append(y_test_pred)

print(f'Average Single Tree Accuracy: {np.mean(test_score):.4f}')

all_predict = np.array(all_predict)  # 100 x 200 형태의 배열로 변환
final_pred = np.mean(all_predict, axis=0)  # 다수결 방식으로 최종 예측값 계산

# 정확도 측정
accuracy = r2_score(y_test, final_pred)
print(f"Ensemble Accuracy: {accuracy:.4f}")

