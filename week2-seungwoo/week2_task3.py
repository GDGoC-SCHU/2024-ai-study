from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# make_moons 데이터셋 생성
x, y = make_moons(n_samples=1000, noise=0.4)

# 데이터 가시성을 위해서 시각화
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=100, edgecolor="k", linewidth=2)
plt.show()

# 데이터셋을 훈련 데이터와 테스트 데이터로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)  # 입력 데이터 크기
print(y_train.shape)

# RandomizedSearchCV를 사용하여 하이퍼파라미터 탐색
param_RF = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    
}


from sklearn.metrics import accuracy_score
# RandomForestClassifier 모델 생성
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV 설정
rscv = RandomizedSearchCV(estimator=rf, param_distributions=param_RF, n_iter=10, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
rscv.fit(x_train, y_train)

# 최적 모델 출력
best_model = rscv.best_estimator_

# 테스트 데이터에 대한 예측
y_pred = best_model.predict(x_test)

# 최적 파라미터와 정확도 출력
print("Best Parameters:", rscv.best_params_)
print(f'Best Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')



