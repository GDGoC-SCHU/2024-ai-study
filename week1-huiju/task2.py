from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1.데이터 탐색
iris = load_iris()

print("데이터셋 샘플:")
print(iris.data[:5])
print("타깃 샘플:")
print(iris.target[:5])
print("특성(shape):", iris.data.shape)
print("타깃(shape):", iris.target.shape)

# 2.데이터 분할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 3. 모델 정의 및 훈련
# Logistic Regression 모델 정의
model = LogisticRegression(max_iter=200)

# 훈련 데이터로 모델 학습
model.fit(X_train, y_train)

# 4. 예측 및 평가
# 테스트 데이터를 사용하여 품종 예측
y_pred = model.predict(X_test)

# 예측 결과 기반으로 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("모델의 정확도: {:.2f}%".format(accuracy * 100))
