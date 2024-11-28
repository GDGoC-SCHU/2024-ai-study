from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1.데이터 탐색
iris = load_iris()

# 데이터 및 타깃 확인
print("데이터셋 샘플:\n", iris.data[:5])
print("타깃 샘플:\n", iris.target[:5])
print("특성(shape): ", iris.data.shape)
print("타깃(shape): ", iris.target.shape)

# 2.데이터 분할
# 훈련 데이터와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

print("훈련 데이터 크기(shape): ", X_train.shape)
print("테스트 데이터 크기(shape): ", X_test.shape)

# 3. 모델 정의 및 훈련
# Logistic Regression 모델 정의 및 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. 예측 및 평가
# 테스트 데이터 사용하여 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression 모델 정확성: ", accuracy)