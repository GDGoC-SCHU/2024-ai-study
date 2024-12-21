
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 준비
iris = load_iris()
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)   # 타깃 이진화: Virginica 품종을 1로, 나머지는 0으로 변환

# 데이터 정보 출력
print("특성 이름:", iris.feature_names)
print("타깃 이름:", iris.target_names)
print("데이터 크기:", y.reshape(-1, 1).shape)

# 2. 데이터 분할
# 데이터 분할: 훈련 세트(80%)와 테스트 세트(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 훈련 세트와 테스트 세트 크기 출력
print("훈련 세트 크기:", y_train.reshape(-1, 1).shape)
print("테스트 세트 크기:", y_test.reshape(-1, 1).shape)

# 3. 모델 학습 / 예측 및 평가
# 로지스틱 회귀 모델 정의 및 학습
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

res=log_reg.predict(X_test) # 예측 수행
accuracy=accuracy_score(y_test, res) # 모델의 정확도 계산

print("모델 정확도(Accuracy) ", accuracy) # 결과 출력

# 4. 결정 경계 시각화
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.axvline(x=decision_boundary, color="r", linestyle="--", label="Decision Boundary")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.title("Logistic Regression - Decision Boundary")
plt.legend() # 범례를 추가
plt.grid()
plt.show()
print(f"결정 경계: {decision_boundary}")

# 4. 결정 경계 시각화
x_min, x_max = 0, 3
xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
probabilities = model.predict_proba(np.hstack((np.zeros((1000, 1)), xx)))[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(xx, probabilities, label='Iris Virginica 확률', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='결정 경계')
plt.title('Iris Virginica 확률 곡선 및 결정 경계')
plt.xlabel('꽃잎 너비 (cm)')
plt.ylabel('확률')
plt.legend()
plt.grid()
plt.show()

# 분석 :
# 모델의 성능은 정확도 1.0으로 매우 우수하며, 결정 경계 시각화를 통해 꽃잎 너비에 따라 Iris Virginica 품종과 나머지 품종을 잘 구분하고 있는 것을 확인할 수 있습니다. 