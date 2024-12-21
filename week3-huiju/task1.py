import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 데이터셋 로드
file_path = 'car_evaluation.csv'  # 데이터셋 경로를 실제 경로로 업데이트
data = pd.read_csv(file_path)

# 데이터셋의 첫 몇 행 출력
print("데이터셋의 첫 몇 행:")
print(data.head())

# 전처리: LabelEncoder를 사용하여 범주형 데이터를 숫자로 변환
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 특성(X)과 타겟(y)으로 데이터 분리
X = data.iloc[:, :-1]  # 마지막 열 제외한 모든 열
y = data.iloc[:, -1]   # 마지막 열

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 결정 트리 모델 학습
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# 결정 트리 모델 예측 및 정확도 계산
dt_predictions = decision_tree.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"결정 트리 정확도: {dt_accuracy:.2f}")

# 랜덤 포레스트 모델 학습
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# 랜덤 포레스트 모델 예측 및 정확도 계산
rf_predictions = random_forest.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"랜덤 포레스트 정확도: {rf_accuracy:.2f}")

# 정확도 비교 시각화
plt.figure(figsize=(8, 6))
models = ['결정 트리', '랜덤 포레스트']
accuracies = [dt_accuracy, rf_accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('모델 정확도 비교')
plt.ylabel('정확도')
plt.show()

# 혼동 행렬 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay.from_predictions(y_test, dt_predictions, ax=plt.gca())
plt.title('결정 트리 혼동 행렬')
plt.subplot(1, 2, 2)
ConfusionMatrixDisplay.from_predictions(y_test, rf_predictions, ax=plt.gca())
plt.title('랜덤 포레스트 혼동 행렬')
plt.tight_layout()
plt.show()

# 랜덤 포레스트의 특성 중요도 분석
feature_importances = random_forest.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='orange')
plt.title('특성 중요도 (랜덤 포레스트)')
plt.xlabel('중요도')
plt.ylabel('특성')
plt.show()

# 분석 결과
# 랜덤 포레스트는 결정 트리보다 정확도가 높으며, 과적합에 덜 취약합니다. 특성 중요도 분석을 통해 데이터셋에서 중요한 예측 변수를 확인할 수 있습니다.
