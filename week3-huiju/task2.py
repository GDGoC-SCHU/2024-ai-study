import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터셋 불러오기
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

# PCA 적용을 위한 데이터 정규화
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# PCA를 사용하여 차원을 2와 3으로 축소
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_normalized)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_normalized)

# 2D에서 PCA 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title('2D PCA 시각화')
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.colorbar(label='타겟')
plt.show()

# 3D에서 PCA 결과 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', s=50, alpha=0.7)
ax.set_title('3D PCA 시각화')
ax.set_xlabel('주성분 1')
ax.set_ylabel('주성분 2')
ax.set_zlabel('주성분 3')
plt.colorbar(sc, label='타겟')
plt.show()

# PCA를 적용하지 않은 Random Forest 모델 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
rf_no_pca_predictions = random_forest.predict(X_test)
rf_no_pca_accuracy = accuracy_score(y_test, rf_no_pca_predictions)
print(f"PCA를 적용하지 않은 Random Forest 정확도: {rf_no_pca_accuracy:.2f}")

# PCA(2D)를 적용한 Random Forest 모델 학습 및 평가
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca_2d, y, test_size=0.3, random_state=42)
random_forest_pca = RandomForestClassifier(random_state=42)
random_forest_pca.fit(X_pca_train, y_pca_train)
rf_pca_predictions = random_forest_pca.predict(X_pca_test)
rf_pca_accuracy = accuracy_score(y_pca_test, rf_pca_predictions)
print(f"PCA를 적용한 Random Forest 정확도: {rf_pca_accuracy:.2f}")

# 정확도 비교 시각화
plt.figure(figsize=(8, 6))
models = ['PCA 미적용', 'PCA 적용']
accuracies = [rf_no_pca_accuracy, rf_pca_accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Random Forest 정확도 비교')
plt.ylabel('정확도')
plt.show()

# 주성분 기여도 분석
explained_variance_ratio = pca_2d.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='orange', tick_label=[f'PC{i}' for i in range(1, len(explained_variance_ratio) + 1)])
plt.title('주성분별 설명 분산 비율')
plt.xlabel('주성분')
plt.ylabel('설명 분산 비율')
plt.show()

# 요약 분석
# PCA는 데이터의 차원을 축소하면서 중요한 분산을 유지합니다. PCA 적용 시 Random Forest 성능이 약간 감소할 수 있으나 시각화와 계산 효율성이 향상됩니다.
