# 과제 1: 초승달 데이터셋 클러스터링
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 1. 데이터셋 생성
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2. DBSCAN 모델 학습
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 이상치 개수 계산
n_outliers = np.sum(labels == -1)

# 3. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50) # 데이터 산점도: 각 포인트를 클러스터 레이블에 따라 색깔로 구분
plt.colorbar(label='Cluster Label') # 색상 막대 추가헤서 클러스터 레이블과 색상 매핑 표시
plt.title('DBSCAN Clustering on Make Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print("이상치 개수:", n_outliers)

# 과제 2: 실제 데이터셋에서 이상치 탐지
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd

# 1. 데이터셋 로드
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)

# 2. 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN 적용
# eps 값과 min_samples 값을 적절히 설정합니다.
dbscan = DBSCAN(eps=1.5, min_samples=5)  # eps와 min_samples 값을 조정 가능
labels = dbscan.fit_predict(X_scaled)

# 4. 클러스터 및 이상치 개수 계산
labels_series = pd.Series(labels, name="count")
cluster_counts = labels_series.value_counts().sort_index()  # 클러스터별 샘플 수
n_clusters = len(cluster_counts[cluster_counts.index != -1])  # -1(이상치)를 제외한 클러스터 개수
n_outliers = cluster_counts.get(-1, 0)  # 이상치 개수 (-1)

# 5. 결과 출력
print(f"클러스터 개수: {n_clusters}")
print(f"이상치 개수: {n_outliers}")
print("클러스터별 샘플 수:")
print(cluster_counts)
