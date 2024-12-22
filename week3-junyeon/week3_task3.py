# 과제 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 데이터 로드
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# DBSCAN 모델 학습
dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(X)

# 이상치 탐지
outliers = X[clusters == -1]

# 결과 시각화
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers', edgecolor='k')
plt.title('DBSCAN Clustering on Make Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 과제 2
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

# 데이터 로드
data = load_wine()
X = data.data 
feature_names = data.feature_names

# 데이터 스케일링
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# DBSCAN 모델 생성
eps = 4.0  # eps 값 4.0으로 설정
min_samples = 3  # min_samples 3으로 설정
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# 데이터 클러스터링
labels = dbscan.fit_predict(X_scaled)

# 결과 분석
clusters, counts = np.unique(labels, return_counts=True)

outliers = np.sum(labels == -1)  # 이상치 (-1로 표시)
total_samples = len(labels)
outlier_ratio = outliers / total_samples

# 결과 출력
print(f"클러스터 개수: {len(clusters) - 1}") 
print(f"이상치 개수: {outliers}")
print(f"클러스터별 샘플 수:")

cluster_counts = pd.Series(labels).value_counts().sort_index()
print(cluster_counts)