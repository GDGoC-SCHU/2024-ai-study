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

