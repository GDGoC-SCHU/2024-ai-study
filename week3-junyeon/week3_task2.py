import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
data = pd.read_csv('car_evaluation.csv')

# 범주형 데이터를 숫자로 변환
label_encoders = {} 
for column in data.columns:
    if data[column].dtype == 'object': 
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column]) 
        label_encoders[column] = le 

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 데이터 정규화
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# PCA 적용
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_scaled) 

# PCA 결과 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2') 
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# PCA 적용 전후 랜덤포레스트 모델 성능 비교
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 (PCA 전)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred) # 예측 결과와 실제값 비교해 정확도 계산

# 랜덤포레스트 모델 (PCA 후)
rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# 결과 비교(정확도)
print(f"Accuracy before PCA: {accuracy_original:.2f}")
print(f"Accuracy after PCA: {accuracy_pca:.2f}")

# 모델 성능 비교 시각화
plt.figure(figsize=(6, 4))
plt.bar(['Before PCA', 'After PCA'], [accuracy_original, accuracy_pca], color=['blue', 'green'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# 주성분 기여도 분석
explained_variance_ratio = pca.explained_variance_ratio_ 

plt.figure(figsize=(6, 4))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='orange')
plt.title("Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.show()