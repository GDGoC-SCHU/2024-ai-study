# 과제 2. Iris 데이터셋을 사용하여 꽃의 품종을 예측하는 간단한 분류 모델 만들기

# 라이브러리 가져오기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

##############################################################################################

# 1. 데이터 탐색
iris = load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

print(iris_df.head())
print("data shape:", iris.data.shape)
print(iris_df.info()) # 데이터 타입 및 결측치 : 결측치 없음
print(iris_df.describe()) # 데이터 통계
print(iris_df['target'].value_counts()) # Target Value 분포: Imbalanced 하지 않음

##############################################################################################

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

##############################################################################################

# 3. 모델 정의 및 훈련
logistic_model = LogisticRegression(max_iter=200, random_state=42)
logistic_model.fit(X_train, y_train)

##############################################################################################

# 4. 예측 및 평가
y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Model Accuracy: {:.2f}%".format(accuracy * 100)) # 정확도 100%