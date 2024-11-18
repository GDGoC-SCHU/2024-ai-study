# 1장 – 한눈에 보는 머신러닝

## 목차
1. [1. 머신러닝 개요](#1-머신러닝-개요)
2. [2. 데이터 준비](#2-데이터-준비)
3. [3. 선형 회귀 모델](#3-선형-회귀-모델)
4. [4. k-최근접 이웃 회귀](#4-k-최근접-이웃-회귀)
5. [과제](#과제-1-선형-회귀와-k-최근접-이웃을-사용한-삶의-만족도-예측)

---

## 1. 머신러닝 개요
- **정의**: 머신러닝이란 데이터를 사용해 패턴을 학습하고 새로운 데이터에 대한 결과를 예측하는 기술
- **사용 예제**:
  - 가격 예측: 아파트의 크기와 위치로 가격 예측
  - 사용자 만족도: GDP로 국가의 삶의 만족도 예측

---

## 2. 데이터 준비

### 데이터 설명
- 데이터는 **1인당 GDP**(독립 변수)와 **삶의 만족도**(종속 변수)로 구성
- 예제 데이터:
  | GDP per capita (USD) | Life satisfaction |
  |-----------------------|-------------------|
  | 30000                | 5.5               |
  | 40000                | 6.0               |
  | 50000                | 6.5               |
  | 60000                | 7.0               |

### 데이터 시각화
```python
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
data = {"GDP per capita (USD)": [30000, 40000, 50000, 60000],
        "Life satisfaction": [5.5, 6.0, 6.5, 7.0]}
df = pd.DataFrame(data)

# 데이터 시각화
df.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.title("GDP와 삶의 만족도 관계")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.show()
```

## 3. 선형 회귀 모델

### 정의
- **선형 회귀**는 독립 변수와 종속 변수 간의 **직선 관계**를 모델링
- 선형 모델의 수식:
<img width="135" alt="스크린샷 2024-11-18 오후 8 43 37" src="https://github.com/user-attachments/assets/d6178ea5-0190-4c30-a6d0-e7e5072aab69">
<img width="249" alt="스크린샷 2024-11-18 오후 8 46 09" src="https://github.com/user-attachments/assets/c17e8da2-0d34-4ffb-a0ce-82435573db91">

### 코드
```python
from sklearn.linear_model import LinearRegression

# 독립 변수(X)와 종속 변수(y)
X = df[["GDP per capita (USD)"]].values
y = df[["Life satisfaction"]].values

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측
X_new = [[37655]]  # 키프로스의 1인당 GDP
prediction = model.predict(X_new)
print(f"선형 회귀 예측 결과: {prediction[0, 0]:.2f}")
```
### 출력 예시
```
선형 회귀 예측 결과: 6.28
```

## 4. k-최근접 이웃 회귀

### 정의
- **k-최근접 이웃 회귀(k-NN)**는 새로운 데이터 포인트와 가장 가까운 k개의 데이터를 사용하여 예측하는 모델
- 주요 작동 원리:
  1. 새로운 데이터와 기존 데이터 간의 거리를 계산
  2. 가장 가까운 k개의 데이터를 선택
  3. 선택된 데이터들의 평균을 계산하여 예측값으로 사용
     
- k-NN의 특징
장점: 데이터 분포를 잘 반영하며, 복잡한 패턴을 다룰 수 있음.
단점: 데이터 양이 많을수록 계산 비용이 증가하며, 최적의 k 값을 찾는 것이 중요.

### k-NN 회귀 코드

```python
from sklearn.neighbors import KNeighborsRegressor

# k-최근접 이웃 회귀 모델 생성
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# 예측
prediction_knn = model.predict(X_new)
print(f"k-최근접 이웃 예측 결과: {prediction_knn[0, 0]:.2f}")
```
### 출력 예시
```
k-최근접 이웃 예측 결과: 6.33
```
-- 

# 과제: 1. 선형 회귀와 k-최근접 이웃을 사용한 삶의 만족도 예측

## 목표
- **선형 회귀**와 **k-최근접 이웃**을 사용하여 키프로스의 삶의 만족도를 예측하고, 결과를 비교하세요.

---

## 단계별 과제

### 1. 데이터 준비
1. 아래의 데이터를 사용하여 **GDP**와 **삶의 만족도**를 생성하세요.
2. `pandas`를 사용해 데이터를 DataFrame으로 변환하세요.
3. `matplotlib`를 사용해 데이터를 시각화하세요.

| GDP per capita (USD) | Life satisfaction |
|-----------------------|-------------------|
| 30000                | 5.5               |
| 40000                | 6.0               |
| 50000                | 6.5               |
| 60000                | 7.0               |

---

### 2. 모델 학습
1. **선형 회귀 모델**:
   - **`LinearRegression`**을 사용해 모델을 정의하고 학습시키세요.
2. **k-최근접 이웃 모델**:
   - **`KNeighborsRegressor`**를 사용해 \( k=3 \)으로 모델을 정의하고 학습시키세요.

---

### 3. 예측
1. 키프로스의 1인당 GDP\(37655 USD\)를 입력하여 두 모델의 삶의 만족도 예측값을 출력하세요.

---

### 4. 결과 비교
두 모델의 예측 결과를 비교후, 장단점을 간단히 작성하세요.

