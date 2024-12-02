# 1장 – 한눈에 보는 머신러닝

## 목차
1. [1. 머신러닝 개요](#1-머신러닝-개요)
2. [2. 데이터 준비](#2-데이터-준비)
3. [3. 선형 회귀 모델](#3-선형-회귀-모델)
4. [4. k-최근접 이웃 회귀](#4-k-최근접-이웃-회귀)
5. [5. 의료분야 적용 및 느낀점](#5-의료분야-적용-및-느낀점)

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
<img width="574" alt="스크린샷 2024-11-18 오후 9 11 31" src="https://github.com/user-attachments/assets/b78c0d0d-3fbb-4f40-97cb-a1d66c3ba91a">

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
선형 회귀 예측 결과: 5.88
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
k-최근접 이웃 예측 결과: 6.00
```
-- 

## 5. 의료분야 적용 및 느낀점

### 1) 의료분야에서 인공지능(IBM)
- 머신러닝모델을 사용해 의료데이터를 처리하고 의료전문가에게 인사이트 제공해 의료서비스 질 향상

# 활용

# 임상 의사 결정 보조
- 의료서비스 제공자가 환자정보나 연구에 빠른 접근
- 치료, 약물, 정신건강 및 기타 환자 요구 사항 결정

# 의료영상 분석
- CT스캔, 엑스레이, MRI 및 기타 이미지
- 방사선 전문의가 놓칠 수 있는 병변이나 기타 소견 발견
- 예: COVId-19 환자 선별 AI 기반 도구


### 2) 김경 외 2인, 선형회귀분석을 이용한 생체신호 데이터 기반의 건강관리, 2019.

# 목적
- 스마트기기로부터 측정된 생체정보 기반으로 선형회귀분석 활용해 비만 지표를 대상으로 건강 예측

# 방법
- 피험자 대상 75일 동안 개인 운동 수행 시 건강정보 파라미터 변화 양상 분석
- 피험자의 개인 운동 정보: 스마트 시계 활용
- 비만 관련 파라미터(체중, 체지방, 근육정보): 스마트 체중계 활용

# 결론
- 개인 운동 정보와 체중 관련 파라미터 상관관계분석을 통해 상관성 확인하고 선형회귀분석
- 체지방: 감소하는 경향
- 근육: 증가하는 경향
    - 건강정보가 사용자에게 시각적 피드백 방법으로 전달되어 개인의 건강관리 참여 독려 기대


### 3) k-NN 의료산업에 적용(IBM)
# 활용
- 심장마비와 전립선 암 등의 위험도 예측
- 알고리즘: 가장 가능성이 높은 유전자 발현 계산
      

### 4) 느낀점
- 추후 보건역학 대학원에 석박통합으로 진학 예정
- 의료분야 적용의 무한한 가능성 존재의 인식
- 막학기지만 GDGoC SCHU를 통해 AI(Data Analytics) 관련 지식과 경험을 더 많이 쌓고 싶어짐
    
