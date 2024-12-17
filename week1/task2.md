# 과제2: Iris 데이터셋을 사용하여 꽃의 품종을 예측하는 간단한 분류 모델 만들기

## 목차
1. [1. 분류(Classification)](#1-분류classification)
2. [2. Iris 데이터셋](#2-iris-데이터셋)
3. [3. Logistic Regression](#3-logistic-regression)
4. [4. 정확도(Accuracy)](#4-정확도accuracy)
5. [과제](#과제-iris-데이터셋을-사용하여-꽃의-품종을-예측하는-간단한-분류-모델-만들기)


## 1. 분류(Classification)
- **정의**: 머신러닝에서 분류는 입력 데이터의 레이블(클래스)을 예측하는 작업입니다.
  - 예: 이메일이 스팸인지 아닌지를 예측하거나, 꽃의 품종을 분류하는 문제.
- **사용 예**:
  - 스팸 메일 필터링
  - 질병 진단
  - 이미지에서 객체 분류
- **알고리즘**:
  - Logistic Regression, Decision Tree, SVM, Random Forest 등.

## 2. Iris 데이터셋

### Iris 데이터셋이란?
- 머신러닝의 기본 예제로 사용되는 데이터셋으로, 세 가지 종류의 꽃(Setosa, Versicolor, Virginica)에 대한 정보를 담고 있습니다.
- 데이터의 각 샘플은 4개의 특성(feature)을 포함합니다:
  - 꽃받침 길이(Sepal Length)
  - 꽃받침 너비(Sepal Width)
  - 꽃잎 길이(Petal Length)
  - 꽃잎 너비(Petal Width)

### 데이터 구성:
- 총 150개의 샘플로 이루어져 있으며, 각 클래스에 50개씩의 샘플이 포함되어 있습니다.

## 3. Logistic Regression
- **정의**:  
  Logistic Regression은 이름과는 달리 회귀가 아닌 **분류 알고리즘**입니다.
  - 입력 데이터를 분석하여 특정 클래스에 속할 확률을 예측합니다.

- **작동 원리**:  
  데이터를 선형적으로 분리할 수 있는 초평면(Hyperplane)을 찾고, 그 결과를 확률로 변환하여 예측합니다.

## 4. 정확도(Accuracy)
- **정의**:  
  모델이 예측한 값 중에서 실제 레이블과 일치하는 예측의 비율입니다.

- **공식**:  
  \[
  \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
  \]

- **주의**:  
  데이터가 불균형한 경우(예: 한 클래스가 90% 이상을 차지하는 경우) 높은 정확도가 반드시 좋은 모델을 의미하지 않을 수 있습니다.

</br>
</br>
</br>

## 과제: Iris 데이터셋을 사용하여 꽃의 품종을 예측하는 간단한 분류 모델 만들기

## 목표
주어진 데이터에서 꽃의 특성(길이와 너비)을 바탕으로 꽃 품종을 예측하는 모델을 만들어보세요.


### 1. 데이터 탐색
- `sklearn.datasets.load_iris`를 사용하여 Iris 데이터셋을 불러오세요.
- 데이터를 간단히 출력하고 특성과 타깃의 형태를 확인하세요.


### 2. 데이터 분할
- 데이터를 훈련 세트와 테스트 세트로 나누세요. (`train_test_split` 사용)
- 테스트 데이터 비율은 20%로 설정하고, `random_state=42`로 고정하세요.


### 3. 모델 정의 및 훈련
- `LogisticRegression` 모델을 정의하고 훈련 데이터로 학습시키세요.


### 4. 예측 및 평가
- 테스트 데이터를 사용하여 품종을 예측하세요.
- 예측 결과를 기반으로 정확도를 출력하세요. (`accuracy_score` 사용)


### 참고
- **Documentation**:
  - [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - [Scikit-learn Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- **필요한 라이브러리**:
  - `sklearn.datasets`, `sklearn.model_selection`, `sklearn.linear_model`, `sklearn.metrics`


**출력 예시**:
```plaintext
데이터셋 샘플:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
타깃 샘플:
[0 0 0 0 0]
특성(shape): (150, 4)
타깃(shape): (150,)