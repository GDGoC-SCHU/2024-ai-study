# 1주차 과제 개념 정리

- **선형 회귀 (Linear Regression)**:

  - 독립 변수와 종속 변수 간의 선형 관계를 모델링하는 기법
  - 직선 방정식 \( y = wx + b \)로 계산
  - 잠점: 데이터가 선형적 패턴을 따를 때 효과적
  - 단점: 비선형 데이터에 적합하지 않음

- **k-최근접 이웃 회귀 (k-NN)**:

  - 새로운 데이터가 주어지면, 가장 가까운 k개의 데이터의 평균으로 결과를 예측하는 방법으로 구동됨
  - 데이터의 분포를 잘 반영하지만, 데이터가 많아질수록 계산 비용 증가
  - k 값에 따라 모델의 유연성과 안정성이 달라짐

- **장단점 비교**:
  - 선형 회귀: 해석이 쉬움, 데이터가 선형 관계일 때 효과적임
  - k-NN: 비선형 데이터를 처리 가능, 적절한 K값에 대한 주관성

---

- **분류 (Classification)**:

  - 입력 데이터가 주어지면 해당 데이터가 어떤 클래스에 속하는지 예측

- **Logistic Regression**:
  - 각 클래스에 속할 확률을 예측하는 회귀 모델
  - 결정 경계를 사용하여 데이터를 분류(적절한 경계선(margin)을 찾는 문제)
  - 선형 회귀와 마찬가지로 비선형 데이터를 잘 구분하지 못함