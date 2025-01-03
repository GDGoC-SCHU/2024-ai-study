# 인공 신경망 (Artificial Neural Networks, ANN)

## 1. 퍼셉트론 (Perceptron)

### 정의
- 1957년 프랭크 로젠블라트가 제안한 가장 기본적인 인공 신경망 모델.

### 구조
- 입력값(𝑥₁, 𝑥₂, ...)에 가중치(𝑤₁, 𝑤₂, ...)를 곱하고 더한 값(𝑧=∑𝑤𝑖𝑥𝑖)을 기준으로 출력을 생성.
- 결과를 결정하는 **계단 함수(Step Function)**를 사용.

### 한계
- 선형 문제만 해결 가능.
- XOR과 같은 비선형 문제를 해결할 수 없음.



## 2. 다층 퍼셉트론 (MLP): 퍼셉트론의 한계 극복

### 구성
1. **입력층 (Input Layer)**: 데이터 입력.
2. **은닉층 (Hidden Layer)**: 데이터 처리 및 특징 추출.
3. **출력층 (Output Layer)**: 결과 출력.

### 특징
- 여러 은닉층으로 복잡한 비선형 문제도 해결 가능.
- 퍼셉트론과 달리 비선형 활성화 함수(예: ReLU, 시그모이드)를 사용하여 더 복잡한 패턴 학습 가능.


## 3. 역전파 알고리즘 (Backpropagation)

### 정의
- 신경망 훈련을 위한 핵심 알고리즘. 신경망의 출력 오차를 기반으로 가중치를 조정하여 학습을 진행.

### 작동 원리
1. **순전파 (Forward Propagation)**: 입력 데이터가 입력층 → 은닉층 → 출력층으로 전달되며 결과를 계산.
2. **오차 계산**: 예측값과 실제값의 차이를 손실 함수로 측정.
3. **역전파 (Backward Propagation)**: 오차를 출력층에서 입력층으로 전파하며, 각 가중치가 오차에 얼마나 기여했는지 계산.
4. **경사 하강법 (Gradient Descent)**: 가중치를 조정하여 오차를 최소화.

### 효과
- 큰 신경망을 효율적으로 학습 가능.
- 딥러닝 기술의 핵심 기법.

## 4. 신경망 설계의 주요 요소

### 4.1 은닉층의 수
- **이론**: 은닉층이 하나라도 충분히 큰 뉴런 수를 가지면 복잡한 함수도 근사 가능.
- **실제**: 심층 신경망은 파라미터 효율성이 높아 더 적은 뉴런으로 복잡한 문제를 해결.

### 4.2 은닉층의 뉴런 수
- 첫 번째 은닉층에 가장 많은 뉴런을 두고 점차 줄이는 구조가 효과적.
- 과대적합을 방지하기 위해 규제 기법 사용.

### 4.3 학습률 (Learning Rate)
- 가장 중요한 하이퍼파라미터.
- 학습률이 너무 크면 발산, 너무 작으면 학습 속도가 느려짐.
- 최적의 학습률: 손실 그래프에서 손실이 급격히 감소하기 시작하는 지점의 약 10배 낮은 값.

### 4.4 배치 크기
- 작은 배치 크기: 더 좋은 일반화 성능.
- 큰 배치 크기: 훈련 속도 향상.

## 5. 케라스를 활용한 신경망

### 5.1 정의
- **케라스(Keras)**는 텐서플로(TensorFlow)의 고수준 딥러닝 API로, 신경망 설계를 쉽게 도와주는 도구입니다.
- 간단한 사용법과 강력한 기능을 제공하여 초보자와 전문가 모두에게 적합합니다.
- 주요 특징:
  - **모듈화**: 층(Layer), 손실 함수(Loss), 옵티마이저(Optimizer) 등 독립적인 구성 요소를 조합하여 사용.
  - **확장성**: 맞춤형 층과 알고리즘 설계 가능.
  - **유연성**: 다양한 백엔드 엔진(TensorFlow, Theano, CNTK 등) 지원.

---

### 5.2 케라스의 주요 구성 요소

### 1. 모델 정의 방식
케라스는 신경망 모델을 정의하기 위한 세 가지 주요 API를 제공합니다:
1. **시퀀셜 API**:
   - 간단한 모델을 설계할 때 사용.
   - 층을 순서대로 쌓는 방식.
2. **함수형 API**:
   - 다중 입력/출력, 복잡한 신경망 구조 설계에 적합.
   - 네트워크의 유연한 구성 가능.
3. **서브클래싱 API**:
   - 동적인 모델 설계를 위한 명령형 프로그래밍 방식.
   - 완전한 유연성과 제어를 제공.

### 2. 주요 층 (Layers)
- **Dense**: 모든 뉴런이 이전 층의 모든 뉴런과 연결된 완전 연결 층.
- **Conv2D**: 이미지 데이터 처리를 위한 2D 합성곱 층.
- **LSTM**: 순환 신경망 층으로 시계열 데이터를 처리.
- **Dropout**: 과대적합 방지를 위해 일부 뉴런을 무작위로 비활성화.


### 5.3 모델 설계와 훈련

### 1. 데이터 준비
- 데이터를 전처리하여 모델에 적합한 형태로 변환.
- 일반적으로 특성 스케일링, 정규화를 수행.

### 2. 모델 컴파일
- 모델 학습을 위해 손실 함수, 옵티마이저, 평가지표 설정.
- 손실 함수(Loss): 모델의 학습 목표를 정의.
- 옵티마이저(Optimizer): 가중치 조정을 위한 알고리즘.
- 평가지표(Metrics): 학습 성능 평가 기준.

### 3. 모델 훈련
- 주어진 데이터셋을 사용하여 모델의 가중치를 조정.
- 검증 세트를 사용해 과대적합 여부를 모니터링.

### 4. 모델 평가와 예측
- 테스트 데이터를 통해 모델의 성능 확인.
- 학습된 모델을 사용하여 새로운 데이터에 대한 예측 생성.


### 5.4 케라스의 장점
1. **사용 용이성**: 간단한 API로 신경망 설계와 훈련 가능.
2. **확장성**: 사용자 정의 구성 요소와 복잡한 모델 설계 지원.
3. **유연성**: 다양한 백엔드 엔진과 하드웨어 가속기(GPU/TPU) 활용 가능.
4. **커뮤니티 지원**: 풍부한 예제와 라이브러리 제공.


## 6. 하이퍼파라미터 튜닝

### 정의
- 신경망의 성능은 층의 개수, 뉴런 수, 학습률, 배치 크기 등 하이퍼파라미터 설정에 따라 달라짐.

### 튜닝 전략
1. **랜덤 서치**: 하이퍼파라미터 값을 무작위로 탐색.
2. **그리드 서치**: 모든 조합을 체계적으로 탐색.
3. **케라스 튜너**: 다양한 하이퍼파라미터 튜닝 전략을 제공하는 라이브러리.


# 과제: 손글씨 숫자 분류기 만들기

## 과제 목표
MNIST 데이터셋을 사용하여 간단한 신경망(MLP)을 설계하고 손글씨 숫자를 분류하세요.

---

## 조건 및 요구사항

### 1. 데이터셋
- **데이터**: MNIST 데이터셋 (케라스에서 제공).
- **데이터 분할**: 훈련 데이터와 테스트 데이터로 나누어 사용.
- **정규화**: 입력 데이터를 0~1 범위로 정규화.

---

### 2. 모델 구조
- **신경망 구조**:
  - 은닉층: 1개.
  - 은닉층의 뉴런 수: `128`.
  - 은닉층의 활성화 함수: `ReLU`.
  - 출력층: 10개의 뉴런과 `softmax` 활성화 함수.
- **모델 구성**: `tensorflow.keras.Sequential`을 사용하여 구성.

---

### 3. 모델 컴파일
- **손실 함수**: `categorical_crossentropy`.
- **옵티마이저**: `adam`.
- **평가지표**: `accuracy`.

---

### 4. 훈련
- **에포크 수**: `5`.
- **배치 크기**: `32`.

---

### 5. 성능 평가
- **훈련 후**: 테스트 데이터에서 모델의 정확도를 출력하세요.

---

## 출력 결과 예시
1. **정확도 출력**:
   - 훈련 정확도와 테스트 정확도를 출력하세요.
2. **새로운 데이터 예측**:
   - 테스트 데이터 첫 번째 샘플에 대한 모델의 예측 값과 실제 값을 출력하세요.
