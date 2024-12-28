# 텐서플로를 사용한 사용자 정의 모델

### 1. 텐서와 연산
- **텐서(Tensor):** 다차원 배열로, 텐서플로의 기본 데이터 구조입니다.
- 텐서는 넘파이의 배열처럼 연산할 수 있지만 GPU와 TPU를 활용한 고성능 계산을 지원합니다.
- 텐서를 생성하려면 `tf.constant()`를 사용하며, 텐서의 연산은 자동으로 최적화되어 실행됩니다.

#### 예제

```python
import tensorflow as tf

# 텐서 생성
tensor = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print("Tensor:", tensor)

# 텐서 정보 확인
print("Shape:", tensor.shape)  # (2, 3)
print("Data type:", tensor.dtype)  # tf.float32

# 기본 연산
print("Tensor + 10:\n", tensor + 10)  # Broadcasting
print("Squared Tensor:\n", tf.square(tensor))
print("Matrix Multiplication:\n", tensor @ tf.transpose(tensor))  # 행렬 곱
```

### 2. 사용자 정의 손실 함수
- 손실 함수(Loss Function): 모델의 예측값과 실제값 사이의 오차를 계산합니다.
- 후버 손실(Huber Loss)은 MSE와 MAE의 장점을 결합한 손실 함수로, 작은 오차는 MSE를 사용하고 큰 오차는 MAE를 사용합니다.
- 사용자 정의 손실 함수는 일반 파이썬 함수로 작성하거나 keras.losses.Loss를 상속받아 클래스로 구현할 수 있습니다.

#### 예제
```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

# 모델 컴파일에 사용
model.compile(loss=huber_fn, optimizer='nadam')
```
### 3. 사용자 정의 층
- 층(Layer): 모델에서 데이터를 처리하는 기본 단위입니다.
- 사용자 정의 층은 keras.layers.Layer 클래스를 상속받아 구현하며, 초기화, 가중치 생성, 데이터 처리를 직접 정의할 수 있습니다.
- 층은 가중치와 계산을 포함하는 일종의 함수로, 입력 데이터를 처리해 출력을 생성합니다.

#### 예제
```python
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[input_shape[-1], self.units], initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)

# 테스트
layer = MyDense(units=10, activation="relu")
output = layer(tf.constant([[1., 2., 3.]]))
```
### 4. 사용자 정의 모델
- 모델(Model): 여러 층으로 구성된 신경망입니다.
- 사용자 정의 모델은 tf.keras.Model 클래스를 상속받아 구현하며, __init__ 메서드에서 층을 정의하고, call 메서드에서 데이터를 처리하는 논리를 작성합니다.
- 잔차 블록(Residual Block)은 입력값에 특정 연산 결과를 더해 정보를 유지하도록 돕는 구조입니다.

#### 예제
```python
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation='elu') for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # 잔차 연결

class ResidualModel(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(30, activation='elu', kernel_initializer='he_normal')
        self.residual_block = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden(inputs)
        Z = self.residual_block(Z)
        return self.out(Z)

# 모델 테스트
model = ResidualModel(output_dim=1)
output = model(tf.constant([[1., 2., 3.]]))
```

### 5. 자동 미분
- 자동 미분(Automatic Differentiation): 텐서플로는 GradientTape를 사용해 미분값(그레이디언트)을 자동으로 계산합니다.
- 미분값은 손실 함수의 기울기를 계산하고 경사 하강법으로 최적화를 수행하는 데 사용됩니다.
  
#### 예제
```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

### 6. 사용자 정의 훈련 반복
- 텐서플로의 기본 fit 메서드를 사용하지 않고, 사용자 정의 훈련 루프를 작성해 세부 훈련 논리를 구현합니다.
- GradientTape를 사용해 그레이디언트를 계산하고 optimizer.apply_gradients로 매개변수를 갱신합니다.

#### 예제
```python
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

# 훈련 반복
for epoch in range(10):  # 10 epochs
    for step in range(len(X_train) // 32):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_batch, y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
---
## 과제: 사용자 정의 층과 손실 함수를 사용한 신경망 만들기

### 목표
텐서플로와 케라스를 사용하여 사용자 정의 층과 손실 함수를 활용한 신경망을 설계하고 훈련합니다.

### **과제 요구사항**

**1. 데이터 준비**
- 데이터셋: `sklearn.datasets`에서 제공하는 `make_regression` 데이터를 사용합니다.
- 데이터 처리:
  1. **훈련 데이터(80%)** 와 **테스트 데이터(20%)** 로 분리하세요.
  2. 입력 데이터를 **표준화**하여 평균 0, 표준편차 1로 변환하세요.

**2. 사용자 정의 층**
- `MyDenseLayer`라는 사용자 정의 층을 만드세요.
- 요구사항:
  1. 입력 크기에 따라 **가중치(weight)** 와 **편향(bias)** 를 생성하고 학습 가능하도록 설정합니다.
  2. 입력 데이터를 받아 **ReLU 활성화 함수**를 적용합니다.

**3. 사용자 정의 손실 함수**
- **후버 손실(Huber Loss)** 를 구현하세요.
- 요구사항:
  1. 임계값(`delta`)은 1로 설정합니다.
  2. 작은 오차는 **제곱 손실**, 큰 오차는 **선형 손실**로 처리합니다.

**4. 모델 설계 및 훈련**
- 사용자 정의 층과 후버 손실 함수를 사용하여 신경망을 설계합니다.
- 요구사항:
  1. **구조:** 2개의 은닉층(각 32개의 뉴런)과 1개의 출력층.
  2. **Optimizer:** Adam.
  3. **평가지표:** MSE (Mean Squared Error).
  4. **훈련:** 10 epoch, batch size=32.

**5. 평가 및 예측**
- 요구사항:
  1. 테스트 데이터에서 **MSE**를 출력하세요.
  2. 테스트 데이터 중 첫 번째 샘플의 **예측값**과 **실제값**을 출력하세요.
---
### **출력**
1. **모델 훈련 중 손실값 출력:**
2. **테스트 데이터에서의 MSE 출력:**
3. **첫 번째 샘플의 예측값과 실제값 출력:**
