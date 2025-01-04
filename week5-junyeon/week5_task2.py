import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 준비
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
y = y.reshape(-1, 1)

# 훈련 데이터(80%) 와 테스트 데이터(20%) 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 입력 데이터를 표준화
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 사용자 정의 층
class MyDenseLayer(Layer):
    def __init__(self, units, **kwargs):
        super(MyDenseLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True, name='weight')
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True, name='bias')

    def call(self, inputs):
        z = tf.matmul(inputs, self.weight) + self.bias
        return tf.nn.relu(z)

# 사용자 정의 손실 함수 (후버 손실)
def huber_loss(y_true, y_pred):
    delta = 1.0
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)

# 모델 설계
model = Sequential([
    MyDenseLayer(32, name='hidden1'),
    MyDenseLayer(32, name='hidden2'),
    tf.keras.layers.Dense(1, name='output')  # 출력층
])

# 모델 컴파일
model.compile(optimizer=Adam(),
              loss=huber_loss,
              metrics=['mse'])

# 모델 훈련
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)

# 모델 평가
test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 데이터에서의 MSE: {test_mse:.4f}")

# 첫 번째 샘플의 예측값과 실제값 출력
first_sample_pred = scaler_y.inverse_transform(model.predict(X_test[:1]))
first_sample_true = scaler_y.inverse_transform(y_test[:1])

print(f"첫 번째 샘플의 예측값: {first_sample_pred[0][0]:.4f}")
print(f"첫 번째 샘플의 실제값: {first_sample_true[0][0]:.4f}")