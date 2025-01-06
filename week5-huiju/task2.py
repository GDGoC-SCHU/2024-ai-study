import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
# 데이터 생성
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
X_test = tf.constant(X_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)

# 2. 사용자 정의 층
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# 3. 사용자 정의 손실 함수
def huber_loss(y_true, y_pred):
    delta = 1.0
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * (tf.abs(error) - (delta / 2))
    return tf.where(is_small_error, squared_loss, linear_loss)

# 4. 모델 설계 및 훈련
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = MyDenseLayer(32)
        self.dense2 = MyDenseLayer(32)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# 모델 초기화
model = MyModel()

# 모델 컴파일
model.compile(optimizer='adam', loss=huber_loss, metrics=['mse'])

# 모델 훈련
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 5. 평가 및 예측
# 테스트 데이터에서 MSE 출력
test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print("테스트 데이터에서의 MSE:", test_mse)

# 첫 번째 샘플의 예측값과 실제값 출력
predictions = model.predict(X_test)
print("첫 번째 샘플의 예측값:", predictions[0][0])
print("첫 번째 샘플의 실제값:", y_test[0].numpy())
