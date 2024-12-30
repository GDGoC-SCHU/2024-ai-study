import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화(0~1)
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델 구조 설계
model = Sequential([
    Flatten(input_shape=(28, 28)),  
    Dense(128, activation='relu'), # 활성함수 ReLU인 Dense 층 추가
    Dense(10, activation='softmax') # 활성함수 softmax인 Dense 층 추가(출력층) 
])

# 모델 컴파일
model.compile(
    optimizer='adam', # 옵티마이저 Adam
    loss='categorical_crossentropy', # 손실함수 교차 엔트로피
    metrics=['accuracy'] # 메트릭 accuracy
)

# 모델 훈련
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 성능 평가 및 출력
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"훈련 정확도: {train_acc:.4f}")
print(f"테스트 정확도: {test_acc:.4f}")

# 데이터 예측
predictions = model.predict(x_test)

# 예측 값과 실제 값 출력
first_sample_pred = tf.argmax(predictions[0]).numpy()
first_sample_actual = tf.argmax(y_test[0]).numpy()

print(f"테스트 데이터 샘플 예측 값: {first_sample_pred}")
print(f"테스트 데이터 샘플 실제 값: {first_sample_actual}")