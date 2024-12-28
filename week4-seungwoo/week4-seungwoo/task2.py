import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# 1. 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화: 0-255 범위의 픽셀 값을 0-1 범위로 변환
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1D 벡터로 변환
    Dense(128, activation='relu'),  # 은닉층: 128개의 뉴런, ReLU 활성화 함수
    Dense(10, activation='softmax')  # 출력층: 10개의 뉴런, softmax 활성화 함수
])

# 3. 모델 컴파일
model.compile(
    optimizer=Adam(),  # Adam 옵티마이저
    loss='sparse_categorical_crossentropy',  # 손실 함수: sparse_categorical_crossentropy (정수 레이블)
    metrics=['accuracy']  # 평가지표: accuracy
)

# 4. 모델 훈련
history = model.fit(
    x_train, y_train,
    epochs=5,  # 에포크 수: 5
    batch_size=32,  # 배치 크기: 32
    validation_data=(x_test, y_test)  # 검증 데이터로 테스트 데이터를 사용
)

# 5. 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_acc:.4f}")

# 6. 예측: 테스트 데이터의 첫 번째 샘플 예측
import numpy as np

predictions = model.predict(x_test[:1])  # 첫 번째 샘플에 대해 예측

predicted_label = np.argmax(predictions)  # 예측된 레이블
actual_label = y_test[0]  # 실제 레이블

print(f"Predicted label: {predicted_label}")
print(f"Actual label: {actual_label}")