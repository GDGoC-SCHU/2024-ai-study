# 필요한 라이브러리 가져오기import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터 불러오기 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화 (픽셀 값을 0~1 범위로 변환)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 레이블 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 입력층: Flatten 처리
    Dense(128, activation='relu'),  # 은닉층: 128개 뉴런, ReLU 활성화 함수
    Dense(10, activation='softmax')  # 출력층: 10개 뉴런, Softmax 활성화 함수
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 학습
history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# 학습 정확도 확인
training_accuracy = history.history['accuracy'][-1]  # 마지막 에포크의 학습 정확도
print(f"Training Accuracy: {training_accuracy:.4f}")  # 학습 정확도 출력

# 5. 성능 평가# 테스트 데이터에 대한 성능 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")  # 테스트 정확도 출력

# 6. 새로운 데이터 예측# 테스트 데이터의 첫 번째 샘플에 대해 예측 수행
predicted = model.predict(x_test[:1])  # 첫 번째 테스트 샘플의 예측값
predicted_label = tf.argmax(predicted, axis=1).numpy()[0]  # 예측된 클래스 (0~9)
actual_label = tf.argmax(y_test[:1], axis=1).numpy()[0]    # 실제 클래스 (0~9)

# 예측값 및 실제값 출력
print(f"Predicted Label: {predicted_label}")  # 모델이 예측한 값
print(f"Actual Label: {actual_label}")        # 실제 정답