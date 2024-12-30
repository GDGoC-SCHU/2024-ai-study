import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        Dense(256, input_shape=(784,)), # Dense 층 추가
        BatchNormalization(), # 배치 정규화
        ReLU(), # ReLU 활성함수 추가
        Dense(128), # Dense 층 추가
        BatchNormalization(), ReLU(), # 배치 정규화, ReLU 추가
        Dropout(0.3), # 드롭아웃 층 추가
        Dense(64), # Dense 층 추가
        BatchNormalization(), # 배치 정규화
        ReLU(), Dense(10, activation='softmax') # ReLU 추가, 활성함수 softmax인 Dense 층 추가
    ])
    return model

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 형태 변환 후, 0~1 범위로 정규화
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# 모델 생성 후 컴파일
model = create_model()
model.compile(
    optimizer=Adam(),  # Adam 옵티마이저
    loss='sparse_categorical_crossentropy',  # sparse categorical crossentropy 손실 함수
    metrics=['accuracy']  # accuracy 메트릭
)

# 모델 훈련 (배치 크기 64, 에포크 5)
model.fit(x_train, y_train, batch_size=64, epochs=5)

# 모델을 평가 후 테스트 정확도 출력
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")