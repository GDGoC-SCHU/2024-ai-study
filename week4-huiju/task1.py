import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout

# 배치 정규화를 포함한 딥러닝 모델 정의 함수
def create_model():
    model = Sequential([
        # 첫 번째 Dense 층(유닛 수: 256)과 적절한 입력 형태를 추가
        Dense(256, input_shape=(784,)),

        # 첫 번째 Dense 층 다음에 배치 정규화를 추가
        BatchNormalization(),

        # 첫 번째 배치 정규화 다음에 ReLU 활성화 함수를 추가
        ReLU(),

        # 유닛 수가 128인 Dense 층을 추가
        Dense(128),

        # 해당 층에 배치 정규화와 ReLU를 추가
        BatchNormalization(),
        ReLU(),

        # 드롭아웃 층(드롭아웃 비율: 0.3)을 추가
        Dropout(0.3),

        # 유닛 수가 64인 Dense 층을 추가
        Dense(64),

        # 해당 층에 배치 정규화와 ReLU를 추가
        BatchNormalization(),
        ReLU(),

        # 유닛 수가 10인 출력층(Dense)과 softmax 활성화 함수를 추가
        Dense(10, activation='softmax')
    ])
    return model

# MNIST 데이터셋을 로드하고 전처리합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 형태를 (784,)로 변환하고, 0~1 범위로 정규화
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 모델 생성 및 컴파일
model = create_model()
model.compile(
    optimizer='adam',  # Adam 옵티마이저
    loss='sparse_categorical_crossentropy', # sparse categorical crossentropy 손실 함수
    metrics=['accuracy'] # accuracy 메트릭
)

# 배치 크기를 64로, 에포크 수를 5로 설정하여 모델 훈련
model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)

# 모델 평가 및 테스트 정확도 출력
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
