import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout, Softmax

# 배치 정규화를 포함한 딥러닝 모델 정의 함수
def create_model():
    model = Sequential([
        # TODO: 첫 번째 Dense 층(유닛 수: 256)과 적절한 입력 형태를 추가하세요.
        Dense(256, input_shape=(784,)),

        # TODO: 첫 번째 Dense 층 다음에 배치 정규화를 추가하세요.
        BatchNormalization(),

        # TODO: 첫 번째 배치 정규화 다음에 ReLU 활성화 함수를 추가하세요.
        ReLU(),

        # TODO: 유닛 수가 128인 Dense 층을 추가하세요.
        Dense(128),

        # 해당 층에 배치 정규화와 ReLU를 추가합니다.
        BatchNormalization(),
        ReLU(),

        # TODO: 드롭아웃 층(드롭아웃 비율: 0.3)을 추가하세요.
        Dropout(0.3),

        # TODO: 유닛 수가 64인 Dense 층을 추가하세요.
        Dense(64),

        # 해당 층에 배치 정규화와 ReLU를 추가합니다.
        BatchNormalization(),
        ReLU(),

        # TODO: 유닛 수가 10인 출력층(Dense)과 softmax 활성화 함수를 추가하세요.
        Dense(10),
        Softmax()

    ])
    return model

# MNIST 데이터셋을 로드하고 전처리합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# TODO: 데이터 형태를 (784,)로 변환하고, 0~1 범위로 정규화하세요.
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# TODO: Adam 옵티마이저, sparse categorical crossentropy 손실 함수, accuracy 메트릭으로 모델을 컴파일하세요.
model = create_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# TODO: 배치 크기를 64로, 에포크 수를 5로 설정하여 모델을 훈련하세요.
model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)


# TODO: 모델을 평가하고 테스트 정확도를 출력하세요.
print(f"Test Accuracy: {test_accuracy:.4f}")
