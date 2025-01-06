import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input

# TODO: 저장된 기존 모델('my_model.h5')을 불러오세요.
base_model = load_model('./my_model.h5')

# TODO: 새로운 입력 레이어를 정의하세요. (입력 형태: (784,))
input_layer = Input(shape=(784,))

# TODO: 기존 모델에서 hidden1부터 hidden3까지를 가져와 연결하세요.
hidden1 = base_model.get_layer('hidden1')(input_layer)
hidden2 = base_model.get_layer('hidden2')(hidden1)
hidden3 = base_model.get_layer('hidden3')(hidden2)

# TODO: hidden1과 hidden2 레이어를 동결하세요.
for layer in [base_model.get_layer('hidden1'), base_model.get_layer('hidden2')]:
    layer.trainable = False

# TODO: 새로운 hidden4와 logits 레이어를 추가하여 모델을 확장하세요.
hidden4 = Dense(32, activation='relu', name='hidden4')(hidden3)
logits = Dense(10, activation='softmax', name='logits')(hidden4)

# TODO: 새로운 Transfer Learning 모델을 생성하세요.
transfer_model = Model(inputs=input_layer, outputs=logits)

# TODO: 모델을 컴파일하세요. (옵티마이저: Adam, 손실 함수: Sparse Categorical Crossentropy, 메트릭: Accuracy)
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# TODO: MNIST 데이터셋을 로드하고 전처리하세요. (데이터를 (784,)로 변환 및 정규화)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# TODO: 모델을 학습하세요. (배치 크기: 64, 에포크: 5, 검증 데이터: 20%)
history = transfer_model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)

# TODO: 모델을 평가하고 테스트 정확도를 출력하세요.
test_loss, test_acc = transfer_model.evaluate(x_test, y_test, verbose=0)
print(f"테스트 정확도: {test_acc:.4f}")
