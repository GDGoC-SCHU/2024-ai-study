from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


feature, target = load_iris(return_X_y=True)
print("데이터셋 샘플 :")
print([feature[:5]])
print("타깃 샘플 : ")
print([target[:5]])

print(f"특성(shape) : {feature.shape}")
print(f"타깃(shape) : {target.shape}")


X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
print(f"예측 결과: {pred}")

accuracy  = accuracy_score(y_test, pred)
print(f"정확도 점수 : accuracy_score {accuracy :.3f}")



