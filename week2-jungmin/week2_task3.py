from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# 데이터 로드 및 준비
x, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 결정 트리 하이퍼 파라미터 최적화
model = DecisionTreeClassifier(random_state=42)
gscv = GridSearchCV(model, param_grid={
    'max_depth': range(1, 11),
    'max_leaf_nodes': range(15, 26),
    'min_samples_split': range(1, 5)  
}) # 문제에 따로 범위가 주어져 있지 않아서 임의 부여 
gscv.fit(x_train, y_train)

print("Best Parameters ", gscv.best_params_)

# 테스트 정확도 확인
res = gscv.predict(x_test)
accuracy = accuracy_score(y_test, res)
print("Single Tree Accuracy", accuracy)

# Random Forest
n_samples = len(x_train)
avg_forest = []

for _ in range(100):
    x_sub, y_sub = resample(x_train, y_train, n_samples=n_samples, random_state=42)
    model = DecisionTreeClassifier(max_depth=gscv.best_params_['max_depth'], max_leaf_nodes=gscv.best_params_['max_leaf_nodes'], random_state=42)
    model.fit(x_sub, y_sub)
    res = model.predict(x_test)
    avg_forest.append(accuracy_score(y_test, res))
    
print("Average Single Tree Accuracy ", sum(avg_forest)/100)

# 여러 결정 트리 생성
estimators = [('tree' + str(i), DecisionTreeClassifier(max_depth=gscv.best_params_['max_depth'], max_leaf_nodes=gscv.best_params_['max_leaf_nodes'], random_state=42)) for i in range(100)]
# VotingClassifier를 사용하여 다수결 앙상블 생성
voting_clf = VotingClassifier(estimators=estimators, voting='hard')
voting_clf.fit(x_train, y_train)
# 예측 및 정확도 계산 
y_pred = voting_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble Accuracy ", accuracy)
