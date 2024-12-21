# Task 1

## 의사결정트리(의사결정나무)

> Decision Tree

데이터를 트리 구조로 나눠 예측 구조를 만드는 분류 기법의 알고리즘으로, 각 노드에서 특정 조건으로 데이터를 분할하고 최종 노드에서 결과를 예측한다.

각 트리는 정해진 기준에 따라 가지치기한다. 따라서, 한쪽 가지에 비슷한 클래스가 모여있으면 좋은 모델로 평가한다. 트리 구조로 표현이 가능하므로, 단순하면서도 직관적이고 해석이 쉽다는 장점을 갖는다.

다만, 과적합 발생 가능성이 다소 높다.

## Random Forest

- 여러 개의 결정 트리를 결합한 앙상블 학습 방법.
- 단일 결정 트리에 비해 과적합이 덜하고 분산, 편향을 줄여 일반화 성능을 높인다.
- 앙상블 방식 중 병렬 학습 결과를 다수결 혹은 평균으로 결정하는 배깅(Bagging) 방식을 사용한다.

> 앙상블이란 여러 방식의 모델을 혼합하여 더 좋은 결과를 도모하는 방식으로, 서로 다른 알고리즘으로 예측한 결과를 다수결이나 평균으로 결정하는 Voting, 같은 알고리즘을 사용하면서 앞선 결과로 다음 학습기가 순차 학습해 오차를 줄여나가는 부스팅(Boosting) 방식 등이 있다.

### 작동 원리

- 데이터 샘플링 - 훈련 데이터의 랜덤 서브셋 생성(bootstrap sampling)
- 랜덤성 도입 - 각 트리가 서로 다른 특성 집합 사용
- 다수결 - 모든 트리의 예측 결과를 결합해 최종 예측 (앙상블의 특성)

## 기타

- LabelEncoding은 선형 회귀 등에선 예측 성능이 떨어는 경우가 있어 적용하지 않으나, 이번 예시와 같이 트리 계열의 알고리즘 이용에는 문제가 없다.

# Task 2

# Task 3

# 참고자료

- 2024-1 DSC공유대학 모빌리티 AI/SW 과정 '딥러닝'
- 2024-2 순천향대학교 컴퓨터소프트웨어공학과 '머신러닝'
- [싸이킷런 데이터 전처리 스케일 조정(스케일러)](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=demian7607&logNo=222009975984)
- [Google Colab 데이터 업로드 및 활용](https://stackoverflow.com/questions/52761829/how-do-i-upload-my-own-data-for-tensor-flow-using-google-colab)
- [CSV 파일 가져오는 법(사용하지 않음)](https://ericabae.medium.com/tensorflow-2-0-csv-%ED%8C%8C%EC%9D%BC-%ED%98%95%EC%8B%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0-eddaa88d3112)
- [레이블 없을 때의 `train_test_split`](https://hye0archive.tistory.com/8)
- [Label Encoding](https://velog.io/@hhhs101/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EC%BD%94%EB%94%A9-Label-Encoding-One-hot-Encodingdummies)
- [마지막 열 제외하기](https://velog.io/@hhhs101/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EC%BD%94%EB%94%A9-Label-Encoding-One-hot-Encodingdummies)
- [마지막 열만 선택하기](https://stackoverflow.com/questions/40144769/how-to-select-the-last-column-of-dataframe)