# 선형회귀 모델 구현하기
# 텐서플로우의 추상화 접해보기
# 선형회귀란 주어진 x와 y 값을 가지고 서로 간의 관계를 파악하는 것이다
# 이 관계를 알고나면 새로운 x값이 주어졌을 때, y값을 예측할 수 있다.
# 어떤 입력에 대해 출력을 예측하는 것, 이것이 바로 머신러닝의 기본이다.


import tensorflow as tf

x_data=[1,2,3]
y_data=[1,2,3]


# x와 y의 상관관계를 설명하기 위한 변수들인 W와 b를 각각 -1.0부터 1.0사의의 균등분포를 가진 무작위 값으로 초기화

W=tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b=tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X=tf.placeholder(tf.float32, name="X")
Y=tf.placeholder(tf.float32, name="Y")

# W와의 곱과 b와의 합을 통해 X와 Y의 관계를 설명하겠다는 뜻이다
# X가 주어졌을 때, Y를 만들어 낼 수 있는 W와 b를 찾아내겠다는 의미이다.
# W:가중치 (weight), b: 편향(bias)

hypothesis=W*X+b

# 손실함수(loss function)는 한 쌍(x,y)의 데이터에 대한 손실값을 계산하는 함수이다
# 손실값이란 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값이다.
# 손실값이 작을수록 그 모델이 X와 Y의 관계를 잘 설명하고 있다는 뜻이며, 주어진 X값에 대한 Y값을 정확하게 예측할 수 있다는
# 뜻이다.
# 이 손실을 전체 데이터에 대해 구한 경우 비용(cost)라고 한다

# 학습이란, 변수들(W:가중치 (weight), b: 편향(bias))의 값을 다양하게 넣어 계산해보면서 이 손실값을 최소화하는
# W 와 b의 값을 구하는 것이다.

# 손실값으로는 '예측값과 실제값의 거리'를 가장 많이 사용한다.
# 손실값은 예측값에서 실제값을 뺀 뒤 제곱하여 구하며, 그리고, 비용은 모든 데이터에 대한 손실값의 평균을 내어 구한다

cost=tf.reduce_mean(tf.square(hypothesis-Y))

# 텐서플로가 제공하는 경사하강법(Gradient Descent) 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성한다.

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op=optimizer.minimize(cost)

# 최적화 함수란 가중치(W)와 편향(b) 값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치(W)와 편향(b)값을 찾아주는 함수이다
# 가중치(W)와 편향(b)값을 무작위로 변경하면 시간이 너무 오래 걸리고, 학습 시간도 예측하기 어렵다
# 빠르게 최적화하기 위한 방법 중의 하나가 경사하강법이다.
# 경사하강법은 최적화 방법 중 가장 기본적인 알고리즘으로 음의 경사 방향으로 계속 이동하면서 최적의 값을 찾아 나가는 방법이다

# 학습률은 학습을 얼마나 급하게 할 것인가를 설정하는 값이다.
# 학습률이 너무 크면 최적의 손실값을 찾지 못하고 지나치게 되고, 값이 너무 작으면 학습 속도가 매우 느려진다.
# 학습 진행에 영향을 주는 변수를 하이퍼파라미터(hyperparameter)라 하면, 이 값에 따라 학습 속도나 신경망 성능이 크게 달라진다.
# 머신러닝에서는 하이퍼파라미터를 잘 튜닝하는 것이 큰 과제이다.

# 선형회귀모델을 다 만들었으니, 그래프를 실행해 학습을 시키고, 결과를 확인하자
# 파이썬 with 기능을 이용해 세션 블록을 만들고, 세션 종료를 자동으로 처리하자
# 최적화를 수행하는 그래프인 train_op를 실행하고, 실행 시마다 변화하는 손실값을 출력한다
# 학습은 100번 수행하면, feed_dict 매개변수를 통해, 상관관계를 알아내고자 하는 데이터인 x_data와 y_data를 입력한다.


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val=sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))


    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))

