# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(x_train.shape[0])
        loss_total = 0

        # batch 뽑기
        for start_pos in range(0, x_train.shape[0], batch_size):
            batch_indices = indices[start_pos:start_pos + batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            loss = model.loss(x_batch, y_batch)
            loss_total += loss

            # 배치를 살리면서 gradient를 따로 만들어줘야한다.
            pred = model.forward(x_batch)
            pred[np.arange(pred.shape[0]), y_batch] -= 1
            gradient = pred / pred.shape[0]
            
            model.backward(gradient)
            optimizer.update(model.params, model.grads)

        # 평균 손실 계산
        loss_average = (loss_total + 1e-5) / epoch
        loss_history.append(loss_average)

    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
