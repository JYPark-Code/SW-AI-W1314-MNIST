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
    n_samples = x_train.shape[0]
    # epoch별 평균 loss를 모아두는 리스트 (시각화/디버깅에 사용).
    loss_history = []

    for epoch in range(epochs):
        # 매 epoch마다 데이터 순서를 새로 섞는다.
        # permutation: 새 인덱스 배열을 반환 (원본 보존) → x와 y에 같은 인덱스를 적용해 정답 매칭 유지.
        # shuffle을 따로 두 번 부르면 x와 y의 짝이 깨지므로 절대 금지.
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []
        # 미니배치 루프. start를 batch_size씩 늘리며 슬라이스.
        for start in range(0, n_samples, batch_size):
            x_batch = x_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]
            # 실제 배치 크기. 마지막 배치는 batch_size보다 작을 수 있음.
            # 이 값으로 dout을 나눠야 마지막 배치의 gradient 스케일이 어긋나지 않는다.
            bs = x_batch.shape[0]

            # [1] Forward: 학습 모드(train=True)로 BatchNorm/Dropout 학습 동작 활성화.
            y_pred = model.forward(x_batch, train=True)
            # [2] Loss: backward에는 직접 안 쓰지만 진행 상황 기록용으로 계산.
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_losses.append(loss)

            # [3] Backward: Softmax + CrossEntropy 결합 gradient를 직접 만든다.
            #     수식: dL/dz_i = (y_pred_i - y_true_i) / N
            #     코드: y_pred를 복사한 뒤 정답 위치만 1을 빼고, 전체를 배치 크기로 나눔.
            #     copy()를 안 하면 Softmax가 forward에서 저장한 self.out까지 망가질 수 있다.
            dout = y_pred.copy()
            dout[np.arange(bs), y_batch] -= 1
            dout /= bs
            model.backward(dout)

            # [4] Update: optimizer가 model.params를 in-place 갱신.
            #     이때 각 layer의 self.W 등도 같은 ndarray 참조라 함께 갱신됨.
            optimizer.update(model.params, model.grads)

        # 이번 epoch 전체 배치의 평균 loss를 기록.
        loss_history.append(float(np.mean(epoch_losses)))

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
