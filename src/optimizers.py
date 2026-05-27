# -*- coding: utf-8 -*-
"""파라미터 업데이트 규칙을 모아 둔 optimizer 모듈."""

import numpy as np


class SGD:
    """
    확률적 경사하강법(SGD).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params, grads):
        """params dict의 모든 파라미터를 제자리(in-place)에서 갱신합니다."""
        # 각 파라미터에 대해 θ ← θ - lr * dL/dθ.
        # 중요: `-=`는 in-place 연산이라 ndarray의 값을 직접 수정 →
        # 같은 배열을 참조하는 layer(예: Affine.W)도 자동으로 갱신된다.
        # 만약 `params[key] = params[key] - lr * grads[key]`로 쓰면 새 배열이 생겨
        # dict는 갱신되지만 layer가 들고 있는 참조는 옛 배열 그대로라 학습이 안 됨.
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """

    def __init__(self, lr=0.001):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # 첫 호출 시 m(1차 모멘트), v(2차 모멘트)를 params와 같은 shape의 0으로 초기화.
        # __init__에서는 params의 key/shape를 모르니, 첫 update에서 lazy 초기화.
        if not self.m:
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # t는 호출 횟수(=미니배치 학습 횟수). bias correction 분모 (1 - β^t)에 사용.
        self.t += 1
        # Adam 표준 하이퍼파라미터. 거의 변경하지 않으니 함수 내부에 박아 둠.
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for key in params.keys():
            # m: gradient의 부드러운 평균 (momentum 효과). 노이즈가 덜한 방향 정보를 만듦.
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            # v: gradient 제곱의 평균 (분산 척도). 큰 변동을 가진 파라미터는 더 작게 움직이게.
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
            # bias correction: m, v가 0에서 출발해 초반에 작은 값으로 편향됨 → 보정.
            # t가 클수록 분모가 1에 가까워져 보정 효과가 자연히 사라진다.
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            # 핵심: m_hat을 자기 표준편차 √v_hat로 나눠 파라미터별로 학습률을 자동 조절.
            # eps는 sqrt(v_hat)이 0에 가까울 때 0 나눗셈 방지.
            # in-place 갱신(`-=`) — SGD와 동일한 이유.
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
